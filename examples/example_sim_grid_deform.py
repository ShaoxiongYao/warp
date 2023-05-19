# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Neo-Hookean
#
# Shows a simulation of an Neo-Hookean FEM beam being twisted through a
# 180 degree rotation.
#
###########################################################################

import os
import math
import time
from pathlib import Path 

import numpy as np
import warp as wp
from warp.sim.integrator_euler import integrate_particles
import warp.sim.render

wp.init()

@wp.kernel
def eval_linear_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat33),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    Ds = wp.mat33(x10, x20, x30)
    inv_Dm = pose[tid]

    inv_rest_volume = wp.determinant(inv_Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * inv_Dm

    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    I3 = wp.mat33f(1.0, 0.0, 0.0, 
                   0.0, 1.0, 0.0, 
                   0.0, 0.0, 1.0)

    Ic = wp.dot(col1, col1) + wp.dot(col2, col2) + wp.dot(col3, col3)

    P = k_mu*(F + wp.transpose(F) - 2.0*I3) + k_lambda*(Ic - 3.0)*I3

    # deviatoric part
    H = P * wp.transpose(inv_Dm)

    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    f0 = (f1 + f2 + f3) * (0.0 - 1.0)

    # apply forces
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)


class Example:
    def __init__(self, stage, states_path):
        self.sim_width = 8
        self.sim_height = 8

        self.sim_fps = 60.0
        self.sim_substeps = 64
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        self.sim_render = True
        self.sim_iterations = 1
        self.sim_relaxation = 1.0
        self.lift_speed = 2.5 / self.sim_duration * 2.0  # from Smith et al.
        self.rot_speed = math.pi / self.sim_duration

        builder = wp.sim.ModelBuilder()

        cell_dim = 1
        cell_size = 2.0 / cell_dim

        center = cell_size * cell_dim * 0.5

        builder.add_soft_grid(
            pos=(-center, 0.0, -center),
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, 0.0),
            dim_x=cell_dim,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=100.0,
            fix_bottom=True,
            fix_top=True,
            k_mu=1000.0,
            k_lambda=5000.0,
            k_damp=0.0,
        )

        self.model = builder.finalize()
        self.model.ground = False
        self.model.gravity[1] = 0.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.rest = self.model.state()
        self.rest_vol = (cell_size * cell_dim) ** 3

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.volume = wp.zeros(1, dtype=wp.float32)

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=20.0)

        self.step_count = 0
        self.states_path = states_path

        self.save_states(self.state_0, 'init')

    def update(self):
        with wp.ScopedTimer("simulate"):
            xform = wp.transform(
                (0.0, self.lift_speed * self.sim_time, 0.0),
                wp.quat_from_axis_angle((0.0, 1.0, 0.0), self.rot_speed * self.sim_time),
            )
            wp.launch(
                kernel=self.twist_points,
                dim=len(self.state_0.particle_q),
                inputs=[self.rest.particle_q, self.state_0.particle_q, self.model.particle_mass, xform],
            )

            for s in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.state_1.clear_forces()

                wp.launch(
                    kernel=eval_linear_tetrahedra,
                    dim=self.model.tet_count,
                    inputs=[
                        self.state_0.particle_q,
                        self.model.tet_indices,
                        self.model.tet_poses,
                        self.model.tet_materials,
                    ],
                    outputs=[self.state_0.particle_f],
                    device=self.model.device,
                )

                wp.launch(
                    kernel=integrate_particles,
                    dim=len(self.state_0.particle_q),
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.state_0.particle_f,
                        self.model.particle_inv_mass,
                        self.model.gravity,
                        self.sim_dt,
                    ],
                    outputs=[self.state_1.particle_q, self.state_1.particle_qd],
                    device=self.model.device,
                )

                self.sim_time += self.sim_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

            self.save_states(self.state_1, self.step_count)

            self.volume.zero_()
            wp.launch(
                kernel=self.compute_volume,
                dim=self.model.tet_count,
                inputs=[self.state_0.particle_q, self.model.tet_indices, self.volume],
            )
        
            self.step_count += 1

    def render(self, is_live=False):
        with wp.ScopedTimer("render"):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()
    
    def save_states(self, state, suffix):
        particle_q = state.particle_q
        np.save(f'{self.states_path}/particle_q_{suffix}.npy', particle_q.numpy())
        particle_qd = state.particle_qd
        np.save(f'{self.states_path}/particle_qd_{suffix}.npy', particle_qd.numpy())
        particle_f = state.particle_f
        np.save(f'{self.states_path}/particle_f_{suffix}.npy', particle_f.numpy())

    @wp.kernel
    def twist_points(
        rest: wp.array(dtype=wp.vec3), points: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), xform: wp.transform
    ):
        tid = wp.tid()

        r = rest[tid]
        p = points[tid]
        m = mass[tid]

        # twist the top layer of particles in the beam
        if m == 0 and p[1] != 0.0:
            points[tid] = wp.transform_point(xform, r)

    @wp.kernel
    def compute_volume(points: wp.array(dtype=wp.vec3), indices: wp.array2d(dtype=int), volume: wp.array(dtype=float)):
        tid = wp.tid()

        i = indices[tid, 0]
        j = indices[tid, 1]
        k = indices[tid, 2]
        l = indices[tid, 3]

        x0 = points[i]
        x1 = points[j]
        x2 = points[k]
        x3 = points[l]

        x10 = x1 - x0
        x20 = x2 - x0
        x30 = x3 - x0

        v = wp.dot(x10, wp.cross(x20, x30)) / 6.0

        wp.atomic_add(volume, 0, v)


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_neo_hookean.usd")
    time_id = int(time.time())
    states_path = os.path.join(os.path.dirname(__file__), f"outputs/box_deform_{time_id}")
    Path(states_path).mkdir(parents=True, exist_ok=True)

    example = Example(stage_path, states_path)

    for i in range(example.sim_frames):
        example.update()
        example.render()

    example.renderer.save()
