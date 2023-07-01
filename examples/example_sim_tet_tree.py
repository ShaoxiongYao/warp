# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid FEM
#
# Shows how to set up a rigid sphere colliding with an FEM beam
# using wp.sim.ModelBuilder().
#
###########################################################################

import os

import math
import warp as wp
import warp.sim
import pyvista as pv
import warp.sim.render
from warp.sim.integrator_euler import compute_forces
import numpy as np
from pxr import Usd, UsdGeom
from touch_utils import compute_contact_forces, TouchSeq

wp.init()
@wp.kernel
def damp_vel_kernel(qd: wp.array(dtype=wp.vec3f), damp: wp.float32):
    i = wp.tid()
    qd[i] = wp.mul(qd[i], damp)

class Example:
    def __init__(self, stage='default.usd', touch_seq: TouchSeq=None):
        self.sim_width = 8
        self.sim_height = 8

        self.sim_fps = 60.0
        self.sim_substeps = 64
        self.sim_duration = 10.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0

        builder = wp.sim.ModelBuilder(gravity=-0.0)
        builder.default_particle_radius = 0.001

        # tet_mesh = pv.read('/home/motion/tet_tree.ply_.msh')
        # # tet_mesh = pv.read('/home/motion/tet_palm_tree_.msh')
        # tet_mesh.plot()

        # points = tet_mesh.points
        # elements = tet_mesh.cells.reshape(-1, 5)[:, 1:]

        points = np.load("/home/motion/tet_tree_fine_points.npy")
        elements = np.load("/home/motion/tet_tree_fine_elements.npy")

        builder.add_soft_mesh(
            pos=(0.0, 0.1, 0.0),
            rot=wp.quat_from_axis_angle([1.0, 0.0, 0.0], -np.pi/2),
            vel=(0.0, 0.0, 0.0),
            scale=1.0,
            vertices=points,
            indices=elements.flatten(),
            density=500.,
            k_mu=100000.0,
            k_lambda=100000.0,
            k_damp=10000.0
        )

        b = builder.add_body(origin=wp.transform((-1.0, 1.5, 0.0), wp.quat_identity()), m=0.0)
        builder.add_shape_sphere(body=b, radius=0.75, density=0.0)

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_margin = 0.01

        # # setup fix points
        pts_ary = self.model.particle_q.numpy()

        min_y = pts_ary[:, 1].min()
        fix_idx_ary = np.where(pts_ary[:, 1] < min_y + 0.1)[0]

        particle_mass = self.model.particle_mass.numpy()
        particle_inv_mass = self.model.particle_inv_mass.numpy()
        particle_mass[fix_idx_ary] = 0.0
        particle_inv_mass[fix_idx_ary] = 0.0
        self.model.particle_mass = wp.array(particle_mass)
        self.model.particle_inv_mass = wp.array(particle_inv_mass)

        print("updated particle mass:", self.model.particle_mass.numpy())

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.integrator = wp.sim.XPBDIntegrator(iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        compute_forces(self.model, self.state_0, self.state_0.particle_f, self.state_0.body_f, False)
        # np.save("particle_f.npy", self.state_0.particle_f.numpy())
        print("particle f:", self.state_0.particle_f.numpy())
        input()

        self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)
        self.touch_seq = touch_seq

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            if self.sim_time <= 10.0:
                self.state_0.body_q.assign(
                    [[-1.0+self.sim_time/8.0, 1.5, 0.0, 0., 0., 0., 1.]]
                )

            for s in range(self.sim_substeps):
                wp.sim.collide(self.model, self.state_0)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # self.damp_vel(self.state_1, damp=0.99)

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)
            
            # tmp_state = self.model.state()
            # tmp_state.particle_q.assign(self.state_0.particle_q)
            # tmp_state.body_q.assign(self.state_0.body_q)

            # self.state_0.clear_forces()
            # self.state_1.clear_forces()

            # NOTE: state_0 current state, state_1 output state
            # compute_contact_forces(self.model, tmp_state, self.state_1)
            
            # self.touch_seq.save(self.sim_time, self.model, self.state_1)

    def damp_vel(self, state, damp):
        wp.launch(
            kernel=damp_vel_kernel,
            dim=state.particle_qd.shape[0],
            inputs=[state.particle_qd, damp],
            device=self.model.device
        )

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

data_keys = ['particle_q', 'particle_qd', 'particle_f', 'body_q', 'shape_transform',
             'contact_particle', 'contact_normal', 'contact_body_pos']

if __name__ == "__main__":

    touch_seq = TouchSeq(data_keys=data_keys)
    example = Example(touch_seq=touch_seq)

    for i in range(example.sim_frames):
        example.update()
        example.render()
    
    example.touch_seq.end_seq()

