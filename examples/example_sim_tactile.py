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
import pathlib
import time

import warp as wp
import warp.sim
import warp.sim.render
import numpy as np

wp.init()

out_dir = "/media/motion/8AF1-B496/warp_data"
out_dir = out_dir + f'/seq_{int(time.time())}'
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

class Example:
    def __init__(self, stage):
        self.sim_width = 8
        self.sim_height = 8

        self.sim_fps = 60.0
        self.sim_substeps = 32
        self.sim_duration = 40.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0

        builder = wp.sim.ModelBuilder()

        builder.add_soft_grid(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, 0.0),
            dim_x=20,
            dim_y=10,
            dim_z=10,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100000.0,
            k_mu=50000.0,
            k_lambda=20000.0,
            k_damp=1000.0,
            fix_bottom=True
        )

        b = builder.add_body(origin=wp.transform((0.5, 2.5, 0.5), wp.quat_identity()), m=0.0)
        builder.add_shape_sphere(body=b, radius=0.75, density=0.0)

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_distance = 0.01
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf = 1.0e3

        print("collision pairs:", self.model.shape_contact_pairs)
        print("gravity:", self.model.gravity)
        print("particle count:", self.model.particle_count)
        print("tet count:", self.model.tet_count)
        print("body mass:", self.model.body_mass)
        print("body count:", self.model.body_count)
        print("body q:", self.model.body_q)

        np.save('outputs/particle_q.npy', self.model.particle_q.numpy())
        self.model.gravity = wp.vec3([0.0, 0.0, 0.0])

        self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.XPBDIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=40.0)

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            print("sim time:", self.sim_time)

            if self.sim_time <= 4.0:
                self.state_0.body_q.assign(
                    [[0.5, 2.5-self.sim_time/5.0, 0.5, 0., 0., 0., 1.]]
                )

            for s in range(self.sim_substeps):
                wp.sim.collide(self.model, self.state_0)
                if s == 0:
                    print("number of contacts:", self.model.soft_contact_count)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)
            
            np.save(out_dir + f'/particle_q_{self.sim_time:07.3f}.npy', self.state_1.particle_q.numpy())
            np.save(out_dir + f'/particle_qd_{self.sim_time:07.3f}.npy', self.state_1.particle_qd.numpy())
            np.save(out_dir + f'/particle_f_{self.sim_time:07.3f}.npy', self.state_1.particle_f.numpy())
            np.save(out_dir + f'/contact_particle_{self.sim_time:07.3f}.npy', self.model.soft_contact_particle.numpy())

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_fem.usd")

    example = Example(stage_path)

    for i in range(example.sim_frames):
        example.update()
        example.render()

    example.renderer.save()
