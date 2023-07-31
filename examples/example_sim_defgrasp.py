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
        self.sim_substeps = 128
        self.sim_duration = 10.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0

        builder = wp.sim.ModelBuilder(gravity=-10.0)
        builder.default_particle_radius = 0.01

        # points:np.ndarray = np.load("/home/motion/visual-tactile-model/assets/toy_bird_points.npy")
        # points -= np.mean(points, axis=0)
        # elements:np.ndarray = np.load("/home/motion/visual-tactile-model/assets/toy_bird_elements.npy")

        tet_mesh = pv.read('/home/yaosx/Downloads/dgn_dataset/apple1/apple1.msh')

        tet_mesh.plot()

        points = tet_mesh.points
        # points -= np.mean(points, axis=0) 
        # points += np.array([0.0, 0.0, 1.0])
        elements = tet_mesh.cells.reshape(-1, 5)[:, 1:]

        print("number of points:", points.shape)
        print("number of elements:", elements.shape)

        print("points max:", points.max(axis=0))
        print("points min:", points.min(axis=0))

        builder.add_soft_mesh(
            pos=(0.0, 0.01, 0.0),
            rot=wp.quat_from_axis_angle([1.0, 0.0, 0.0], -np.pi/2),
            vel=(0.0, 0.0, 0.0),
            scale=10.0,
            vertices=points,
            indices=elements.flatten(),
            density=100.,
            k_mu=5000.0,
            k_lambda=4000.0,
            k_damp=200.0
        )

        b = builder.add_body(origin=wp.transform((0.0, 10.0, 0.0), wp.quat_identity()), m=0.0)
        builder.add_shape_sphere(body=b, radius=0.75, density=0.0)

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_margin = 0.01

        # setup fix points
        pts_ary = self.model.particle_q.numpy()

        min_y = pts_ary[:, 1].min()
        max_y = pts_ary[:, 1].max()
        fix_idx_ary = np.where(pts_ary[:, 1] < min_y + 0.05)[0]

        self.init_y = max_y + 0.75 + 0.1

        particle_mass = self.model.particle_mass.numpy()
        particle_inv_mass = self.model.particle_inv_mass.numpy()
        particle_mass[fix_idx_ary] = 0.0
        particle_inv_mass[fix_idx_ary] = 0.0
        self.model.particle_mass = wp.array(particle_mass)
        self.model.particle_inv_mass = wp.array(particle_inv_mass)

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.integrator = wp.sim.XPBDIntegrator(iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)
        self.touch_seq = touch_seq

        self.use_capture_graph = True

        if self.use_capture_graph:
            wp.capture_begin()

            for s in range(self.sim_substeps):
                wp.sim.collide(self.model, self.state_0)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # self.damp_vel(self.state_1, damp=1.0)

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

            self.graph = wp.capture_end()

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            if self.sim_time <= 10.0:
                self.state_0.body_q.assign(
                    [[0.0, self.init_y-self.sim_time/10.0, 0.0, 0., 0., 0., 1.]]
                )

            if self.use_capture_graph:
                wp.capture_launch(self.graph)
                self.sim_time += self.sim_dt*self.sim_substeps

            else:
                for s in range(self.sim_substeps):
                    wp.sim.collide(self.model, self.state_0)

                    self.state_0.clear_forces()
                    self.state_1.clear_forces()

                    self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

                    # self.damp_vel(self.state_1, damp=1.0)

                    # swap states
                    (self.state_0, self.state_1) = (self.state_1, self.state_0)
            
            tmp_state = self.model.state()
            tmp_state.particle_q.assign(self.state_0.particle_q)
            tmp_state.body_q.assign(self.state_0.body_q)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # NOTE: state_0 current state, state_1 output state
            compute_contact_forces(self.model, tmp_state, self.state_1)
            
            self.touch_seq.save(self.sim_time, self.model, self.state_1)

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

    touch_seq = TouchSeq(seq_dir="/home/yaosx/sim_data", data_keys=data_keys)
    example = Example(touch_seq=touch_seq)

    for i in range(example.sim_frames):
        example.update()
        example.render()
    
    example.touch_seq.end_seq()

