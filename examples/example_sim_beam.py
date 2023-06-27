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

import warp as wp
import warp.sim
import warp.sim.render
import numpy as np
from touch_utils import compute_contact_forces, TouchSeq

wp.init()

@wp.kernel
def damp_vel_kernel(qd: wp.array(dtype=wp.vec3f), damp: wp.float32):
    i = wp.tid()
    qd[i] = wp.mul(qd[i], damp)

config = {
    'sim_substeps': 32,
    'sim_duration': 20.0,
    'density': 100.0,
    'k_mu': 100000.0,
    'k_lambda': 100000.0,
    'k_damp': 10000.0,
    'integrator': 'XPBD',
    'soft_contact_ke': 1.0e3,
    'soft_contact_kd': 10.0,    
    'soft_contact_kf': 1.0e3,
    'soft_contact_restitution': 0.1,
    'vel_damp': 1.0
}

class Example:
    def __init__(self, stage, touch_seq: TouchSeq):
        self.sim_width = 8
        self.sim_height = 8

        self.sim_fps = 60.0
        self.sim_substeps = config['sim_substeps']
        self.sim_duration = config['sim_duration']
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0

        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder.default_particle_radius = 0.01

        builder.add_soft_grid(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, 0.0),
            dim_x=10,
            dim_y=40,
            dim_z=10,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=config['density'],
            k_mu=config['k_mu'],
            k_lambda=config['k_lambda'],
            k_damp=config['k_damp'],
            fix_bottom=True
        )

        b = builder.add_body(origin=wp.transform((0.5, 2.5, 0.5), wp.quat_identity()), m=0.0)
        builder.add_shape_sphere(body=b, radius=0.75, density=0.0)

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = config['soft_contact_ke']
        self.model.soft_contact_kd = config['soft_contact_kd']
        self.model.soft_contact_kf = config['soft_contact_kf']
        self.model.soft_contact_restitution = config['soft_contact_restitution']

        if config['integrator'] == 'semi_implicit':
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif config['integrator'] == 'XPBD':
            self.integrator = wp.sim.XPBDIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)
        self.touch_seq = touch_seq

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            if self.sim_time <= 10.0:
                self.state_0.body_q.assign(
                    [[-1.0+self.sim_time/20.0, 2.5, 0.25, 0., 0., 0., 1.]]
                )

            for s in range(self.sim_substeps):
                wp.sim.collide(self.model, self.state_0)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                self.damp_vel(self.state_1, damp=0.99)

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)
            
            # NOTE: state_0 current state, state_1 output state
            compute_contact_forces(self.model, self.state_0, self.state_1)
            
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
    
    def close(self):
        self.touch_seq.end_seq(config=config)


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_fem.usd")

    touch_seq = TouchSeq()
    example = Example(stage_path, touch_seq)

    for i in range(example.sim_frames):
        example.update()
        example.render()
    
    example.close()
