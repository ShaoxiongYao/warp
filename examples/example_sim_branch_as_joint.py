# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Chain
#
# Shows how to set up a chain of rigid bodies connected by different joint
# types using wp.sim.ModelBuilder(). There is one chain for each joint
# type, including fixed joints which act as a flexible beam.
#
###########################################################################

import numpy as np
import os

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    frame_dt = 1.0 / 100.0

    episode_duration = 20.0  # seconds
    episode_frames = int(episode_duration / frame_dt)

    sim_substeps = 32  # 5
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0
    render_time = 0.0

    def __init__(self, stage=None, render=True):
        self.chain_length = 2
        self.chain_width = 1.0
        self.chain_types = [
            wp.sim.JOINT_REVOLUTE,
            wp.sim.JOINT_BALL,
            wp.sim.JOINT_UNIVERSAL,
            wp.sim.JOINT_COMPOUND,
        ]

        self.enable_rendering = render
        builder = wp.sim.ModelBuilder(gravity=-1.0)

        # start a new articulation
        builder.add_articulation()

        for i in range(self.chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 0.0, 0.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform([self.chain_width, 0.0, 0.0], wp.quat_identity())

            # create body
            b = builder.add_body(origin=wp.transform([i, 0.0, 0.0], wp.quat_identity()), armature=0.1)

            # create shape
            s = builder.add_shape_box(
                pos=(self.chain_width * 0.5, 0.0, 0.0),
                hx=self.chain_width * 0.5,
                hy=0.1,
                hz=0.1,
                density=10.0,
                body=b,
            )

            joint_limit_lower = -np.deg2rad(180.0)
            joint_limit_upper = np.deg2rad(180.0)
            builder.add_joint_revolute(
                parent=parent,
                child=b,
                axis=(0.0, 0.0, 1.0),
                parent_xform=parent_joint_xform,
                child_xform=wp.transform_identity(),
                limit_lower=joint_limit_lower,
                limit_upper=joint_limit_upper,
                target_ke=10.0,
                target_kd=0.0,
                limit_ke=30.0,
                limit_kd=30.0,
            )
        
        builder.add_articulation()
        parent = -1
        b = builder.add_body(origin=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), m=1.0)
        builder.add_shape_sphere(body=b, radius=0.75, density=10.0)
        builder.add_joint_prismatic(
            parent=-1, child=b, parent_xform=wp.transform([1.0, 2.0, 0.0], wp.quat_identity()), 
            child_xform=wp.transform_identity(), axis=(0.0, 1.0, 0.0),
        )

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.model.joint_q = wp.array([0.0, 3.0, 0.0], dtype=wp.float32)

        print("body_q:", self.model.body_q)
        input()

        self.integrator = wp.sim.XPBDIntegrator(iterations=5)

        # -----------------------
        # set up Usd renderer
        self.renderer = None
        if render:
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)

    def update(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        profiler = {}

        # create update graph
        wp.capture_begin()

        # simulate
        self.update()

        graph = wp.capture_end()

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):
            for f in range(0, self.episode_frames):
                with wp.ScopedTimer("simulate", active=True):
                    
                    # if self.sim_time <= 10.0:
                    #     self.model.joint_q = wp.array([0.0, 0.0, -self.sim_time], dtype=wp.float32)
                    #     wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

                        # self.state_0.body_q.assign(
                        #     [[0.,  0.,  1.,  0.,  0.,  0.,  1. ],
                        #      [1.,  0.,  1.,  0.,  0.,  0.,  1. ],
                        #      [0.0, 3.5-self.sim_time/5.0, 0.0, 0., 0., 0., 1.]]
                        # )

                        # self.state_0.body_q[-1, :] = wp.array(
                        #     [0.0, 3.5-self.sim_time/5.0, 0.0, 0., 0., 0., 1.], dtype=wp.float32
                        # )

                    wp.capture_launch(graph)
                self.sim_time += self.frame_dt

                if self.enable_rendering:
                    with wp.ScopedTimer("render", active=True):
                        self.render()

            wp.synchronize()


if __name__ == "__main__":
    stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_chain.usd")
    robot = Example(stage, render=True)
    robot.run()
