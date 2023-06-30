# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import open3d as o3d

import warp.sim
import warp.sim.render

from pxr import Usd, UsdGeom

wp.init()

@wp.kernel
def clamp_update(prev_ary: wp.array(dtype=wp.vec3f), curr_ary: wp.array(dtype=wp.vec3f)):
    i = wp.tid()

    prev = prev_ary[i]
    curr = curr_ary[i]

    min_v = wp.vec3f(-0.001, -0.001, -0.001)
    max_v = wp.vec3f( 0.001,  0.001,  0.001)

    out = wp.min(curr-prev, min_v)
    out = prev + wp.max(out, max_v)

    curr_ary[i] = out


class Example:
    def __init__(self, stage):
        self.sim_width = 64
        self.sim_height = 32

        self.sim_fps = 60.0
        self.sim_substeps = 32
        self.sim_duration = 10.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        # self.sim_use_graph = wp.get_device().is_cuda
        self.sim_use_graph = False

        builder = wp.sim.ModelBuilder(gravity=-0.1)
        builder.default_particle_radius = 0.0001

        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        indices = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]

        out_geom = o3d.io.read_triangle_mesh("/home/motion/sparse_tree_tri_mesh.ply")
        out_geom.compute_vertex_normals()
        out_geom.remove_degenerate_triangles()
        print("number of triangles:", len(out_geom.triangles))
        o3d.visualization.draw_geometries([out_geom])

        verts = np.array(out_geom.vertices)
        print("number of vertices:", verts.shape[0])
        indices = np.array(out_geom.triangles).reshape(-1)

        mesh = wp.sim.Mesh(verts, indices)

        # builder.add_shape_mesh(
        #     body=-1,
        #     mesh=mesh,
        #     pos=(1.0, 0.0, 1.0),
        #     rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
        #     scale=(1.0, 1.0, 1.0),
        #     ke=1.0e2,
        #     kd=1.0e2,
        #     kf=1.0e1,
        # )

        builder.add_cloth_mesh(
            pos=(0.0, 0.5, 0.0),
            rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=verts,
            indices=indices,
            density=1.0
        )

        # builder.add_cloth_grid(
        #     pos=(0.0, 10.0, 0.0),
        #     rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.5),
        #     vel=(0.0, 0.0, 0.0),
        #     dim_x=16,
        #     dim_y=8,
        #     cell_x=0.4,
        #     cell_y=0.4,
        #     mass=0.1,
        #     fix_left=True,
        #     tri_ke=1.0e3,
        #     tri_ka=1.0e3,
        #     tri_kd=1.0e1,
        # )

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.model.particle_q.numpy())

        tri_mesh = o3d.geometry.TriangleMesh()
        tri_mesh.vertices = o3d.utility.Vector3dVector(self.model.particle_q.numpy())
        tri_mesh.triangles = o3d.utility.Vector3iVector(self.model.tri_indices.numpy())

        o3d.visualization.draw_geometries([pcd, tri_mesh])

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.integrator = wp.sim.XPBDIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        print("init particle q:", self.state_0.particle_q.numpy())

        self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)
        # self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=40.0)

        if self.sim_use_graph:
            # create update graph
            wp.capture_begin()

            wp.sim.collide(self.model, self.state_0)

            for s in range(self.sim_substeps):
                print("s:", s)
                self.state_0.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

            self.graph = wp.capture_end()

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            if self.sim_use_graph:
                wp.capture_launch(self.graph)
            else:
                print("INFO: update")
                wp.sim.collide(self.model, self.state_0)

                for s in range(self.sim_substeps):
                    self.state_0.clear_forces()

                    self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

                    # self.clamp_state(self.state_0, self.state_1)

                    print("state diff:", np.linalg.norm(self.state_0.particle_q.numpy() - self.state_1.particle_q.numpy()))

                    # swap states
                    (self.state_0, self.state_1) = (self.state_1, self.state_0)
                
                print("new state:", self.state_0.particle_q.numpy())
                # input()

    def clamp_state(self, state0, state1):
        wp.launch(
            kernel=clamp_update,
            dim=state0.particle_q.shape[0],
            inputs=[state0.particle_q, state1.particle_q],
            device=self.model.device
        )

        wp.launch(
            kernel=clamp_update,
            dim=state0.particle_qd.shape[0],
            inputs=[state0.particle_qd, state1.particle_qd],
            device=self.model.device
        )

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

        self.sim_time += 1.0 / self.sim_fps


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_cloth.usd")

    example = Example(stage_path)

    for i in range(example.sim_frames):
        example.update()

        if np.isnan(example.state_0.particle_q.numpy()).any():
            break

        example.render()
