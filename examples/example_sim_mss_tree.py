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

import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

import warp.sim
import warp.sim.render
from warp.sim.integrator_euler import compute_forces

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
        self.sim_substeps = 128
        self.sim_duration = 10.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0
        # self.sim_use_graph = wp.get_device().is_cuda
        self.sim_use_graph = False

        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder.default_particle_radius = 0.0001

        # verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        # indices = [0, 1, 2, 1, 2, 3]

        out_geom = o3d.io.read_triangle_mesh("/home/motion/tree_models/sparse_tree_tri_mesh.ply")
        out_geom.compute_vertex_normals()
        out_geom.remove_degenerate_triangles()
        print("number of triangles:", len(out_geom.triangles))
        o3d.visualization.draw_geometries([out_geom])

        verts = np.array(out_geom.vertices)
        print("number of vertices:", verts.shape[0])
        indices = np.array(out_geom.triangles).reshape(-1)

        # builder.add_cloth_mesh(
        #     pos=(0.0, 0.5, 0.0),
        #     rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
        #     scale=1.0,
        #     vel=(0.0, 0.0, 0.0),
        #     vertices=verts,
        #     indices=indices,
        #     density=1.0,
        #     tri_ke=100000.0,
        #     tri_ka=100000.0,
        #     tri_kd=100000.0,
        #     edge_ke=100000.0,
        #     edge_kd=100000.0,
        # )
        
        verts *= 0.5
        verts[:, [1, 2]] = verts[:, [2, 1]]
        verts[:, 1] += 0.5

        for i in range(verts.shape[0]):
            builder.add_particle(pos=verts[i], vel=(0.0, 0.0, 0.0), mass=1000.0, radius=0.001)

        A_mat = kneighbors_graph(verts, n_neighbors=20, include_self=False)
            
        row_counts = A_mat.indptr[1:] - A_mat.indptr[:-1]
        row_idx_lst = np.repeat(np.arange(len(row_counts)), row_counts)
        col_idx_lst = A_mat.indices

        springs_idx_lst = []
        for row_idx, col_idx in zip(row_idx_lst, col_idx_lst):
            if np.linalg.norm(verts[row_idx, :] - verts[col_idx, :]) > 1e-3:
                springs_idx_lst.append([row_idx, col_idx])
                builder.add_spring(row_idx, col_idx, ke=100000.0, kd=1.0, control=0.0)
        print("number of springs:", len(springs_idx_lst))
        
        # for tri in np.array(out_geom.triangles):

        #     if np.linalg.norm(verts[tri[0]] - verts[tri[1]]) > 1e-3:
        #         springs_idx_lst.append([tri[0], tri[1]])
        #         builder.add_spring(tri[0], tri[1], ke=100000.0, kd=1.0, control=0.0)
        #     if np.linalg.norm(verts[tri[1]] - verts[tri[2]]) > 1e-3:
        #         springs_idx_lst.append([tri[1], tri[2]])
        #         builder.add_spring(tri[1], tri[2], ke=100000.0, kd=1.0, control=0.0)
        #     if np.linalg.norm(verts[tri[0]] - verts[tri[2]]) > 1e-3:
        #         springs_idx_lst.append([tri[0], tri[2]])
        #         builder.add_spring(tri[0], tri[2], ke=100000.0, kd=1.0, control=0.0)
            # builder.add_triangle(tri[0], tri[1], tri[2]) 
                                #  tri_ke=100000.0, tri_ka=100000.0, tri_kd=100000.0)
    
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(verts)
        line_set.lines = o3d.utility.Vector2iVector(springs_idx_lst)
        o3d.visualization.draw_geometries([line_set])

        b = builder.add_body(origin=wp.transform((-2.0, 3.5, 0.0), wp.quat_identity()), m=0.0)
        builder.add_shape_sphere(body=b, radius=0.75, density=0.0)

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.model.particle_q.numpy())

        # tri_mesh = o3d.geometry.TriangleMesh()
        # tri_mesh.vertices = o3d.utility.Vector3dVector(self.model.particle_q.numpy())
        # tri_mesh.triangles = o3d.utility.Vector3iVector(self.model.tri_indices.numpy())

        # o3d.visualization.draw_geometries([pcd, tri_mesh])

        self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.XPBDIntegrator(iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        compute_forces(self.model, self.state_0, self.state_0.particle_f, self.state_0.body_f, False)
        # print("particle f:", self.state_0.particle_f.numpy())
        # print("body f:", self.state_0.body_f.numpy())
        # input()

        print("init particle q:", self.state_0.particle_q.numpy())

        self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=1.0)
        # self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=40.0)

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            print("INFO: update")
            if self.sim_time <= 10.0:
                self.state_0.body_q.assign(
                    [[-2.0 + self.sim_time/5.0, 1.5, 0.0, 0., 0., 0., 1.]]
                )

            wp.sim.collide(self.model, self.state_0)

            for s in range(self.sim_substeps):
                self.state_0.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

                # self.clamp_state(self.state_0, self.state_1)

                # print("state diff:", np.linalg.norm(self.state_0.particle_q.numpy() - self.state_1.particle_q.numpy()))

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)
            
            # print("new state:", self.state_0.particle_q.numpy())
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
            # body_q = self.state_0.body_q.numpy()
            # self.renderer.render_line_list()
            # self.renderer.render_sphere(name='sphere', pos=body_q[:3], rot=body_q[3:], radius=0.75)
            # self.renderer.render_points(self, points=self.state_0.particle_q.numpy(), radius=0.01)
            self.renderer.end_frame()

        self.sim_time += 1.0 / self.sim_fps


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_cloth.usd")

    example = Example(stage_path)

    for i in range(example.sim_frames):
        example.update()
        print(f"step {i}: update")

        if np.isnan(example.state_0.particle_q.numpy()).any():
            break

        example.render()
