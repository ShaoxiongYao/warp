import open3d as o3d
import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from touch_utils import TouchSeq

view_params = {    
    "front" : [ 0.52760563961593143, 0.48244076633819, 0.6992018278154627 ],
    "lookat" : [ 0.79439834992880953, 0.34262569417500893, -0.098711468081005935 ],
    "up" : [ -0.30336364405820648, 0.87581582927351098, -0.37538930812914506 ],
    "zoom" : 0.96000000000000019,
    "point_show_normal": True
}

state_keys = ['particle_q', 'particle_qd', 'particle_f', 'body_q', 'shape_transform']
contact_keys = ['contact_particle', 'contact_normal', 'contact_body_pos']

if __name__ == '__main__':
    touch_seq = TouchSeq(seq_id='1687970050', data_keys=state_keys+contact_keys)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    sim_time_lst = touch_seq.sim_time_lst

    prev_qd = None
    change_qd_lst = []

    tot_f_lst = []

    # mesh = o3d.io.read_triangle_mesh('/home/motion/visual-tactile-model/assets/toy_bird_trimesh.ply')
    triangles = np.load('outputs/toy_bird_triangles.npy')
    fix_idx_ary = np.load('outputs/toy_bird_fix_idx_ary.npy')

    for idx in range(len(touch_seq)):

        sim_time, q, qd, f, body_q, body_sb = touch_seq.load_group(idx, state_keys)
        _, cid, cn, cbp = touch_seq.load_group(idx, contact_keys)

        if prev_qd is not None:
            change_qd = np.linalg.norm(qd-prev_qd)
            print("change in qd:", change_qd)
            change_qd_lst.append(change_qd)
        prev_qd = qd

        nz_cid = cid[cid != 0]

        if sim_time >= 0.0:
            if nz_cid.shape[0] != 0:
                tot_f_lst.append(f[nz_cid, :].sum(axis=0))
            else:
                tot_f_lst.append(np.zeros(3))

        if idx % 100 == -1:
            body_pcd = o3d.geometry.PointCloud()
            body_pcd.points = o3d.utility.Vector3dVector(cbp[cid != 0, :] + body_q[0, :3])

            print("pts pen:", np.linalg.norm(cbp[cid != 0, :]-q[nz_cid, :], axis=1))

            dr = cbp[cid != 0, :]-q[nz_cid, :]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(q)
            normals = np.zeros((q.shape[0], 3))
            normals[nz_cid, :] = -cn[cid != 0, :]
            print("nz cn:", np.linalg.norm(cn[cid != 0, :], axis=1))
            pcd.normals = o3d.utility.Vector3dVector(normals)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(q)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()

            colors = np.zeros((q.shape[0], 3))
            colors[:, :] = [0.0, 0.0, 1.0]
            colors[nz_cid, :] = [1.0, 0.0, 0.0]
            colors[fix_idx_ary, :] = [0.0, 1.0, 0.0]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # o3d.visualization.draw_geometries([body_pcd, coord_frame])
            o3d.visualization.draw_geometries([mesh, coord_frame, body_pcd, pcd], **view_params)
    
    plt.plot(sim_time_lst[1:], change_qd_lst, label=r'$\Delta ||v||$')
    plt.legend()
    plt.show()

    for i, n in enumerate(['x', 'y', 'z']):
        plt.plot(sim_time_lst[-len(tot_f_lst):],
                 [tot_f[i] for tot_f in tot_f_lst], label=f'f{n}')
    plt.xlabel('simulation time (s)')
    plt.ylabel('total force (N)')
    # plt.savefig('outputs/contact_toy_forces.png')
    plt.legend()
    plt.show()