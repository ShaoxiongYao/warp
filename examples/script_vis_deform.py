import open3d as o3d
import numpy as np
import glob
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from touch_utils import TouchSeq

# view_params = {    
#     "front" : [ 0.52760563961593143, 0.48244076633819, 0.6992018278154627 ],
#     "lookat" : [ 0.79439834992880953, 0.34262569417500893, -0.098711468081005935 ],
#     "up" : [ -0.30336364405820648, 0.87581582927351098, -0.37538930812914506 ],
#     "zoom" : 0.96000000000000019,
#     "point_show_normal": True
# }

view_params = {	
    "front" : [ -0.1469213537001246, 0.26014213613068427, 0.95432708482798889 ],
    "lookat" : [ 1.0014895121060376, 1.3375622434324592, -0.44790526694035909 ],
    "up" : [ -0.041985972165301158, 0.96228722875000672, -0.26877586857076091 ],
    "zoom" : 0.65999999999999992,
    "point_show_normal": True
}

state_keys = ['particle_q', 'particle_qd', 'particle_f', 'body_q']
contact_keys = ['contact_particle', 'contact_normal', 'contact_body_pos']

if __name__ == '__main__':
    touch_seq = TouchSeq(seq_id='1688478776', data_keys=state_keys)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    sim_time_lst = touch_seq.sim_time_lst

    prev_qd = None
    change_qd_lst = []

    tot_f_lst = []

    # triangles = np.load('outputs/toy_bird_triangles.npy')
    # fix_idx_ary = np.load('outputs/toy_bird_fix_idx_ary.npy')

    for idx in range(len(touch_seq)):
        print("step index:", idx)

        start_time = time.time()
        sim_time, q, qd, f, body_q, body_sb = touch_seq.load_group(idx, group_keys=state_keys)
        _, cid, cn, cbp = touch_seq.load_group(idx, group_keys=contact_keys)
        print("load time:", time.time()-start_time)

        if prev_qd is not None:
            change_qd = np.linalg.norm(qd-prev_qd)
            change_qd_lst.append(change_qd)
        prev_qd = qd

        nz_cid = cid[cid != 0]
        print("number of contacts:", nz_cid.shape[0])

        nz_mask = np.zeros(q.shape[0], dtype=bool)
        nz_mask[nz_cid] = True
        # assert np.allclose(f[~nz_mask, :], 0.0)

        if sim_time >= 0.0:
            if nz_cid.shape[0] != 0:
                tot_f_lst.append(f[nz_cid, :].sum(axis=0))
            else:
                tot_f_lst.append(np.zeros(3))

        if idx % 50 == -1:
            print("body_q:", body_q)
            body_pcd = o3d.geometry.PointCloud()
            body_pcd.points = o3d.utility.Vector3dVector(cbp[cid != 0, :] + body_q[0, :3])

            dr = cbp[cid != 0, :]-q[nz_cid, :]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(q)
            # normals = np.zeros((q.shape[0], 3))
            # normals[nz_cid, :] = -cn[cid != 0, :]
            # pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.normals = o3d.utility.Vector3dVector(0.01*f)

            # mesh = o3d.geometry.TriangleMesh()
            # mesh.vertices = o3d.utility.Vector3dVector(q)
            # mesh.triangles = o3d.utility.Vector3iVector(triangles)
            # mesh.compute_vertex_normals()

            # NOTE: colorize fixed points
            # colors = np.zeros((q.shape[0], 3))
            # colors[:, :] = [0.0, 0.0, 1.0]
            # colors[nz_cid, :] = [1.0, 0.0, 0.0]
            # colors[fix_idx_ary, :] = [0.0, 1.0, 0.0]
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # o3d.visualization.draw_geometries([body_pcd, coord_frame])
            o3d.visualization.draw_geometries([coord_frame, body_pcd, pcd], **view_params)
    
    plt.plot(sim_time_lst[1:], change_qd_lst, label=r'$\Delta ||v||$')
    plt.legend()
    plt.show()

    for i, n in enumerate(['x', 'y', 'z']):
        plt.plot(sim_time_lst[-len(tot_f_lst):],
                 [tot_f[i] for tot_f in tot_f_lst], label=f'f{n}')
    plt.xlabel('simulation time (s)')
    plt.ylabel('total force (N)')
    plt.savefig('outputs/contact_forces_touch1.png')
    plt.legend()
    plt.show()