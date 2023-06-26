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


if __name__ == '__main__':
    touch_seq = TouchSeq(seq_id='1687751055')

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    sim_time_lst = touch_seq.sim_time_lst

    prev_qd = None
    change_qd_lst = []

    tot_f_lst = []


    for idx in range(len(touch_seq)):

        sim_time, q, qd, f, cid = touch_seq.load(idx)

        if prev_qd is not None:
            change_qd = np.linalg.norm(qd-prev_qd)
            print("change in qd:", change_qd)
            change_qd_lst.append(change_qd)
        prev_qd = qd

        nz_cid = cid[cid != 0]
        print("number of contacts:", nz_cid.shape[0])

        if sim_time >= 0.0:
            if nz_cid.shape[0] != 0:
                tot_f_lst.append(f[nz_cid, :].sum(axis=0))
            else:
                tot_f_lst.append(np.zeros(3))

        # if idx % 30 == 0:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(q)
        #     pcd.normals = o3d.utility.Vector3dVector(1*f)

        #     colors = np.zeros((q.shape[0], 3))
        #     colors[:, :] = [0.0, 0.0, 1.0]
        #     colors[nz_cid, :] = [1.0, 0.0, 0.0]
        #     pcd.colors = o3d.utility.Vector3dVector(colors)

        #     o3d.visualization.draw_geometries([coord_frame, pcd], **view_params)
    
    plt.plot(sim_time_lst[1:], change_qd_lst, label=r'$\Delta ||v||$')
    plt.legend()
    plt.show()

    for i, n in enumerate(['x', 'y', 'z']):
        plt.plot(sim_time_lst[-len(tot_f_lst):],
                 [tot_f[i] for tot_f in tot_f_lst], label=f'f{n}')
    plt.xlabel('simulation time (s)')
    plt.ylabel('total force (N)')
    plt.legend()
    plt.show()