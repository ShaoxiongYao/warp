import open3d as o3d
import numpy as np
import glob
import matplotlib
import time
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
    touch_seq1 = TouchSeq(seq_id='1687710626')
    touch_seq2 = TouchSeq(seq_id='1687708155')

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    tot_f_lst = []
    sim_time_lst = touch_seq1.sim_time_lst

    q1_abs_lst = []
    q2_abs_lst = []
    q_diff_lst = []

    q1_0, q2_0 = None, None

    assert len(touch_seq1) == len(touch_seq2)
    num_obs = len(touch_seq1)
    for idx in range(num_obs):
        print("idx:", idx)

        start_time = time.time()
        sim_time, q1, qd1, f1, cid1 = touch_seq1.load(idx)
        sim_time, q2, qd2, f2, cid2 = touch_seq2.load(idx)

        if idx == 0:
            q1_0 = q1.copy()
            q2_0 = q2.copy()
        print("load time:", time.time() - start_time)

        q1_abs_lst.append(np.linalg.norm(q1-q1_0))
        q2_abs_lst.append(np.linalg.norm(q2-q2_0))
        q_diff_lst.append(np.linalg.norm(q1 - q2))

    plt.plot(sim_time_lst, q1_abs_lst, label='u1 abs')
    plt.plot(sim_time_lst, q2_abs_lst, label='u2 abs')
    plt.plot(sim_time_lst, q_diff_lst, label='norm u1-u2')
    plt.xlabel('simulation time (s)')
    plt.ylabel('q diff (m)')
    plt.legend()
    plt.show()