import open3d as o3d
import numpy as np
import glob

view_params = {
    "front" : [ -0.89004986250522844, -0.3103822547697187, 0.33387736997060685 ],
    "lookat" : [ 0.0, 1.0, 0.0 ],
    "up" : [ 0.29416978303035585, 0.16844822209821539, 0.94079186604892795 ],
    "zoom" : 1.3000000000000005,
    "point_show_normal": True
}

if __name__ == '__main__':

    out_dir = "/media/motion/8AF1-B496/warp_data"

    q_fn_lst = sorted(glob.glob(out_dir + '/particle_q_*.npy'))
    f_fn_lst = sorted(glob.glob(out_dir + '/particle_f_*.npy'))

    for q_fn, f_fn in zip(q_fn_lst, f_fn_lst):
        q = np.load(q_fn)
        f = np.load(f_fn)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(q)
        pcd.normals = o3d.utility.Vector3dVector(f)

        o3d.visualization.draw_geometries([pcd], **view_params)