import open3d as o3d
import numpy as np
import glob

view_params = {    
    "front" : [ 0.52760563961593143, 0.48244076633819, 0.6992018278154627 ],
    "lookat" : [ 0.79439834992880953, 0.34262569417500893, -0.098711468081005935 ],
    "up" : [ -0.30336364405820648, 0.87581582927351098, -0.37538930812914506 ],
    "zoom" : 0.96000000000000019,
    "point_show_normal": True
}

if __name__ == '__main__':

    out_dir = "/media/motion/8AF1-B496/warp_data"

    q_fn_lst = sorted(glob.glob(out_dir + '/particle_q_*.npy'))
    f_fn_lst = sorted(glob.glob(out_dir + '/particle_f_*.npy'))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    for idx, (q_fn, f_fn) in enumerate(zip(q_fn_lst, f_fn_lst)):
        sim_time = float(q_fn.split('/')[-1][len('particle_q_'):-len('.npy')])
        print("sim time:", sim_time)
        print("idx:", idx)

        q = np.load(q_fn)
        f = np.load(f_fn)

        if idx % 30 == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(q)
            pcd.normals = o3d.utility.Vector3dVector(0.01*f)
            pcd.paint_uniform_color([0.1, 0.1, 0.7])

            o3d.visualization.draw_geometries([coord_frame, pcd], **view_params)