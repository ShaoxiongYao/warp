import numpy as np
import glob
import meshio

def extract_tet_mesh(in_fn):
    pts_lst = []
    tet_lst = []

    with open(in_fn, 'r') as in_f:
        for line in in_f:
            line = line.rstrip()
            if len(line) == 0:
                continue

            l = line.split()
            if l[0] == 'v':
                # print("vertex line:", line)
                pts_lst.append([float(num) for num in l[1:-1]])
            if l[0] == 't':
                # print("tetrahedron line:", line)
                tet_lst.append([int(num) for num in l[1:]])

    return pts_lst, tet_lst

if __name__ == '__main__':

    obj_dir_lst = glob.glob('/home/yaosx/Downloads/dgn_dataset/*')
    for obj_dir in obj_dir_lst:
        print("obj_dir:", obj_dir)


        # pts_lst, tet_lst = extract_tet_mesh(in_fn)

        # cells = [
        #     ('tetra', tet_lst)
        # ]

        # meshio.write_points_cells(out_fn, pts_lst, cells)
