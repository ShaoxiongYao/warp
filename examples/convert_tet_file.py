import numpy as np
import glob
import meshio
import os

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

    fail_obj_lst = []

    obj_dir_lst = glob.glob('/home/yaosx/Downloads/dgn_dataset/*')
    for obj_dir in obj_dir_lst:
        obj_name = obj_dir.split('/')[-1]
        print("obj_name:", obj_name)

        in_fn = obj_dir + '/' + obj_name + '.tet'
        out_fn = obj_dir + '/' + obj_name + '.msh'

        print("in_fn:", in_fn)
        print("out_fn:", out_fn)
        
        if os.path.exists(in_fn):
            pts_lst, tet_lst = extract_tet_mesh(in_fn)

            cells = [ ('tetra', tet_lst) ]

            meshio.write_points_cells(out_fn, pts_lst, cells)
        else:
            print(f"INFO: object {obj_name} has no .tet file")
            fail_obj_lst.append(obj_name)
    
    print("failed obj lst:", fail_obj_lst)