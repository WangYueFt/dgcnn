
import os
import re
import numpy as np
import argparse
import sys
from natsort import natsorted
from plyfile import PlyData, PlyElement
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon

'''
script to evaluate a model

execution example:
    
INPUT
    LOW INSTANCES:  X Y Z R G B C I 
    HIGH PC:        X Y Z R G B
OUTPUT
    HIGH INSTANCES: X Y Z R G B C I

 - python3 project_inst.py --path_in /home/uib/Desktop/test_evaluate_instances/

'''




def write_ply(data, path_out):

    f = open(path_out, 'w')

    f.write("ply" + '\n')
    f.write("format ascii 1.0" + '\n')
    f.write("comment VCGLIB generated" + '\n')
    f.write("element vertex " + str(data.shape[0]) + '\n')
    f.write("property float x" + '\n')
    f.write("property float y" + '\n')
    f.write("property float z" + '\n')
    f.write("property uchar red" + '\n')
    f.write("property uchar green" + '\n')
    f.write("property uchar blue" + '\n')
    f.write("property int class" + '\n')
    f.write("property int inst" + '\n')
    f.write("element face 0" + '\n')
    f.write("property list uchar int vertex_indices" + '\n')
    f.write("end_header" + '\n')

    for row in range(data.shape[0]):
        line = ' '.join(map(str, data[row, :-5])) + ' ' + str(int(data[row, 3])) + ' ' + str(int(data[row, 4])) + ' ' + str(int(data[row, 5])) + ' ' + str(int(data[row, 6])) + ' ' + str(int(data[row, 7])) + '\n'
        f.write(line)

    f.close()


def read_ply(filename, type):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data

    if type == "low":
        pc_array = np.array([[x, y, z, r, g, b, c, i] for x,y,z,r,g,b,c,i in pc])
    if type == "high":
        pc_array = np.array([[x, y, z, r, g, b] for x,y,z,r,g,b,a in pc])

    return pc_array


def project_inst(low, high):

    inst_low_list = list()
    for i in range(len(set(low[..., 7]))):
        inst = low[np.where(low[..., 7] == float(i+1))]
        inst_low_list.append(inst)

    high_list = list()

    for i, inst_low in enumerate(inst_low_list):

        hull = ConvexHull(np.array(inst_low[:, 0:2]))

        hull_points = hull.points
        hull_vertices = hull.vertices
        coords = hull_points[hull_vertices]
        poly = Polygon(coords)

        mins = np.amin(inst_low, axis=0)
        maxs = np.amax(inst_low, axis=0)

        mask = (high[:, 0] > mins[0]) & (high[:, 0] < maxs[0]) & (high[:, 1] > mins[1]) & (high[:, 1] < maxs[1])
        # TODO DAR UN PEQIEÑO MARGEN?
        idx_mask = np.where(mask)
        if mask.size != 0:
            high_sub = high[mask,:]
            high_sub = np.hstack([high_sub,np.array(idx_mask).T])     

        inst_high = list()
        del_list = list()

        for i, p in enumerate(high_sub):
            point = Point(p[0],p[1])
            if point.within(poly) == True:
                p_high = np.hstack([p,inst_low[0,6:]])
                inst_high.append(p_high)
                del_list.append(int(high_sub[i,6])) # append el indice que tenga i en high sub de 6, que es el idice previo a la amscara, para borrar sobre el high grande, sin sub

        high = np.delete(high, del_list, 0)
        inst_high = np.array(inst_high)
        
        if inst_high.shape[0] != 0:
            if mask.size != 0:
                inst_high = np.delete(inst_high,6,1)   # borrar columna mask idx
            high_list.append(inst_high)

    if len(high_list)>0:
        projections = np.vstack(high_list)   # QUITANDO ESTO SE SACA LISTA Y SE PUEDE AHCER VSTACK FUERA
    else:
        print("NO PROJECTIONS FOUND")
        projections = None

    return projections


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='path to the run folder.')
    parsed_args = parser.parse_args(sys.argv[1:])
    path_in = parsed_args.path_in

    files = natsorted(os.listdir(path_in))
    cases = [s for s in files if s.endswith(".ply")]
    names = natsorted(set([re.split("[.\_]+", string)[0] for string in cases]))

    for name in names:

        print("evaluating case: " + name)
        path_low = os.path.join(path_in, name + "_pred_inst_ref.ply")
        path_high = os.path.join(path_in, name + "_high.ply")

        low = read_ply(path_low, "low")
        high = read_ply(path_high, "high")

        projections = project_inst(low, high)

        if projections is not None:
            file_path_out = os.path.join(path_in, name + "_projections.ply")
            write_ply(projections, file_path_out)