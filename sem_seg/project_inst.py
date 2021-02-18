
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

    high_list = list()

    for i in set(low[..., 7]):  # for each isntance low

        inst_low = low[np.where(low[..., 7] == float(i))] # get inst

        hull = ConvexHull(np.array(inst_low[:, 0:2]))   # get its convex hull  TODO que pasa con las tuberias en "L", hace falta reproyectar tuberias?? ....y valvulas?

        hull_points = hull.points                       # get hull points
        hull_vertices = hull.vertices                   # get hull vertices idx
        coords = hull_points[hull_vertices]             # get hull vertices
        poly = Polygon(coords)                          # create a polygon with those vertices

        mins = np.amin(inst_low, axis=0)                # mins of instance
        maxs = np.amax(inst_low, axis=0)                # maxs of instance

        mask = (high[:, 0] > mins[0]) & (high[:, 0] < maxs[0]) & (high[:, 1] > mins[1]) & (high[:, 1] < maxs[1])    # create a square mask around instance TODO MARGEN
        idx_mask = np.where(mask)                                                                                   # get idx of high points inside the mask
        if mask.size != 0:
            high_sub = high[mask,:]                                                                                 # get high points inside the mask (high_sub)
            high_sub = np.hstack([high_sub,np.array(idx_mask).T])                                                   # save original idx

        inst_high = list()
        del_list = list()

        for i, p in enumerate(high_sub):                    # for each point in high sub
            point = Point(p[0],p[1])                        # get point
            if point.within(poly) == True:                  # if its inside the convex hull
                p_high = np.hstack([p,inst_low[0,6:]])      # give the point the class and instance number of inst_low
                inst_high.append(p_high)                    # append to isntance high
                del_list.append(int(high_sub[i,6]))         # annotate original idx

        high = np.delete(high, del_list, 0)                 # delete from high the points that already were found to correspont to an instance
        inst_high = np.array(inst_high)                     # convert inst high to numpy
        
        if inst_high.shape[0] != 0:                         # if inst high contains any points
            if mask.size != 0:                              # if there was a mask
                inst_high = np.delete(inst_high,6,1)        # delete from inst high the original idx column
            high_list.append(inst_high)                     # append inst high

    if len(high_list)>0:
        projections = np.vstack(high_list)  # deleting this line, the ouput becomes a list of numpys
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