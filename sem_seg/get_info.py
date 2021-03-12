import os
import re
import sys
import time
import math
import argparse
import numpy as np
from natsort import natsorted
from plyfile import PlyData, PlyElement

'''
script to evaluate a model

execution example:

INPUT
    PC:        X Y Z R G B C
OUTPUT
    INSTANCES: X Y Z R G B C I

 - python3 get_info.py --path_run "path/to/run/" --path_cls "path/to/classes.txt" --test_name name
'''


def get_info_classes(cls_path):

    classes = []
    colors = []

    for line in open(cls_path):
        data = line.split()
        classes.append(data[0])
        colors.append([int(data[1]), int(data[2]), int(data[3])])

    labels = {cls: i for i, cls in enumerate(classes)}

    label2color = {classes.index(cls): colors[classes.index(cls)] for cls in classes}

    return classes, labels, label2color

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, r, g, b, c, i] for x,y,z,r,g,b,c,i in pc])
    return pc_array


def get_distance(p1,p2, dim):
    if dim == 2:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    if dim == 3:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2)+((p2[2]-p1[2])**2))
    return d


def get_match(source, target):

    source_copy = copy.deepcopy(source)

    source_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
    source_copy.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))

    voxel_size = 0.05
    distance_threshold = 0.2 # voxel_size * 1.5
    radius_feature = 0.05 # voxel_size * 5

    _, source_fpfh = preprocess_point_cloud(source_copy, radius_feature)
    result_ransac = execute_global_registration(source_copy, target, source_fpfh, target_fpfh, distance_threshold)
    source_copy.transform(result_ransac.transformation)

    threshold = 0.02
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0], 
                             [0, 0, 0, 1]])

    reg_p2l = o3d.pipelines.registration.registration_icp(source_copy, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())

    transformation = np.matmul(result_ransac.transformation,reg_p2l.transformation)

    return reg_p2l.fitness, transformation

def get_info_skeleton(instances):

    z = 1
    return info


def get_info_matching(instances, models):

    info_list = list()

    for inst in instances:
        match_list = list()

        for model in models:
            match, transform = get_match(inst, model)
            match_list.append(match)
        
        match_max = max(match_list)
        # coger el maximo match de match_lsit
        # ver si es superior al match_thr
        # si -> info: tal instances ha hecho match con tal modelo
        # no -> info: tal instance no ha hecho match con ningun modelo

        
        
    z = 1
    return info_list


def get_info(instances, method, models=0):

    if method == "skeleton":
        info = get_info_skeleton(instances)
    elif method == "matching":
        info = get_info_matching(instances, models)



    return info
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='path in data.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_in = parsed_args.path_in
    path_cls = parsed_args.path_cls  # get class txt path
    classes, labels, label2color = get_info_classes(path_cls)


    for file_name in natsorted(os.listdir(path_in)):

        if "projections" in file_name:

            print("evaluating case: " + file_name)
            path_projections = os.path.join(path_in, file_name)
            projections = read_ply(path_projections)

            instances_pipe = projections[projections[:,6] == [labels["pipe"]]]       # get data label pipe
            instances_valve = projections[projections[:,6] == [labels["valve"]]]     # get data label valve

            instances_pipe_list = list()
            instances_valve_list = list()

            for i in set(instances_pipe[:,7]):
                inst = instances_pipe[instances_pipe[:,7] == i]
                instances_pipe_list.append(inst)

            for i in set(instances_valve[:,7]):
                inst = instances_valve[instances_valve[:,7] == i]
                xyz_central = np.mean(inst, axis=0)[0:3]
                inst[:, 0:3] -= xyz_central                # move instance to origin
                instances_valve_list.append(inst)

            models_vales_list = 0 # MODELS HA DE SER UNA LISTA DE NUMPYS X Y Z R G B DE LOS DIFERENTES TIPOS DE VALVULAS CON QUE SE QUIERA HACER MATCHING

            info_pipe = get_info(instances_pipe_list, method="skeleton")
            info_valve = get_info(instances_valve_list, method="matching", models_vales_list)
