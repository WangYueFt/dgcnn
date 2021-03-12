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

def read_ply(filename, type):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    if type == "proj":
        pc_array = np.array([[x, y, z, r, g, b, c, i] for x,y,z,r,g,b,c,i in pc])
    if type == "model":
        pc_array = np.array([[x, y, z, r, g, b] for x,y,z,r,g,b in pc])
    return pc_array


def get_distance(p1,p2, dim):
    if dim == 2:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    if dim == 3:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2)+((p2[2]-p1[2])**2))
    return d


def match(source, target):

    target_pc = target[0]
    target_fpfh = target[1]
    source_pc = source[0]
    source_fpfh = source[1]
    
    threshold = 0.2

    result_ransac = execute_global_registration(source_pc, target_pc, source_fpfh, target_fpfh, threshold)
    source_pc.transform(result_ransac.transformation)

    threshold = 0.02
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0], 
                             [0, 0, 0, 1]])

    reg_p2l = o3d.pipelines.registration.registration_icp(source_pc, target_pc, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane())

    transformation = np.matmul(result_ransac.transformation,reg_p2l.transformation)

    return reg_p2l.fitness, transformation

def get_info_skeleton(instances):
    z = 1
    return info


def get_info_matching(instances, models):
    info_list = list()
    for inst in instances:
        for model in models:
            fitness, transform = match(inst, model)
            info_list.append([fitness, transform])
    return info_list


def get_info(instances, method, models=0):
    if method == "skeleton":
        info = get_info_skeleton(instances)
    elif method == "matching":
        info = get_info_matching(instances, models)
    return info
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_proj', help='path in projections.')
    parser.add_argument('--path_models', help='path in valve models.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_proj = parsed_args.path_proj
    path_cls = parsed_args.path_cls  # get class txt path
    classes, labels, label2color = get_info_classes(path_cls)

    radius_feature = 0.05

    models_fpfh_list = list()

    for file_name in natsorted(os.listdir(path_models)):
        path_model = os.path.join(path_models, file_name)
        model = read_ply(path_model, "model")

        model_o3d = o3d.geometry.PointCloud()
        model_o3d.points = o3d.utility.Vector3dVector(model[:,0:3])
        model_o3d.colors = o3d.utility.Vector3dVector(model[:,3:6])

        model_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
        model_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))

        _, model_fpfh = preprocess_point_cloud(model_o3d, radius_feature)

        models_fpfh_list.append([model_o3d, model_fpfh])

    for file_name in natsorted(os.listdir(path_proj)):

        if "projections" in file_name:

            print("evaluating case: " + file_name)
            path_projections = os.path.join(path_proj, file_name)
            projections = read_ply(path_projections, "proj")

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

                inst_o3d = o3d.geometry.PointCloud()
                inst_o3d.points = o3d.utility.Vector3dVector(inst[:,0:3])
                inst_o3d.colors = o3d.utility.Vector3dVector(inst[:,3:6])
                inst_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
                inst_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))
                _, inst_fpfh = preprocess_point_cloud(inst_o3d, radius_feature)

                instances_valve_list.append([inst_o3d, inst_fpfh])

            #info_pipe = get_info(instances_pipe_list, method="skeleton")
            info_valve = get_info(instances_valve_list, method="matching", models_fpfh_list)

            # TODO match_max = max(match_list)
            # TODO coger el maximo match de match_list
            # TODO ver si es superior al match_thr
            # TODO si -> info: tal instances ha hecho match con tal modelo
            # TODO no -> info: tal instance no ha hecho match con ningun modelo