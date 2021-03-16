import os
import re
import sys
import time
import copy
import math
import argparse
import numpy as np
import open3d as o3d
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
        pc_array = np.array([[x, y, z, nx, ny ,nz, r, g, b] for x,y,z,nx,ny,nz,r,g,b in pc])
    return pc_array


def get_distance(p1,p2, dim):
    if dim == 2:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    if dim == 3:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2)+((p2[2]-p1[2])**2))
    return d

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, radius_feature):
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    #print("--fpfh")
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def execute_global_registration(source, target, source_fpfh,
                                target_fpfh, distance_threshold):
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def match1(source, target):

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

    #print("--matching: " + str(reg_p2l.fitness))
    #draw_registration_result(source_pc, target_pc, reg_p2l.transformation)

    return reg_p2l.fitness, transformation


def match2(source, target):

    target_pc = target[0]
    source_pc = source[0]
    
    threshold = 0.02
    matchings = list()

    for i in range(16): # 0-15
        trans = np.eye(4)
        trans[:3,:3] = source_pc.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*i))
        reg_p2l = o3d.pipelines.registration.evaluate_registration(source_pc, target_pc, threshold, trans)
        matchings.append(reg_p2l.fitness)
        #print("- matching: " + str(reg_p2l.fitness))
        #draw_registration_result(source_pc, target_pc, trans)
    
    best_idx = matchings.index(max(matchings))
    best_matching = matchings[best_idx]

    #print("-- best matching: " + str(best_matching) + " at angle: " + str((360/16)*(best_idx)))

    trans[:3,:3] = source_pc.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*(best_idx)))
    reg_p2l = o3d.pipelines.registration.evaluate_registration(source_pc, target_pc, threshold, trans)
    #draw_registration_result(source_pc, target_pc, trans)

    return reg_p2l.fitness, (360/16)*(best_idx)

def get_info_skeleton(instances):
    z = 1
    return info


def get_info_matching(instances, models):
    info_list = list()
    for inst in instances:
        info_inst = list()
        for model in models:
            #fitness, transform = match1(inst, model)
            fitness, transform = match2(inst, model)
            info_inst.append([fitness, transform])
        info_list.append(info_inst)
    return info_list


def get_info(instances, models, method):
    if method == "skeleton":
        info = get_info_skeleton(instances)
    elif method == "matching":
        info = get_info_matching(instances, models)
    return info
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_projections', help='path in projections.')
    parser.add_argument('--path_models', help='path in valve models.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_projections = parsed_args.path_projections
    path_models = parsed_args.path_models
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


    for file_name in natsorted(os.listdir(path_projections)):
        
        info_valves_list = list()
        centrals_list = list()
        inst_list = list()

        if "_pred_inst_ref." in file_name:

            print("evaluating case: " + file_name)
            path_projection = os.path.join(path_projections, file_name)
            projection = np.loadtxt(path_projection, usecols=[1,2,3,4,5,6,7,8])
            #print(projection)
            #print(projection.shape)

            proj_o3d = o3d.geometry.PointCloud()
            proj_o3d.points = o3d.utility.Vector3dVector(projection[:,0:3])
            proj_o3d.colors = o3d.utility.Vector3dVector(projection[:,3:6])
            #o3d.visualization.draw_geometries([proj_o3d])

            instances_pipe = projection[projection[:,6] == [labels["pipe"]]]       # get data label pipe
            instances_valve = projection[projection[:,6] == [labels["valve"]]]     # get data label valve

            instances_pipe_list = list()
            instances_valve_list = list()

            for i in set(instances_pipe[:,7]):
                inst = instances_pipe[instances_pipe[:,7] == i]
                instances_pipe_list.append(inst)

            for i in set(instances_valve[:,7]):
                inst = instances_valve[instances_valve[:,7] == i]
                xyz_central = np.mean(inst, axis=0)[0:3]
                centrals_list.append(xyz_central)
                inst[:, 0:3] -= xyz_central                # move instance to origin

                inst_o3d = o3d.geometry.PointCloud()
                inst_o3d.points = o3d.utility.Vector3dVector(inst[:,0:3])
                inst_o3d.colors = o3d.utility.Vector3dVector(inst[:,3:6])
                inst_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
                inst_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))
                _, inst_fpfh = preprocess_point_cloud(inst_o3d, radius_feature)
                inst_list.append(inst_o3d)

                instances_valve_list.append([inst_o3d, inst_fpfh, xyz_central])

            #info_pipe = get_info(instances_pipe_list, models=0, method="skeleton")
            info_valves = get_info(instances_valve_list, models_fpfh_list, method="matching")


            print(info_valves)

            for i, info_valve in enumerate(info_valves):
                max_fitness =  max(info_valve) 
                max_idx = info_valve.index(max_fitness)
                if max_fitness[0] < 0.1:
                    max_fitness[0] = 0
                info_valves_list.append([max_fitness, max_idx])

                central = centrals_list[i]
                inst_o3d = inst_list[i]
                max_model = copy.deepcopy(models_fpfh_list[max_idx][0])

                #max_model_points_np = np.asarray(max_model.points)
                #max_model_points_np[:, 0:3] += xyz_central 
                #max_model.points = o3d.utility.Vector3dVector(max_model_points_np)

                trans = np.eye(4)
                trans[:3,:3] = max_model.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*(max_fitness[1]*(16/360))))
                draw_registration_result(inst_o3d, max_model, trans)

       
            print(" ")
            print(info_valves_list)
            print(" ")
