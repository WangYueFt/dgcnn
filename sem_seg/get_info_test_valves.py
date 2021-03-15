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

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
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
    print("--fpfh")
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

    print("--matching: " + str(reg_p2l.fitness))

    draw_registration_result(source_pc, target_pc, reg_p2l.transformation)

    return reg_p2l.fitness, transformation


def match2(source, target):

    target_pc = target[0]
    target_fpfh = target[1]
    source_pc = source[0]
    source_fpfh = source[1]
    
    threshold = 0.02
    matchings = list()

    for i in range(16): # 0-15
        k = i+1
        trans = np.eye(4)
        start = time.time()
        trans[:3,:3] = source_pc.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*k))
        mid = time.time()
        reg_p2l = o3d.pipelines.registration.evaluate_registration(source_pc, target_pc, threshold, trans)
        end = time.time()
        matchings.append(reg_p2l.fitness)
        #print("Registration check 1/16 took %.5f sec. (%.5f - %.5f)" % (end-start, mid-start, end-mid))
        #print("- matching: " + str(reg_p2l.fitness))
        #draw_registration_result(source_pc, target_pc, trans)
    
    best_idx = matchings.index(max(matchings))
    best_matching = matchings[best_idx]

    print("-- best matching: " + str(best_matching) + " at angle: " + str((360/16)*(best_idx+1)))

    trans[:3,:3] = source_pc.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*(best_idx+1)))
    reg_p2l = o3d.pipelines.registration.evaluate_registration(source_pc, target_pc, threshold, trans)
    #draw_registration_result(source_pc, target_pc, trans)

    return reg_p2l.fitness, trans

def get_info_skeleton(instances):
    z = 1
    return info


def get_info_matching(instances, models):
    info_list = list()
    for inst in instances:
        start1 = time.time()
        for model in models:
            #fitness, transform = match1(inst, model)
            start2 = time.time()
            fitness, transform = match2(inst, model)
            end2 = time.time()
            print("Registration check 360 took %.5f sec." % (end2-start2))
            info_list.append([fitness, transform])
        end1 = time.time()
        print("Registration check vs all took %.5f sec." % (end1-start1))

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
    parsed_args = parser.parse_args(sys.argv[1:])

    path_projections = parsed_args.path_projections
    path_models = parsed_args.path_models

    radius_feature = 0.05

    models_fpfh_list = list()
    print("targets")

    for file_name in natsorted(os.listdir(path_models)):
        print("--" + file_name)
        path_model = os.path.join(path_models, file_name)
        model = read_ply(path_model)

        model_o3d = o3d.geometry.PointCloud()
        model_o3d.points = o3d.utility.Vector3dVector(model[:,0:3])
        model_o3d.colors = o3d.utility.Vector3dVector(model[:,3:6])

        model_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
        model_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))

        _, model_fpfh = preprocess_point_cloud(model_o3d, radius_feature)

        models_fpfh_list.append([model_o3d, model_fpfh])


    info = list()
    print("\n")
    print("sources")

    for file_name in natsorted(os.listdir(path_projections)):
        
        projection_fpfh_list = list()

        print("evaluating case: " + file_name)
        path_projection = os.path.join(path_projections, file_name)
        projection = read_ply(path_projection)

        projection_o3d = o3d.geometry.PointCloud()
        projection_o3d.points = o3d.utility.Vector3dVector(projection[:,0:3])
        projection_o3d.colors = o3d.utility.Vector3dVector(projection[:,3:6])
        projection_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
        projection_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))
        _, projection_fpfh = preprocess_point_cloud(projection_o3d, radius_feature)

        projection_fpfh_list.append([projection_o3d, projection_fpfh])

        info_projection = get_info(projection_fpfh_list, models_fpfh_list, method="matching")

        #print(info_projection)

        info.append(info_projection)
        print(" ")

        #print(info_valve)
        # TODO match_max = max(match_list)
        # TODO coger el maximo match de match_list
        # TODO ver si es superior al match_thr
        # TODO si -> info: tal instances ha hecho match con tal modelo
        # TODO no -> info: tal instance no ha hecho match con ningun modelo