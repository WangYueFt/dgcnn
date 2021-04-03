import os
import re
import sys
import time
import copy
import math
import argparse
import numpy as np
import scipy as scp
import open3d as o3d
from natsort import natsorted
import matplotlib.pyplot as plt
from scipy.spatial import distance
from plyfile import PlyData, PlyElement
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize


'''

 - python get_info.py --path_projections data/ --path_models valve_targets/ --path_cls 4.txt 

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

def angle_between_vectors(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle)


def numpy_unique(arr):
    _, index = np.unique(arr, axis=0,return_index=True)
    arr_unique = arr[np.sort(index)]
    return arr_unique

def read_ply(filename, type):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    if type == "proj":
        pc_array = np.array([[x, y, z, r, g, b, c, i] for x,y,z,r,g,b,c,i in pc])
    if type == "model":
        pc_array = np.array([[x, y, z, nx, ny ,nz, r, g, b] for x,y,z,nx,ny,nz,r,g,b in pc])
    return pc_array


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def print_o3d(pc):
    pc_temp = copy.deepcopy(pc)
    o3d.visualization.draw_geometries([pc_temp])

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


def match(source, target):
    
    threshold = 0.02
    matchings = list()

    for i in range(16): # 0-15
        trans = np.eye(4)
        trans[:3,:3] = source.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*i))
        reg_p2l = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans)
        matchings.append(reg_p2l.fitness)
        #print("- matching: " + str(reg_p2l.fitness))
        #draw_registration_result(source, target, trans)
    
    best_idx = matchings.index(max(matchings))
    best_matching = matchings[best_idx]

    #print("-- best matching: " + str(best_matching) + " at angle: " + str((360/16)*(best_idx)))

    #trans[:3,:3] = source.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*(best_idx)))
    #reg_p2l = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans)
    #draw_registration_result(source, target, trans)

    return best_matching, (360/16)*(best_idx)


def print_chain(chain, maxs = np.array([])):

    if maxs.size == 0:
        maxs = np.amax(chain, axis=0)  # get voxel maxs

    matrix = np.zeros((maxs[1]+1, maxs[2]+1), dtype=int)
    for i, v in enumerate(chain):
        matrix[v[0],v[1]] = 1

    plt.imshow(matrix)
    plt.show()

    
def get_connectivity(array):

    chains = list()
    connexions = list()
    # get starting point -> (1 neighbour pixel or 2 if there are no 1 neighbour pixels)
    nonzero = np.transpose(np.nonzero(array))
    mid = False
    start_points = list()
    
    for index in nonzero:
        neighbour_list = get_neighbours(index, array, 1)
        if len(neighbour_list)==1:
            start_points.append(index)
            break
    
    if not start_points: # if there are not 1 neighbour pixels -> mid, select a 2 neighbour pixel # TODO AUN ESTA EL FALLO DE UNIR CHAINS EN LOOP POR 2 LADOS, O DEJARLO ASI, O ARREGLARLO CON LISTA DE TODOS LOS QUE
        mid = True                                                                                # SON IGUALES Y YA SERIVIRA PARA DESPUES, O QUITAR EMPEZAR POR 2 NEIGHBOURS Y YA
        for index in nonzero:
            neighbour_list = get_neighbours(index, array, 1)
            if len(neighbour_list)==2:
                start_points.append(index)
                break
    while start_points:
        start = start_points[0]
        start_points.pop(0)
        array[start[0],start[1]] = 0

        if mid == True:
            mid = False
            neighbour_list = get_neighbours(start, array, 1)

            chain0 = [np.array(start)]
            chain1, connexion1, new_starts1 = get_chain(neighbour_list[0], array, chain0)
            for idx in chain1:
                array[idx[0], idx[1]] = 0
            if connexion1.size != 0:
                connexions.append(connexion1)
                array[connexion1[0], connexion1[1]] = 0
            start_points = start_points + new_starts1
            for idx in new_starts1:
                array[idx[0], idx[1]] = 0

            chain0 = [np.array(start)]
            chain2, connexion2, new_starts2 = get_chain(neighbour_list[1], array, chain0)
            for idx in chain2:
                array[idx[0], idx[1]] = 0
            if connexion2.size != 0:
                connexions.append(connexion2)
                array[connexion2[0], connexion2[1]] = 0
            start_points = start_points + new_starts2
            for idx in new_starts2:
                array[idx[0], idx[1]] = 0

            if chain2.shape[0]>2:                    # TODO CON > 2 YA NO NOTAMOS LOOP CON INICIO Y FIN IGUAL, FASTIDIABA POR ABAJO LOS ANY IN ...
                chain2 = np.delete(chain2, 0, 0)
                chain2 = np.flipud(chain2)
                chain = np.vstack((chain2,chain1))

            chains.append(chain)

        else:
            chain, connexion, new_starts = get_chain(start, array, [])   # chain list de puntos, connexion: array(1,2), new_starts lista de puntos

            chains.append(chain)
            for idx in chain:
                array[idx[0],idx[1]] = 0
            if connexion.size != 0:
                connexions.append(connexion)
                array[connexion[0],connexion[1]] = 0
            start_points = start_points + new_starts
            for idx in new_starts:
                array[idx[0], idx[1]] = 0

    return chains, connexions


def get_chain(start, array, chain = []):
    chain.append(start)
    connexion = np.array([])
    new_starts = list()
    next = True

    while next == True:
        neighbour_list = get_neighbours(chain[-1], array, 1)
        del_list = list()

        for i, n in enumerate(neighbour_list):
            for c in chain:
                if np.all(n == c):
                    del_list.append(i)

        for i in sorted(del_list, reverse=True):
            del neighbour_list[i]

        if len(neighbour_list) == 0:
            next = False
        elif len(neighbour_list) == 1:
            chain.append(neighbour_list[0])
        else:
            next = False
            connexion = chain.pop()
            new_starts = neighbour_list
    chain_np = np.array(chain)

    return chain_np, connexion, new_starts


def get_neighbours(idx, array, dist):
    neighbour_list = list()

    row0 = max(idx[0]-dist,0)
    row1 = min(idx[0]+dist+1, array.shape[0])
    col0 = max(idx[1]-dist,0)
    col1 = min(idx[1]+dist+1, array.shape[1])

    for row in range(row0, row1):
        for col in range(col0, col1):
            if array[row, col] == 1:
                neighbour = np.array([row, col])
                if np.array_equal(neighbour, idx) == False:
                    neighbour_list.append(neighbour)
    return neighbour_list


def get_info_connexions(connexions, chains):

    connexions_info = list()

    for connexion in connexions:
        near_chains_list = list()
        for i, chain in enumerate(chains):
            d_to_start = distance.cityblock(connexion, chain[0])
            d_to_end = distance.cityblock(connexion, chain[-1])
            if d_to_start <= 3 or d_to_end <= 3:   # TODO 3 PARA LLEAGR A 1 MAS LEJOS EN DIAGONAL
                near_chains_list.append(i)
        connexions_info.append([connexion, near_chains_list])
    

    #delete connexions with only 1 near pipe, possible due to the deletion of short chains
    connexion_del_list = list()
    for i, c_info in enumerate(connexions_info):
        if len(c_info[1])<=1:
            connexion_del_list.append(i)
    for i in sorted(connexion_del_list, reverse=True):
        del connexions_info[i]

    connexion_del_list = list()
    new_connexions_info = list()
    for i, c_info1 in enumerate(connexions_info):
        if i not in connexion_del_list:
            for j, c_info2 in enumerate(connexions_info):
                if j not in connexion_del_list:
                    d_to_c = distance.cityblock(c_info1[0],c_info2[0])
                    if d_to_c <= 3 and i != j: # TODO DISTANCIA ENTRE CONEXIONES 3 PARA ACEPTAR CONEXION A 1 EN DIAGONAL
                        connexion_del_list.append(i)
                        connexion_del_list.append(j)
                        new_near_chains_list = list(set(c_info1[1]+c_info2[1]))
                        new_connexions_info.append([c_info1[0], new_near_chains_list])

    for i in sorted(connexion_del_list, reverse=True):
        del connexions_info[i]

    connexions_info = connexions_info + new_connexions_info

    connexion_del_list = list()

    for i, c_info in enumerate(connexions_info):
        if len(c_info[1])==2:

            if any(np.array_equal(chains[c_info[1][0]][0], x) for x in chains[c_info[1][1]]) == False and any(np.array_equal(chains[c_info[1][1]][0], x) for x in chains[c_info[1][0]]) == False: # para evitar que se comparen 2 cadenas que una contiene a la otra
                
                connexion_del_list.append(i)

                d_to_start1 = distance.cityblock(c_info[0], chains[c_info[1][0]][0])
                d_to_end1 = distance.cityblock(c_info[0], chains[c_info[1][0]][-1])
                d_to_start2 = distance.cityblock(c_info[0], chains[c_info[1][1]][0])
                d_to_end2 = distance.cityblock(c_info[0], chains[c_info[1][1]][-1])

                union1 = [d_to_start1, d_to_end1].index(min([d_to_start1, d_to_end1]))
                union2 = [d_to_start2, d_to_end2].index(min([d_to_start2, d_to_end2]))

                chain1 = chains[c_info[1][0]]
                chain2 = chains[c_info[1][1]]

                if union1 == 0:
                    if union2 == 0:
                        chain2 = np.flipud(chain2)
                        chain2 = np.vstack((chain2, c_info[0]))
                        new_chain = np.vstack((chain2, chain1))
                    else:
                        chain2 = np.vstack((chain2, c_info[0]))
                        new_chain = np.vstack((chain2, chain1))
                else:
                    if union2 == 0:
                        chain1 = np.vstack((chain1, c_info[0]))
                        new_chain = np.vstack((chain1, chain2))
                    else:
                        chain2 = np.flipud(chain2)
                        chain1 = np.vstack((chain1, c_info[0]))
                        new_chain = np.vstack((chain1, chain2))

                chains[c_info[1][0]] = new_chain
                chains[c_info[1][1]] = new_chain
            
    for i in sorted(connexion_del_list, reverse=True):
        del connexions_info[i]

    chain_del_list = list()
    for i, chain1 in enumerate(chains):
        if i not in chain_del_list:
            for j , chain2 in enumerate(chains):
                if j not in chain_del_list:
                    if i != j:
                        chain_test = np.vstack((chain1,chain2))
                        chain_test_u = np.unique(chain_test, axis=0)
                        if chain1.shape[0] == chain_test_u.shape[0]:
                            chain_del_list.append(j) 
    for i in sorted(chain_del_list, reverse=True):
        del chains[i]

    return connexions_info, chains


def get_info_skeleton(instance):

    print_opt = False
    #print_o3d(instance)

    # VOXELS FROM POINTCLOUD
    instance1 = copy.deepcopy(instance)
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(instance1,voxel_size=0.008)
    #print_o3d(voxel_grid1)

    voxels = o3d.geometry.VoxelGrid.get_voxels(voxel_grid1)
    #print("n voxels: " + str(len(voxels)))
    #print(voxels)

    voxels_np = np.zeros((len(voxels),3), dtype=int)
    for i in range(len(voxels)):
        voxels_np[i] = voxels[i].grid_index
    #print(voxels_np)    

    voxels_np.T[[0,1,2]] = voxels_np.T[[2,0,1]] 
    #print(voxels_np)

    xyz_max = np.amax(voxels_np, axis=0)  # get voxel maxs
    voxels_matrix = np.zeros(xyz_max+1, dtype=int)
    for i, v in enumerate(voxels_np):
        voxels_matrix[v[0],v[1],v[2]] = 1

    #print(voxels_matrix)
    z,x,y = voxels_matrix.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c= 'red')
    fig.suptitle('3D VOXELS', fontsize=12)
    if print_opt == True:
        plt.show()


    voxels_matrix_2d = np.zeros((xyz_max[1]+1, xyz_max[2]+1), dtype=int)
    for i, v in enumerate(voxels_np):
        voxels_matrix_2d[v[1],v[2]] = 1
    #plt.imshow(voxels_matrix_2d)
    #if print_opt == True:
    #    plt.show()

    close_dist = 6
    voxels_matrix_2d_proc = scp.ndimage.binary_closing(voxels_matrix_2d, structure=np.ones((close_dist,close_dist)))
    #plt.imshow(voxels_matrix_2d_proc)
    #if print_opt == True:
    #    plt.show()

    voxels_matrix_2d_proc = scp.ndimage.binary_opening(voxels_matrix_2d_proc, structure=np.ones((2,2)))
    #plt.imshow(voxels_matrix_2d_proc)
    #if print_opt == True:
    #    plt.show()

    skeleton = skeletonize(voxels_matrix_2d_proc)
    skeleton = skeleton.astype(int)

    fig = plt.figure()
    fig.suptitle('SKELETON', fontsize=12)
    plt.imshow(skeleton)
    if print_opt == True:
        plt.show()

    skeleton2 = skeleton.copy()
    chains, connexions = get_connectivity(skeleton2)

    #print("CHAINS ORIGINALS")
    #if print_opt == True:
    #    for chain in chains:
    #        print_chain(chain, xyz_max)

    
    # delete short chains
    chain_del_list = list()
    for i, chain in enumerate(chains):
        if len(chain) < 10:                  # TODO calcular bien numero y poner como parametro
            chain_del_list.append(i)
    for i in sorted(chain_del_list, reverse=True):
        del chains[i]

    #print("CHAINS SMALL DELETED")
    #if print_opt == True:
    #    for chain in chains:
    #        print_chain(chain, xyz_max)



    connexions, chains = get_info_connexions(connexions, chains)
    if print_opt == True:
        print("CHAINS INFO")

        for chain in chains:
            print_chain(chain, xyz_max)
        print("CONNEXIONS")
        print(connexions)

        print("OVERVIEW")
        chainoverview = chains[0]
        for i, chain in enumerate(chains):
            if i != 0:
                chainoverview = np.vstack((chainoverview, chain))
        print_chain(chainoverview, xyz_max)



    # project to nearest real voxel
    connexions_proj = list()
    for i, con in enumerate(connexions):
        v = con[0]
        v_proj = proj_voxel(v, voxels_matrix_2d, int(math.ceil(close_dist/2)))
        #print("voxel: " + str(v))
        #print("voxel_proj: " + str(v_proj))
        connexions_proj.append([v_proj,con[1]])

    chains_proj = list()    
    for i, chain in enumerate(chains):
        chain_proj= list()
        for v in chain:
            v_proj = proj_voxel(v, voxels_matrix_2d, int(math.ceil(close_dist/2)))
            chain_proj.append(v_proj)
        chain_proj_np = np.array(chain_proj)
        chain_proj_np_unique = numpy_unique(chain_proj_np)
        chains_proj.append(chain_proj_np_unique)

    # convert voxels of chains and connexions to points
    corr_list = list()
    instance1_points = np.asarray(instance1.points)
    #print("n points: " + str(instance1_points.shape))
    for p in instance1_points:
        voxel = o3d.geometry.VoxelGrid.get_voxel(voxel_grid1, p)
        corr_list.append(voxel)

    #print(corr_list)
    connexions_points = list()
    for i, con in enumerate(connexions_proj):
        v = con[0]
        p = voxel_to_point(v, instance1_points, corr_list)
        connexions_points.append([p,con[1]])

    chains_points = list()    
    for i, chain in enumerate(chains_proj):
        chain_point= list()
        for v in chain:
            p = voxel_to_point(v, instance1_points, corr_list)
            chain_point.append(p)
        chain_point_np = np.array(chain_point)
        chains_points.append(chain_point_np)

    info_chains = list()
    for i, chain in enumerate(chains_points):
        info_chain = list()

        chain_o3d = o3d.geometry.PointCloud()
        chain_o3d.points = o3d.utility.Vector3dVector(chain[:,0:3])
        #print_o3d(chain_o3d)

        vector_list = list()
        for j in range(len(chain)-1):
            vector = chain[j] - chain[j+1]
            vector_list.append(vector)
    
        look_ahead = 14
        elbow_size = 9
        angle_elbow = 60
        angle_list = list()
        elbow_idx_list = list()
        if chain.shape[0] > look_ahead*3:
            for i in range(look_ahead, chain.shape[0]-look_ahead):
                vector1 = chain[i-look_ahead] - chain[i]
                vector2 = chain[i] - chain[i+look_ahead]
                angle = angle_between_vectors(vector1, vector2)
                angle_list.append(angle)

            while 1:
                max_angle = max(angle_list)
                if max_angle > angle_elbow:
                    max_index = angle_list.index(max_angle)
                    elbow_idx_list.append(max_index+look_ahead)

                    start = max(max_index-elbow_size,0)                     # TODO 10 SIGNIFICA QUE ELBOW SE LLEVA 20 PUNTOS (+-10)
                    end = min(max_index+elbow_size, len(angle_list))        # TODO 10 SIGNIFICA QUE ELBOW SE LLEVA 20 PUNTOS (+-10)

                    for i in range(start,end):
                        angle_list[i] = 0
                else:
                    elbow_idx_list.sort()
                    break

        elbow_list = list()
        for i in elbow_idx_list:
            elbow_list.append(chain[i])

        vector_chain_list = list()
        if len(elbow_idx_list) == 0:
            vector_chain = chain[0] - chain[-1]
            vector_chain_list.append(vector_chain)
        elif len(elbow_idx_list) == 1:
            vector_chain = chain[0] - chain[elbow_idx_list[0]-elbow_size]
            vector_chain_list.append(vector_chain)
            vector_chain = chain[elbow_idx_list[0]+elbow_size] - chain[-1]
            vector_chain_list.append(vector_chain)
        else:
            vector_chain = chain[0] - chain[elbow_idx_list[0]-elbow_size]
            vector_chain_list.append(vector_chain)

            for e in range(len(elbow_idx_list)-1):
                vector_chain = chain[elbow_idx_list[e]+elbow_size] - chain[elbow_idx_list[e+1]-elbow_size]
                vector_chain_list.append(vector_chain)

            vector_chain = chain[elbow_idx_list[-1]+elbow_size] - chain[-1]
            vector_chain_list.append(vector_chain)

        start_idx = get_position_idx(vector_list, 0)
        start = chain[start_idx]

        mid_idx = get_position_idx(vector_list, 50)
        mid = chain[mid_idx]

        end_idx = get_position_idx(vector_list, 100)
        end = chain[end_idx]

        info_chain = [chain, start, end, elbow_list, vector_chain_list]
        #info_chain = [start, end, elbow_list, vector_chain_list]
        info_chains.append(info_chain)
    
    info = [info_chains, connexions_points]

    return info

def get_position_idx(vector_list, percentage):
    total_dist = 0
    for vector in vector_list:              # TODO SE PUEDE OPTIMIZAR PASANDOLE LA TOTAL DIST DESDE FUERA Y ASI SOLO SE CALCULA 1 VEZ
        dist = np.linalg.norm(vector)
        total_dist = total_dist + dist

    target_dist = total_dist * (percentage/100)
    actual_dist = 0
    for i, vector in enumerate(vector_list):
        dist = np.linalg.norm(vector)
        actual_dist = actual_dist + dist
        if actual_dist >= target_dist:
            if actual_dist-target_dist < dist-(actual_dist-target_dist):
                position = i+1
            else:
                position = i
            break
    return position


def proj_voxel(voxel, voxels, max_d):
    if voxels[voxel[0], voxel[1]] == 1:
        voxel_proj = voxel
    else:
        neighbours = get_neighbours(voxel, voxels, max_d)
        min_d = 99
        for n in neighbours:
            v_to_n = distance.cityblock(voxel, n)
            if v_to_n < min_d:
                min_d = v_to_n
                voxel_proj = n
    return voxel_proj


def voxel_to_point(voxel, points, corr):
    i_corr = [i for i, c in enumerate(corr) if c[0] == voxel[0] and c[1] == voxel[1]]
    corr_point_list = list()
    for i in i_corr:
        corr_point_list.append(points[i])
    corr_point_np = np.array(corr_point_list)
    point = np.mean(corr_point_np, axis=0)    # TODO USAR MEAN?
    return point


def get_info_matching(instance, models):
    info_inst = list()
    for model in models:
        fitness, transform = match(instance, model)
        info_inst.append([fitness, transform])
    return info_inst


def get_info(instance, models, method):
    if method == "skeleton":
        info = get_info_skeleton(instance)
    elif method == "matching":
        info = get_info_matching(instance, models)
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

        models_fpfh_list.append(model_o3d)


    for file_name in natsorted(os.listdir(path_projections)):
        
        info_valves_list = list()
        centrals_list = list()
        inst_v_list = list()

        if "_pred_inst_ref." in file_name:

            print("evaluating case: " + file_name)
            path_projection = os.path.join(path_projections, file_name)
            projection = np.loadtxt(path_projection, usecols=[1,2,3,4,5,6,7,8])

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
                inst_o3d = o3d.geometry.PointCloud()
                inst_o3d.points = o3d.utility.Vector3dVector(inst[:,0:3])
                inst_o3d.colors = o3d.utility.Vector3dVector(inst[:,3:6]/255)

                instances_pipe_list.append(inst_o3d)
                info_pipe = get_info(inst_o3d, models=0, method="skeleton")

            info_valves = list()

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
                inst_v_list.append(inst_o3d)

                instances_valve_list.append([inst_o3d, inst_fpfh, xyz_central])
                info_valve = get_info(inst_o3d, models_fpfh_list, method="matching")
                info_valves.append(info_valve)

            print(info_valves)

            for i, info_valve in enumerate(info_valves):
                max_fitness =  max(info_valve) 
                max_idx = info_valve.index(max_fitness)
                if max_fitness[0] < 0.1:
                    max_fitness[0] = 0
                info_valves_list.append([max_fitness, max_idx])

                central = centrals_list[i]
                inst_o3d = inst_v_list[i]
                max_model = copy.deepcopy(models_fpfh_list[max_idx])

                trans = np.eye(4)
                trans[:3,:3] = max_model.get_rotation_matrix_from_xyz((0,0, (np.pi/8)*(max_fitness[1]*(16/360))))
                #draw_registration_result(inst_o3d, max_model, trans)

       
            print("info valves list: "+ str(info_valves_list))
            print(" ")
