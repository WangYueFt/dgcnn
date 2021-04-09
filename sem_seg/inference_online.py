import os
import re
import sys
import time
import argparse
import get_info
import numpy as np
from model import *
import project_inst
import open3d as o3d
import indoor3d_util
import get_instances
from natsort import natsorted

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

"python inference_online_all.py --path_data a/b/c --path_cls meta/class_or.txt --model_path RUNS/test_indoor --points_sub 128 --points_proj 512 --test_name test_name"

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', help='folder with train test data')
parser.add_argument('--path_cls', help='path to classes txt.')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--points_sub', type=int, default=256, help='Point number sub [default: 4096]')
parser.add_argument('--test_name', help='name of the test')
parser.add_argument('--down_pred', default = True, help='downsample prediction')
parser.add_argument('--targets_path', help='path in valve models.')

parsed_args = parser.parse_args()

path_data = parsed_args.path_data
path_cls = parsed_args.path_cls
model_path = os.path.join(parsed_args.model_path, "model.ckpt")
points_sub = parsed_args.points_sub
down_pred = parsed_args.down_pred
test_name = parsed_args.test_name
targets_path = parsed_args.targets_path
dump_path = os.path.join(parsed_args.model_path, "dump_" + test_name)
if not os.path.exists(dump_path): os.mkdir(dump_path)

classes, labels, label2color = indoor3d_util.get_info_classes(path_cls)
num_classes = len(classes)

batch_size = 1
gpu_index = 0
block_sub = 0.1
stride_sub = 0.1
block_proj = 0.1
stride_proj = 0.1


def evaluate(data, label, xyz_max, sess, ops):

    is_training = False

    label = np.squeeze(label)

    num_batches = data.shape[0] // batch_size

    pred_label_list =list()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size
        
        feed_dict = {ops['pointclouds_pl']: data[start_idx:end_idx, :, :],
                     ops['labels_pl']: label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        pred_label = np.argmax(pred_val, 2)
        pred_label = pred_label.reshape(pred_label.shape[0]*pred_label.shape[1],1)
        pred_label_list.append(pred_label)

    if pred_label_list:
        pred_label_stacked = np.vstack(pred_label_list)  

        data = data.reshape((data.shape[0]*data.shape[1]), data.shape[2])
        data = np.delete(data, [0,1,2], 1)
        data[:, [0,1,2,3,4,5,]] = data[:, [3,4,5,0,1,2]] 
        data[:,0] *= xyz_max[0]
        data[:,1] *= xyz_max[1]
        data[:,2] *= xyz_max[2]
        data[:,3:] *= 255.0
        
        data = data[:pred_label_stacked.shape[0], :]

        pred_sub = np.hstack([data,pred_label_stacked])  
    else:
        pred_sub = np.array([])
    return pred_sub


if __name__=='__main__':

    # get valve matching targets
    targets_list = list()
    for file_name in natsorted(os.listdir(targets_path)):
        target_path = os.path.join(targets_path, file_name)
        target = get_info.read_ply(target_path, "model")
        target_o3d = o3d.geometry.PointCloud()
        target_o3d.points = o3d.utility.Vector3dVector(target[:,0:3])
        target_o3d.colors = o3d.utility.Vector3dVector(target[:,3:6])
        target_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
        target_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))
        targets_list.append(target_o3d)

    # init tensorflow
    with tf.device('/gpu:'+str(gpu_index)):
        pointclouds_pl, labels_pl = placeholder_inputs(batch_size, points_sub)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, model_path)

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}

    # LOOP
    while 1:

        for root, dirs, files in os.walk(path_data):        # for each folder

            for file in enumerate(sorted(files)):           # for each file           

                if re.search("\.(txt)$", file[1]):          # if its a txt       
                    print("working on: " + str(file[1]))
                    
                    filepath = os.path.join(root, file[1])
                    data_label_full = np.loadtxt(filepath)  # read from txt (not on xyz origin)
                    #os.remove(filepath)

                    data_proj = data_label_full.copy()

                    if data_label_full.shape[0]> 2000: # 200000 check pointcloud points, a good PC has ~ 480k points  -> 200K for unfiltered PC, 2k for filtered PC

                        # subsample data_label_full to match ros subscription
                        desired_points = int(6000/(128/points_sub)) # 5000
                        idx_full_sub = np.random.choice(data_label_full.shape[0], desired_points, replace=False)
                        data_label_full_sub = data_label_full[idx_full_sub, 0:6]

                        xyz_min = np.amin(data_label_full_sub, axis=0)[0:3]  # get pointcloud mins
                        data_label_full_sub[:, 0:3] -= xyz_min               # move pointcloud to origin
                        xyz_max = np.amax(data_label_full_sub, axis=0)[0:3] # get pointcloud maxs

                        data_sub, label_sub = indoor3d_util.room2blocks_plus_normalized_parsed(data_label_full_sub, xyz_max, points_sub, block_size=block_sub, stride=stride_sub, random_sample=False, sample_num=None, sample_aug=1) # subsample PC for evaluation

                        with tf.Graph().as_default():
                            pred_sub = evaluate(data_sub, label_sub, xyz_max, sess, ops)  # evaluate PC
                        pred_sub = np.unique(pred_sub, axis=0)                            # delete duplicates from room2blocks
                        pred_sub[:, 0:3] += xyz_min                                       # recover PC's original position

                        if down_pred == True:                  # if subsampling of prediciton is wanted
                            down = 128/points_sub                 # down_pred to 128
                            n_idx_pred_sub_down = int(pred_sub.shape[0] * down)  
                            idx_pred_sub_down = np.random.choice(pred_sub.shape[0], n_idx_pred_sub_down, replace=False)
                            pred_sub = pred_sub[idx_pred_sub_down, 0:7]     # downsample prediciton


                        col_inst = {
                        0: [255, 255, 0],
                        1: [255, 0, 255],
                        2: [0, 255, 255],
                        3: [0, 128, 0],
                        4: [0, 0, 128],
                        5: [128, 0, 0],
                        6: [0, 255, 0],
                        7: [0, 0, 255],
                        8: [255, 0, 0],
                        9: [0, 100, 0],
                        10: [0, 0, 100],
                        11: [100, 0, 0],
                        12: [100, 0, 255],
                        13: [0, 255, 100],
                        13: [255, 100, 0]
                        }

                        # init get_instances parameters
                        rad_p = 0.04               # max distance for pipe growing                             //PARAM
                        rad_v = 0.04               # max distance for valve growing                            //PARAM
                        dim_p = 3                  # compute 2D (2) or 3D (3) distance for pipe growing        //PARAM
                        dim_v = 2                  # compute 2D (2) or 3D (3) distance for valve growing       //PARAM
                        min_p_p = 60               # minimum number of points to consider a blob as a pipe     //PARAM
                        min_p_v = 30 # 40 80 140   # minimum number of points to consider a blob as a valve    //PARAM

                        # get instances ref
                        pred_sub_pipe = pred_sub[pred_sub[:,6] == [labels["pipe"]]]       # get data label pipe
                        pred_sub_valve = pred_sub[pred_sub[:,6] == [labels["valve"]]]     # get data label pipe

                        instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances(pred_sub_valve, dim_v, rad_v, min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
                        #instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances_o3d(pred_sub_valve, dim_v, rad_v, min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
                        

                        info_valves_list = list()
                        for i, inst in enumerate(instances_ref_valve_list):

                            xyz_central = np.mean(inst, axis=0)[0:3]
                            inst[:, 0:3] -= xyz_central                # move instance to origin
                            inst_o3d = o3d.geometry.PointCloud()
                            inst_o3d.points = o3d.utility.Vector3dVector(inst[:,0:3])
                            inst_o3d.colors = o3d.utility.Vector3dVector(inst[:,3:6])
                            inst_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=15))
                            inst_o3d.orient_normals_to_align_with_direction(orientation_reference=([0, 0, 1]))
                            inst[:, 0:3] += xyz_central                # move instance to original position

                            info_valve = get_info.get_info(inst_o3d, targets_list, method="matching")
                            max_info =  max(info_valve)
                            max_idx = info_valve.index(max_info)

                            rad = math.radians(max_info[1])
                            vector = np.array([math.cos(rad), math.sin(rad)])                               # get valve unit vector 
                            vector = vector*0.18                                                            # resize vector to valve size //PARAM

                            info_valves_list.append([xyz_central, max_info, vector, max_idx])  

                        descart_valves_list = [i for i, x in enumerate(info_valves_list) if x[1][0] < 0.35] 

                        for i, idx in enumerate(descart_valves_list):
                            descarted_points = np.vstack(instances_ref_valve_list[idx])
                            stolen_idx = list(np.vstack(stolen_list[idx])[:,0].astype(int))
                            stolen_cls = np.vstack(stolen_list[idx])[:,1].astype(int)
                            stolen_cls = stolen_cls.reshape(stolen_cls.shape[0],1)
                            if len(stolen_idx)>0:
                                stolen_points = descarted_points[stolen_idx, :-2]
                                stolen_points = np.concatenate((stolen_points,stolen_cls),axis=1)
                                pred_sub_pipe_ref = np.concatenate((pred_sub_pipe_ref,stolen_points),axis=0)

                        for index in sorted(descart_valves_list, reverse=True):
                            del info_valves_list[index]
                            del instances_ref_valve_list[index]

                        print("INFO VALVES:")
                        for i, inst in enumerate(info_valves_list):
                            print(inst)


                        instances_ref_pipe_list, _, _  = get_instances.get_instances(pred_sub_pipe_ref, dim_p, rad_p, min_p_p)
                        #instances_ref_pipe_list, _, _  = get_instances.get_instances_o3d(pred_sub_pipe_ref, dim, rad_p, min_p_p)
                        
                        info_pipes_list = list()
                        info_connexions_list = list()
                        k_pipe = 0

                        for i, inst in enumerate(instances_ref_pipe_list): # for each pipe instance
                            # transform instance to o3d pointcloud
                            inst_o3d = o3d.geometry.PointCloud()
                            inst_o3d.points = o3d.utility.Vector3dVector(inst[:,0:3])
                            inst_o3d.colors = o3d.utility.Vector3dVector(inst[:,3:6]/255)

                            info_pipe = get_info.get_info(inst_o3d, models=0, method="skeleton") # get pipe instance info list( list( list(chain1, start1, end1, elbow_list1, vector_chain_list1), ...), list(connexions_points)) 
                            
                            for j, pipe_info in enumerate(info_pipe[0]):                         # stack pipes info
                                info_pipes_list.append(pipe_info)

                            for j, connexion_info in enumerate(info_pipe[1]):                    # stack conenexions info
                                connexion_info[1] = [x+k_pipe for x in connexion_info[1]]
                                info_connexions_list.append(connexion_info)

                            k_pipe += len(info_pipe[0])                                          # update actual pipe idx


                        info_pipes_list2 = get_info.unify_chains(info_pipes_list, info_connexions_list) 

                        info = [info_pipes_list2, info_connexions_list, info_valves_list]         # TODO publish info

                        # print info

                        print(" ")
                        print("INFO VALVES:")
                        for valve in info_valves_list:
                            print(valve)
                        print(" ")

                        print("INFO PIPES:")
                        for pipe in info_pipes_list:
                            pipe.pop(0)
                            print(pipe)
                        print(" ")

                        print("INFO PIPES2")
                        for pipe in info_pipes_list2:
                            pipe.pop(0)
                            print(pipe)
                        print(" ")

                        print("INFO CONNEXIONS:")
                        for connexion in info_connexions_list:
                            print(connexion)
                        print(" ")

                        # PRINTS

                        i = len(instances_ref_valve_list)

                        if len(instances_ref_valve_list)>0:
                            instances_ref_valve = np.vstack(instances_ref_valve_list)
                        if len(instances_ref_pipe_list)>0:
                            instances_ref_pipe = np.vstack(instances_ref_pipe_list)
                            instances_ref_pipe[:,7] = instances_ref_pipe[:,7]+i

                        if len(instances_ref_valve_list)>0 and len(instances_ref_pipe_list)>0:
                            instances_ref = np.concatenate((instances_ref_valve, instances_ref_pipe), axis=0)
                        elif len(instances_ref_valve_list)==0 and len(instances_ref_pipe_list)>0:
                            instances_ref = instances_ref_pipe
                        elif len(instances_ref_valve_list)>0 and len(instances_ref_pipe_list)==0:
                            instances_ref = instances_ref_valve
                        else:
                            instances_ref = None

                        fout_sub = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_sub.obj'), 'w')
                        fout_sub_col = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_sub_col.obj'), 'w')
                        for i in range(pred_sub.shape[0]):
                            fout_sub.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], pred_sub[i,3], pred_sub[i,4], pred_sub[i,5], pred_sub[i,6]))
                        for i in range(pred_sub.shape[0]):
                            color = label2color[pred_sub[i,6]]
                            fout_sub_col.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], color[0], color[1], color[2], pred_sub[i,6]))

                        if instances_ref is not None: # if instances were found
                            instances_ref[:, 0:3] += xyz_min                                       # recover PC's original position
                            fout_inst = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_inst_ref.obj'), 'w')
                            fout_inst_col = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_inst_ref_col.obj'), 'w')
                            for i in range(instances_ref.shape[0]):
                                fout_inst.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], instances_ref[i,3], instances_ref[i,4], instances_ref[i,5], instances_ref[i,6], instances_ref[i,7]))
                            for i in range(instances_ref.shape[0]):
                                color = col_inst[instances_ref[i,7]]
                                fout_inst_col.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], color[0], color[1], color[2], instances_ref[i,6], instances_ref[i,7]))

                                
                    else:
                        print("PC discarted as n_points < 2000")
                        