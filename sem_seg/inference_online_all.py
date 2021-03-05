import argparse
import os
import sys
import re
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from model import *
import indoor3d_util
import time
import numpy as np
import get_instances
import project_inst

"python inference_online_all.py --path_data a/b/c --path_cls meta/class_or.txt --model_path RUNS/test_indoor --points_sub 128 --points_proj 512 --test_name test_name"

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', help='folder with train test data')
parser.add_argument('--path_cls', help='path to classes txt.')
parser.add_argument('--model_path', required=True, help='model checkpoint file path')
parser.add_argument('--points_sub', type=int, default=256, help='Point number sub [default: 4096]')
parser.add_argument('--points_proj', type=int, default=0, help='Point number proj [default: 4096]')
parser.add_argument('--test_name', help='name of the test')
parser.add_argument('--down_pred', default = True, help='downsample prediction')

parsed_args = parser.parse_args()

path_data = parsed_args.path_data
path_cls = parsed_args.path_cls
model_path = os.path.join(parsed_args.model_path, "model.ckpt")
points_sub = parsed_args.points_sub
points_proj = parsed_args.points_proj
down_pred = parsed_args.down_pred
test_name = parsed_args.test_name
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
min_p_v = 30
min_p_p = 70


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

    return pred_sub


if __name__=='__main__':

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

    times_list = list()

    while 1:

        for root, dirs, files in os.walk(path_data):        # for each folder

            for file in enumerate(sorted(files)):           # for each file           

                if re.search("\.(txt)$", file[1]):          # if its a txt       
                    print("working on: " + str(file[1]))
                    
                    filepath = os.path.join(root, file[1])
                    data_label_full = np.loadtxt(filepath)  # read from txt (not on xyz origin)
                    os.remove(filepath)

                    if data_label_full.shape[0]> 2000: # 200000 check pointcloud points, a good PC has ~ 480k points  -> 200K for unfiltered PC, 2k for filtered PC

                        # init times that may not be calculated
                        time_down_pred = 0
                        time_sub_proj = 0
                        time_proj = 0

                        # subsample data_label_full to match ros subscription
                        desired_points = int(5000/(128/points_sub))
                        idx_full_sub = np.random.choice(data_label_full.shape[0], desired_points, replace=False)
                        data_label_full_sub = data_label_full[idx_full_sub, 0:6]


                        start = time.time()
                        xyz_min = np.amin(data_label_full_sub, axis=0)[0:3]  # get pointcloud mins
                        data_label_full_sub[:, 0:3] -= xyz_min               # move pointcloud to origin
                        end = time.time()
                        time_min = end - start

                        start = time.time()
                        xyz_max = np.amax(data_label_full_sub, axis=0)[0:3] # get pointcloud maxs
                        end = time.time()
                        time_max = end - start

                        start = time.time()
                        data_sub, label_sub = indoor3d_util.room2blocks_plus_normalized_parsed(data_label_full_sub, xyz_max, points_sub, block_size=block_sub, stride=stride_sub, random_sample=False, sample_num=None, sample_aug=1) # subsample PC for evaluation
                        end = time.time()
                        time_sub_eval = end - start

                        start = time.time()
                        with tf.Graph().as_default():
                            pred_sub = evaluate(data_sub, label_sub, xyz_max, sess, ops)  # evaluate PC
                        pred_sub = np.unique(pred_sub, axis=0)                            # delete duplicates from room2blocks
                        pred_sub[:, 0:3] += xyz_min                                       # recover PC's original position
                        end = time.time()
                        time_eval = end - start
                    
                        fout_sub = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_sub.obj'), 'w')
                        fout_sub_col = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_sub_col.obj'), 'w')
                        for i in range(pred_sub.shape[0]):
                            fout_sub.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], pred_sub[i,3], pred_sub[i,4], pred_sub[i,5], pred_sub[i,6]))
                        for i in range(pred_sub.shape[0]):
                            color = label2color[pred_sub[i,6]]
                            fout_sub_col.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], color[0], color[1], color[2], pred_sub[i,6]))


                        if down_pred == True:                  # if subsampling of prediciton is wanted
                            start = time.time()
                            down = 128/points_sub                 # down_pred to 128
                            n_idx_pred_sub_down = int(pred_sub.shape[0] * down)  
                            idx_pred_sub_down = np.random.choice(pred_sub.shape[0], n_idx_pred_sub_down, replace=False)
                            pred_sub = pred_sub[idx_pred_sub_down, 0:7]     # downsample prediciton
                            end = time.time()
                            time_down_pred = end - start


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
                        rad_p = 0.045
                        rad_v = 0.045
                        dim = 2
                        min_p_v = min_p_v
                        min_p_p = min_p_p

                        # get instances noref
                        start = time.time()
                        pred_sub_pipe = pred_sub[pred_sub[:,6] == [labels["pipe"]]]       # get data label pipe
                        pred_sub_valve = pred_sub[pred_sub[:,6] == [labels["valve"]]]     # get data label pipe

                        instances_noref_valve_list, _, _  = get_instances.get_instances(pred_sub_valve, dim, rad_v, min_p_v)
                        instances_noref_pipe_list, _, _  = get_instances.get_instances(pred_sub_pipe, dim, rad_p, min_p_p)
                        i = len(instances_noref_valve_list)

                        if len(instances_noref_valve_list)>0:
                            instances_noref_valve = np.vstack(instances_noref_valve_list)
                        if len(instances_noref_pipe_list)>0:
                            instances_noref_pipe = np.vstack(instances_noref_pipe_list)
                            instances_noref_pipe[:,7] = instances_noref_pipe[:,7]+i

                        if len(instances_noref_valve_list)>0 and len(instances_noref_pipe_list)>0:
                            instances_noref = np.concatenate((instances_noref_valve, instances_noref_pipe), axis=0)
                        elif len(instances_noref_valve_list)==0 and len(instances_noref_pipe_list)>0:
                            instances_noref = instances_noref_pipe
                        elif len(instances_noref_valve_list)>0 and len(instances_noref_pipe_list)==0:
                            instances_noref = instances_noref_valve
                        else:
                            instances_noref = None
                        end = time.time()
                        time_inst_noref = end - start


                        # get instances ref
                        start = time.time()
                        pred_sub_pipe = pred_sub[pred_sub[:,6] == [labels["pipe"]]]       # get data label pipe
                        pred_sub_valve = pred_sub[pred_sub[:,6] == [labels["valve"]]]     # get data label pipe
                        instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances(pred_sub_valve, dim, rad_v, min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
                        matches_list = [1, 1, 1, 1, 1, 1, 1, 1, 1] # TODO matches_list = get_info(instances_ref_valve_list, models_list)
                        descart_list = [i for i, x in enumerate(matches_list) if x == None]

                        for i, idx in enumerate(descart_list):
                            descarted_points = np.vstack(instances_ref_valve_list[idx])
                            stolen_idx = list(np.vstack(stolen_list[idx])[:,0].astype(int))
                            stolen_cls = np.vstack(stolen_list[idx])[:,1].astype(int)
                            stolen_cls = stolen_cls.reshape(stolen_cls.shape[0],1)
                            if len(stolen_idx)>0:
                                stolen_points = descarted_points[stolen_idx, :-2]
                                stolen_points = np.concatenate((stolen_points,stolen_cls),axis=1)
                                pred_sub_pipe_ref = np.concatenate((pred_sub_pipe_ref,stolen_points),axis=0)

                        for index in sorted(descart_list, reverse=True):
                            del instances_ref_valve_list[index]

                        instances_ref_pipe_list, _, _  = get_instances.get_instances(pred_sub_pipe_ref, dim, rad_p, min_p_p)
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
                        end = time.time()
                        time_inst_ref = end - start

                        if instances_ref is not None: # if instances were found

                            fout_inst = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_inst_ref.obj'), 'w')
                            fout_inst_col = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_inst_ref_col.obj'), 'w')
                            for i in range(instances_ref.shape[0]):
                                fout_inst.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], instances_ref[i,3], instances_ref[i,4], instances_ref[i,5], instances_ref[i,6], instances_ref[i,7]))
                            for i in range(instances_ref.shape[0]):
                                color = col_inst[instances_ref[i,7]]
                                fout_inst_col.write('v %f %f %f %d %d %d %d %d\n' % (instances_ref[i,0], instances_ref[i,1], instances_ref[i,2], color[0], color[1], color[2], instances_ref[i,6], instances_ref[i,7]))


                            if points_proj != 0:    # if projection  # TODO YA NO SE PUEDE PROYECTAR, A MENOS QUE SE HAGAN 2 REDUCCIONES AL PRINCIPIO, UNA MENOR PARA PODER PROYECTAR SOBRE ESA, quiza habra que bajar fitler..
                                
                                # determine projection number of points
                                if block_proj == 0.2: 
                                    start = 0.3333 # 0.03333  TODO x10 because reduced data_label_full by /10 into data_label_full_sub
                                    by = 1024 / points_proj
                                    reduction = start / by

                                else:
                                    start = 0.5 # 0.05  TODO x10 because reduced data_label_full by /10 into data_label_full_sub 
                                    by = 512 / points_proj
                                    reduction = start / by
                                
                                # TODO USE FULL_SUB SO WE CAN REUSE XYZ MIN MAX
                                start = time.time()
                                n_idx_proj = int(data_label_full_sub.shape[0] * reduction)
                                idx_proj = np.random.choice(data_label_full_sub.shape[0], n_idx_proj, replace=False)
                                data_proj = data_label_full_sub[idx_proj, 0:6]  # subsample projection
                                data_proj[:, 0:3] += xyz_min # move projection to original position (data_label_full is on origin)
                                end = time.time()
                                time_sub_proj = end - start

                                start = time.time()
                                low_list = list()
                                for i in set(instances_ref[..., 7]):  # for each isntance low
                                    inst_low = instances_ref[np.where(instances_ref[..., 7] == float(i))] # get inst
                                    low_list.append(inst_low)
                                projections_list = project_inst.project_inst(low_list, data_proj)
                                end = time.time()
                                time_proj = end - start
                            else:
                                projections_list = low_list
                                
                            if len(projections_list)>0:
                                projections = np.vstack(projections_list) 
                                fout_proj = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_projections.obj'), 'w')
                                fout_proj_col = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_projections_col.obj'), 'w')
                                for i in range(projections.shape[0]):
                                    fout_proj.write('v %f %f %f %d %d %d %d %d\n' % (projections[i,0], projections[i,1], projections[i,2], projections[i,3], projections[i,4], projections[i,5], projections[i,6], projections[i,7]))
                                for i in range(projections.shape[0]):
                                    color = col_inst[projections[i,7]]
                                    fout_proj_col.write('v %f %f %f %d %d %d %d %d\n' % (projections[i,0], projections[i,1], projections[i,2], color[0], color[1], color[2], projections[i,6], projections[i,7]))
                            if points_proj != 0: 
                                fout_proj = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_base_proj.obj'), 'w')
                                for i in range(data_proj.shape[0]):
                                    fout_proj.write('v %f %f %f %d %d %d\n' % (data_proj[i,0], data_proj[i,1], data_proj[i,2], data_proj[i,3], data_proj[i,4], data_proj[i,5]))


                        times = (time_min, time_max, time_sub_eval, time_eval, time_down_pred, time_inst_noref, time_inst_ref, time_sub_proj, time_proj)
                        times_list.append(times)

                        times_mean = np.mean(times_list,axis=0)
                        times_stack = np.vstack(times_list)

                        fout_times = open(os.path.join(dump_path, 'times.txt'), 'w')
                        for i in range(times_stack.shape[0]):
                            fout_times.write('%f     %f     %f     %f     %f     %f     %f     %f     %f\n' % (times_stack[i,0], times_stack[i,1], times_stack[i,2], times_stack[i,3], times_stack[i,4], times_stack[i,5], times_stack[i,6], times_stack[i,7], times_stack[i,8]))
                        fout_times.write('--------- MEAN ---------\n')
                        fout_times.write(' t_min        t_max      t_sub_eval     t_eval    t_down_pred  t_inst_noref  t_inst_ref   t_sub_proj     t_proj\n')
                        fout_times.write('%f     %f     %f     %f     %f     %f     %f     %f     %f\n' % (times_mean[0], times_mean[1], times_mean[2], times_mean[3], times_mean[4], times_mean[5], times_mean[6], times_mean[7], times_mean[8]))
                        fout_times.close()

                    else:
                        print("PC discarted as n_points < 2000")
                        