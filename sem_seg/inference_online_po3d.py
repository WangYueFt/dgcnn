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

parsed_args = parser.parse_args()

path_data = parsed_args.path_data
path_cls = parsed_args.path_cls
model_path = os.path.join(parsed_args.model_path, "model.ckpt")
points_sub = parsed_args.points_sub
test_name = parsed_args.test_name
dump_path = os.path.join(parsed_args.model_path, "dump_" + test_name)
if not os.path.exists(dump_path): os.mkdir(dump_path)

classes, labels, label2color = indoor3d_util.get_info_classes(path_cls)
num_classes = len(classes)

batch_size = 1
gpu_index = 0
block_sub = 0.2
stride_sub = 0.2

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
                    print("loading")
                    data_label_full = np.loadtxt(filepath)  # read from txt (not on xyz origin)
                    #os.remove(filepath)
                    print("loaded")

                    if data_label_full.shape[0]> 2000: # 200000 check pointcloud points, a good PC has ~ 480k points  -> 200K for unfiltered PC, 2k for filtered PC

                        # subsample data_label_full to match ros subscription
                        desired_points = 10000
                        idx_full_sub = np.random.choice(data_label_full.shape[0], desired_points, replace=False)
                        data_label_full_sub = data_label_full[:, 0:6]

                        xyz_min = np.amin(data_label_full_sub, axis=0)[0:3]  # get pointcloud mins
                        data_label_full_sub[:, 0:3] -= xyz_min               # move pointcloud to origin
                        xyz_max = np.amax(data_label_full_sub, axis=0)[0:3] # get pointcloud maxs

                        data_sub, label_sub = indoor3d_util.room2blocks_plus_normalized_parsed(data_label_full_sub, xyz_max, points_sub, block_size=block_sub, stride=stride_sub, random_sample=False, sample_num=None, sample_aug=1) # subsample PC for evaluation

                        with tf.Graph().as_default():
                            pred_sub = evaluate(data_sub, label_sub, xyz_max, sess, ops)  # evaluate PC
                        pred_sub = np.unique(pred_sub, axis=0)                            # delete duplicates from room2blocks

                        pred_sub[:, 0:3] += xyz_min                                       # recover PC's original position

                        fout_sub = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_sub.obj'), 'w')
                        fout_sub_col = open(os.path.join(dump_path, os.path.basename(filepath)[:-4]+'_pred_sub_col.obj'), 'w')
                        for i in range(pred_sub.shape[0]):
                            fout_sub.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], pred_sub[i,3], pred_sub[i,4], pred_sub[i,5], pred_sub[i,6]))
                        for i in range(pred_sub.shape[0]):
                            color = label2color[pred_sub[i,6]]
                            fout_sub_col.write('v %f %f %f %d %d %d %d\n' % (pred_sub[i,0], pred_sub[i,1], pred_sub[i,2], color[0], color[1], color[2], pred_sub[i,6]))

                                
                    else:
                        print("PC discarted as n_points < 2000")
                        