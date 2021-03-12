import os
import sys
import time
import rospy
import ctypes
import struct
import numpy as np
from model import *
import indoor3d_util
import get_instances
import project_inst

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import PointField

import message_filters
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image, CameraInfo


class Pointcloud_Seg:
    def __init__(self, name):



        self.name = name
        # Params inference
        self.fps = 2
        self.period = 1/self.fps
        self.batch_size = 1
        self.points_sub = 256 # 128 256 512
        self.block_sub = 0.1
        self.stride_sub = 0.1
        self.gpu_index = 0
        self.desired_points = int(5000/(128/self.points_sub))


        # Params get_instance
        self.col_inst = {
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
        self.rad_p = 0.04
        self.rad_v = 0.04
        self.dim = 2
        self.min_p_v = 30 # 40 80 140
        self.min_p_p = 60        
        
        self.model_path = "/home/miguel/Desktop/test_ros_subscriber/4_256_11_c7/model.ckpt"
        self.path_cls = "/home/miguel/Desktop/test_ros_subscriber/4.txt"
        self.classes, self.labels, self.label2color = indoor3d_util.get_info_classes(self.path_cls)

        # TODO importar modelos valvulas, convertirlos a o3d pointclouds, calcular fpfh y meterlos en una lista


        self.init = False
        self.new_pc = False

        # Set image subscriber
        pc_sub = message_filters.Subscriber('/stereo_down/scaled_x2/points2_filtered', PointCloud2)
        #pc_sub = message_filters.Subscriber('/stereo_down/scaled_x2/points2', PointCloud2)
        info_sub = message_filters.Subscriber('/stereo_down/left/camera_info', CameraInfo)
        ts_image = message_filters.TimeSynchronizer([pc_sub, info_sub], 10)
        ts_image.registerCallback(self.cb_pc)

        # Set class image publisher
        self.pub_pc_base = rospy.Publisher("/stereo_down/scaled_x2/points2_base", PointCloud2, queue_size=4)
        self.pub_pc_seg = rospy.Publisher("/stereo_down/scaled_x2/points2_seg", PointCloud2, queue_size=4)
        self.pub_pc_inst = rospy.Publisher("/stereo_down/scaled_x2/points2_inst", PointCloud2, queue_size=4)

        # Set classification timer
        rospy.Timer(rospy.Duration(self.period), self.run)

        # CvBridge for image conversion
        self.bridge = CvBridge()


    def cb_pc(self, pc, info):
        self.pc = pc
        self.cam_info = info
        self.new_pc = True

    def set_model(self):
        with tf.device('/gpu:'+str(self.gpu_index)):
            pointclouds_pl, labels_pl = placeholder_inputs(self.batch_size, self.points_sub)
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
        self.sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(self.sess, self.model_path)

        self.ops = {'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'pred_softmax': pred_softmax,
            'loss': loss}

    def run(self,_):
        rospy.loginfo('[%s]: Entro', self.name)	
        t0 = rospy.Time.now()

        # New pc available
        if not self.new_pc:
            rospy.loginfo('[%s]: Not new', self.name)	
            return
        self.new_pc = False

        # Retrieve image
        try:
            pc = self.pc
            header = self.pc.header
            if not self.init:
                rospy.loginfo('[%s]: Start pc segmentation', self.name)	
        except:
            rospy.logwarn('[%s]: There is no input pc to run the segmentation', self.name)
            return

        # Set model
        if not self.init: 
            self.set_model()
            self.init = True

        pc_np = self.pc2array(pc)
        if pc_np.shape[0] < 2000:
            print("not enough input points")
            return

        pc_np[:, 2] *= -1  # flip Z axis
        pc_proj = pc_np.copy()

        # reduce pointcloud to desired number of points, IF PROJ TO HIGHER POINTS NOT NEEDED, THIS CAN GO INTO PC2ARRAY()
        if self.desired_points != 0:
            if pc_np.shape[0] > self.desired_points:
                idx_sub = np.random.choice(pc_np.shape[0], self.desired_points, replace=False)
                pc_np = pc_np[idx_sub, 0:6]

        xyz_min = np.amin(pc_np, axis=0)[0:3]   # get pointcloud mins
        pc_np[:, 0:3] -= xyz_min                # move pointcloud to origin
        pc_proj[:, 0:3] -= xyz_min 
        xyz_max = np.amax(pc_np, axis=0)[0:3]   # get pointcloud maxs

        t1 = rospy.Time.now()

        data_sub, label_sub = indoor3d_util.room2blocks_plus_normalized_parsed(pc_np,  xyz_max, self.points_sub, block_size=self.block_sub, stride=self.stride_sub, random_sample=False, sample_num=None, sample_aug=1) # subsample PC for evaluation

        if data_sub.size == 0:
            print("no data sub")
            return

        t2 = rospy.Time.now()

        with tf.Graph().as_default():
            pred_sub = self.evaluate(data_sub, label_sub, xyz_max)  # evaluate PC

        if pred_sub.size == 0:
            print("no pred sub")
            return

        pred_sub = np.unique(pred_sub, axis=0) # delete duplicates from room2blocks


        t3 = rospy.Time.now()

        pc_np_base = pred_sub.copy()
        pc_np_base = np.delete(pc_np_base,6,1) # delete class prediction

        if self.points_sub != 128:                  # if subsampling
            down = 128/self.points_sub    
            n_idx_pred_sub_down = int(pred_sub.shape[0] * down)  
            idx_pred_sub_down = np.random.choice(pred_sub.shape[0], n_idx_pred_sub_down, replace=False)
            pred_sub = pred_sub[idx_pred_sub_down, 0:7]     # downsample prediciton
        
        # get instances ref
        pred_sub_pipe = pred_sub[pred_sub[:,6] == [self.labels["pipe"]]]       # get data label pipe
        pred_sub_valve = pred_sub[pred_sub[:,6] == [self.labels["valve"]]]     # get data label pipe

        instances_ref_valve_list, pred_sub_pipe_ref, stolen_list  = get_instances.get_instances(pred_sub_valve, self.dim, self.rad_v, self.min_p_v, ref=True, ref_data = pred_sub_pipe, ref_rad = 0.1)
        instances_ref_proj_valve_list = project_inst.project_inst(instances_ref_valve_list, pc_proj) # pc_np_base
        # TODO CALCULATE CENTER OF EACH INSTANCE AND MOVE IT TO ORIGEN
        info_valves_list = [1, 1, 1, 1, 1, 1, 1, 1, 1] # TODO info_valves = get_info(instances_ref_proj_valve_list, method="matching", models_list) #TODO create models_list as list of [o3d, fpfh]
        # TODO RESTORE POSITION to BEFORE MOVING CENTER OF VALVES TO ORIGEN
        descart_valves_list = [i for i, x in enumerate(info_valves_list) if x == None] # TODO if fitness < thr

        for i in descart_valves_list:
            descarted_points = np.vstack(instances_ref_proj_valve_list[i])
            stolen_idx = list(np.vstack(stolen_list[i])[:,0].astype(int))
            stolen_cls = np.vstack(stolen_list[i])[:,1].astype(int)
            stolen_cls = stolen_cls.reshape(stolen_cls.shape[0],1)
            if len(stolen_idx)>0:
                stolen_points = descarted_points[stolen_idx, :-2]
                stolen_points = np.concatenate((stolen_points,stolen_cls),axis=1)
                pred_sub_pipe_ref = np.concatenate((pred_sub_pipe_ref,stolen_points),axis=0)

        for index in sorted(descart_valves_list, reverse=True):
            del instances_ref_proj_valve_list[index]

        instances_ref_pipe_list, _, _  = get_instances.get_instances(pred_sub_pipe_ref, self.dim, self.rad_p, self.min_p_p)
        # TODO info_pipes_list = get_info(instances_ref_pipe_list, method="skeleton")
        # TODO descart_pipes_list = ...  metrics to discart pipes
        # TODO merge info_valves and info_pipes into info
        # TODO SUMAR X Y Z MINIMO A TODAS LAS POSICIONES X Y Z DE  INFO PIPES Y VALVES

        t4 = rospy.Time.now()

        # TODO publish info

        i = len(instances_ref_proj_valve_list)

        if len(instances_ref_proj_valve_list)>0:
            instances_ref_proj_valve = np.vstack(instances_ref_proj_valve_list)
        if len(instances_ref_pipe_list)>0:
            instances_ref_pipe = np.vstack(instances_ref_pipe_list)
            instances_ref_pipe[:,7] = instances_ref_pipe[:,7]+i

        if len(instances_ref_proj_valve_list)>0 and len(instances_ref_pipe_list)>0:
            instances_ref = np.concatenate((instances_ref_proj_valve, instances_ref_pipe), axis=0)
        elif len(instances_ref_proj_valve_list)==0 and len(instances_ref_pipe_list)>0:
            instances_ref = instances_ref_pipe
        elif len(instances_ref_proj_valve_list)>0 and len(instances_ref_pipe_list)==0:
            instances_ref = instances_ref_proj_valve
        else:
            instances_ref = None

        if instances_ref is None: # if instances were not found
            print("no isntances found")
            return

        # Publish
        for i in range(pred_sub.shape[0]):
            color = self.label2color[pred_sub[i,6]]
            pred_sub[i,3] = color[0]
            pred_sub[i,4] = color[1]
            pred_sub[i,5] = color[2]

        for i in range(instances_ref.shape[0]):
            color = self.col_inst[instances_ref[i,7]]
            instances_ref[i,3] = color[0]
            instances_ref[i,4] = color[1]
            instances_ref[i,5] = color[2]

        pc_np_base[:, 0:3] += xyz_min  # return to initial position
        pred_sub[:, 0:3] += xyz_min  # return to initial position
        instances_ref[:, 0:3] += xyz_min  # return to initial position

        pc_base = self.array2pc(header, pc_np_base)
        pc_seg = self.array2pc(header, pred_sub)
        pc_inst = self.array2pc(header, instances_ref)
        self.pub_pc_base.publish(pc_base)
        self.pub_pc_seg.publish(pc_seg)
        self.pub_pc_inst.publish(pc_inst)

        t5 = rospy.Time.now()

        time_read = t1-t0
        time_blocks = t2-t1
        time_inferference = t3-t2
        time_instaces = t4-t3
        time_publish = t5-t4
        time_total = t5-t0


        rospy.loginfo('[%s]: Pc processing took %.2f seconds. Split into:', self.name, time_total.secs + time_total.nsecs*1e-9)
        rospy.loginfo('[%s]: Reading --- %.2f seconds (%i%%)', self.name, time_read.secs + time_read.nsecs*1e-9, (time_read/time_total)*100)
        rospy.loginfo('[%s]: Blocks ---- %.2f seconds (%i%%)', self.name, time_blocks.secs + time_blocks.nsecs*1e-9, (time_blocks/time_total)*100)
        rospy.loginfo('[%s]: Inference - %.2f seconds (%i%%)', self.name, time_inferference.secs + time_inferference.nsecs*1e-9, (time_inferference/time_total)*100)
        rospy.loginfo('[%s]: Instances - %.2f seconds (%i%%)', self.name, time_instaces.secs + time_instaces.nsecs*1e-9, (time_instaces/time_total)*100)
        rospy.loginfo('[%s]: Publish --- %.2f seconds (%i%%)', self.name, time_publish.secs + time_publish.nsecs*1e-9, (time_publish/time_total)*100)



    
    def pc2array(self, ros_pc):
        gen = pc2.read_points(ros_pc, skip_nans=True)
        pc_np = np.array(list(gen))

        if pc_np.size > 0:

            rgb_list = list()

            for rgb in pc_np[...,3]:
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f' ,rgb)
                i = struct.unpack('>l',s)[0]
                # you can get back the float value by the inverse operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000)>> 16
                g = (pack & 0x0000FF00)>> 8
                b = (pack & 0x000000FF)
                rgb_np = np.array([r,g,b])
                rgb_list.append(rgb_np)

            rgb = np.vstack(rgb_list)

            pc_np = np.delete(pc_np, 3, 1) 
            pc_np = np.concatenate((pc_np, rgb), axis=1)
        return pc_np

    def array2pc(self, header, array):


        fields =   [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgba', 12, PointField.UINT32, 1)]
        
        points = list()

        for i, p in enumerate(array):
            r = int(p[3])
            g = int(p[4])
            b = int(p[5])
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]

            p_rgb = [p[0], p[1], p[2], rgb]
            points.append(p_rgb)

        pc = pc2.create_cloud(header, fields, points)
        return pc


    def evaluate(self, data, label, xyz_max):

        is_training = False

        label = np.squeeze(label)

        num_batches = data.shape[0] // self.batch_size

        pred_label_list =list()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx+1) * self.batch_size
            
            feed_dict = {self.ops['pointclouds_pl']: data[start_idx:end_idx, :, :],
                        self.ops['labels_pl']: label[start_idx:end_idx],
                        self.ops['is_training_pl']: is_training}

            loss_val, pred_val = self.sess.run([self.ops['loss'], self.ops['pred_softmax']],feed_dict=feed_dict)

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

if __name__ == '__main__':
    try:
        rospy.init_node('flip_pc')
        Pointcloud_Seg(rospy.get_name())

        rospy.spin()
    except rospy.ROSInterruptException:
        pass