import os
import sys
import pcl
import time
import rospy
import ctypes
import struct
import numpy as np
from model import *
import indoor3d_util

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
        # Params
        self.period = 1
        self.batch_size = 1
        self.points_sub = 128
        self.block_sub = 0.1
        self.stride_sub = 0.1
        self.gpu_index = 0
        self.sub = 0.1
        
        
        self.model_path = "/home/miguel/Desktop/test_ros_subscriber/4_128_11_c7/model.ckpt"
        self.path_cls = "/home/miguel/Desktop/test_ros_subscriber/4.txt"
        self.classes, self.labels, self.label2color = indoor3d_util.get_info_classes(self.path_cls)


        self.init = False
        self.new_pc = False

        # Set image subscriber
        pc_sub = message_filters.Subscriber('/stereo_down/scaled_x2/points2_filtered', PointCloud2)
        #pc_sub = message_filters.Subscriber('/stereo_down/scaled_x2/points2', PointCloud2)
        info_sub = message_filters.Subscriber('/stereo_down/left/camera_info', CameraInfo)
        ts_image = message_filters.TimeSynchronizer([pc_sub, info_sub], 10)
        ts_image.registerCallback(self.cb_pc)

        # Set class image publisher
        self.pub_pc_used = rospy.Publisher("/stereo_down/scaled_x2/points2_used", PointCloud2, queue_size=4)
        self.pub_pc_seg = rospy.Publisher("/stereo_down/scaled_x2/points2_seg", PointCloud2, queue_size=4)

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
        t0 = rospy.Time.now()

        # New pc available
        if not self.new_pc:
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

        if pc_np.shape[0] < 4000:
            return

        pc_np[:, 2] *= -1  # flip Z axis

        xyz_min = np.amin(pc_np, axis=0)[0:3]   # get pointcloud mins
        pc_np[:, 0:3] -= xyz_min                # move pointcloud to origin
        xyz_max = np.amax(pc_np, axis=0)[0:3]   # get pointcloud maxs
        data_sub, label_sub = indoor3d_util.room2blocks_plus_normalized_parsed(pc_np,  xyz_max, self.points_sub, block_size=self.block_sub, stride=self.stride_sub, random_sample=False, sample_num=None, sample_aug=1) # subsample PC for evaluation
        
        with tf.Graph().as_default():
            pred_sub = self.evaluate(data_sub, label_sub, xyz_max)  # evaluate PC

        if pred_sub.size == 0:
            return

        pred_sub[:, 0:3] += xyz_min  # return to initial position

        pc_np_base = pred_sub.copy()
        for i in range(pred_sub.shape[0]):
            color = self.label2color[pred_sub[i,6]]
            pred_sub[i,3] = color[0]
            pred_sub[i,4] = color[1]
            pred_sub[i,5] = color[2]

        
        save = False
        if save:
            t = rospy.Time.now()
            fout_proj = open("/home/miguel/Desktop/test_ros_subscriber/out/pc_np"+str(t)+".obj", 'w')
            for i in range(pc_np.shape[0]):
                fout_proj.write('v %f %f %f %d %d %d\n' % (pc_np[i,0], pc_np[i,1], pc_np[i,2], pc_np[i,3], pc_np[i,4], pc_np[i,5]))


        pc_base = self.array2pc(header, pc_np_base)
        pc_seg = self.array2pc(header, pred_sub)

        # Publish
        self.pub_pc_used.publish(pc_base)
        self.pub_pc_seg.publish(pc_seg)

        time = rospy.Time.now()-t0
        rospy.loginfo('[%s]: Pc flipping took %s seconds', self.name, time.secs + time.nsecs*1e-9)


    
    def pc2array(self, ros_pc):
        gen = pc2.read_points(ros_pc, skip_nans=True)
        pc_np = np.array(list(gen))

        if pc_np.size > 0:

            if self.sub != 1:    # subsample pc_np
                n_idx_sub = int(pc_np.shape[0] * 0.1)
                idx_sub = np.random.choice(pc_np.shape[0], n_idx_sub, replace=False)
                pc_np = pc_np[idx_sub, 0:4]

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