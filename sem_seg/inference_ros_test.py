import os
import sys
import time
import rospy
import ctypes
import struct
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import PointField

import message_filters
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image, CameraInfo


class Pointcloud_Flip:
    def __init__(self, name):


        self.DUMMY_FIELD_PREFIX = '__'
        self.type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]

        self.pftype_to_nptype = dict(self.type_mappings)

        self.pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

        self.name = name
        # Params
        self.period = 1

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
        self.pub_pc_flip = rospy.Publisher("/stereo_down/scaled_x2/points2_flipped", PointCloud2, queue_size=4)

        # Set classification timer
        rospy.Timer(rospy.Duration(self.period), self.run)

        # CvBridge for image conversion
        self.bridge = CvBridge()


    def cb_pc(self, pc, info):
        self.pc = pc
        self.cam_info = info
        self.new_pc = True

    def set_model(self):
        z = 1
        # TODO set flipping ready!

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
                rospy.loginfo('[%s]: Start pc fliping', self.name)	
        except:
            rospy.logwarn('[%s]: There is no input pc to run the flipping', self.name)
            return

        # Set model
        if not self.init: 
            self.set_model()
            self.init = True

        # TODO flip pc
        print("i have pc, lets make ir array")

        pc_np = self.pc2array(pc, sub=1)

        if pc_np.shape[0] < 2000:
            print("not enough input points")
            return
            
        pc_np_flipped = pc_np.copy()
        print(pc_np.shape)
        pc_np_flipped[:, 2] *= -1  # flip Z axis
        flipped_pc = self.array2pc(header, pc_np_flipped)


        save = False
        if save:
            t = rospy.Time.now()
            fout_proj = open("/home/miguel/Desktop/test_ros_subscriber/out/pc_np"+str(t)+".obj", 'w')
            for i in range(pc_np.shape[0]):
                fout_proj.write('v %f %f %f %d %d %d\n' % (pc_np[i,0], pc_np[i,1], pc_np[i,2], pc_np[i,3], pc_np[i,4], pc_np[i,5]))




        # Publish
        self.pub_pc_used.publish(pc)
        self.pub_pc_flip.publish(flipped_pc)

        time = rospy.Time.now()-t0
        rospy.loginfo('[%s]: Pc flipping took %s seconds', self.name, time.secs + time.nsecs*1e-9)


    
    def pc2array(self, ros_pc, sub):
        gen = pc2.read_points(ros_pc, skip_nans=True)
        pc_np = np.array(list(gen))

        if pc_np.size > 0:
            if sub != 1:    # subsample pc_np
                n_idx_sub = int(pc_np.shape[0] * sub)
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




if __name__ == '__main__':
    try:
        rospy.init_node('flip_pc')
        Pointcloud_Flip(rospy.get_name())

        rospy.spin()
    except rospy.ROSInterruptException:
        pass