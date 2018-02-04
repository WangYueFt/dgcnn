import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def input_transform_net(edge_feature, is_training, bn_decay=None, K=3, is_dist=False):
  """ Input (XYZ) Transform Net, input is BxNx3 gray image
    Return:
      Transformation matrix of size 3xK """
  batch_size = edge_feature.get_shape()[0].value
  num_point = edge_feature.get_shape()[1].value

  # input_image = tf.expand_dims(point_cloud, -1)
  net = tf_util.conv2d(edge_feature, 64, [1,1],
             padding='VALID', stride=[1,1],
             bn=True, is_training=is_training,
             scope='tconv1', bn_decay=bn_decay, is_dist=is_dist)
  net = tf_util.conv2d(net, 128, [1,1],
             padding='VALID', stride=[1,1],
             bn=True, is_training=is_training,
             scope='tconv2', bn_decay=bn_decay, is_dist=is_dist)
  
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  
  net = tf_util.conv2d(net, 1024, [1,1],
             padding='VALID', stride=[1,1],
             bn=True, is_training=is_training,
             scope='tconv3', bn_decay=bn_decay, is_dist=is_dist)
  net = tf_util.max_pool2d(net, [num_point,1],
               padding='VALID', scope='tmaxpool')

  net = tf.reshape(net, [batch_size, -1])
  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                  scope='tfc1', bn_decay=bn_decay,is_dist=is_dist)
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                  scope='tfc2', bn_decay=bn_decay,is_dist=is_dist)

  with tf.variable_scope('transform_XYZ') as sc:
    # assert(K==3)
    with tf.device('/cpu:0'):
      weights = tf.get_variable('weights', [256, K*K],
                    initializer=tf.constant_initializer(0.0),
                    dtype=tf.float32)
      biases = tf.get_variable('biases', [K*K],
                   initializer=tf.constant_initializer(0.0),
                   dtype=tf.float32)
    biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
    transform = tf.matmul(net, weights)
    transform = tf.nn.bias_add(transform, biases)

  transform = tf.reshape(transform, [batch_size, K, K])
  return transform