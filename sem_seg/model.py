import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32,
                   shape=(batch_size, num_point, 9))
  labels_pl = tf.placeholder(tf.int32,
                shape=(batch_size, num_point))
  return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
  """ ConvNet baseline, input is BxNx9 gray image """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, -1)

  k = 30

  adj = tf_util.pairwise_distance(point_cloud[:, :, 6:])
  nn_idx = tf_util.knn(adj, k=k) # (batch, num_points, k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
  
  out2 = tf_util.conv2d(out1, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_max_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)
  net_mean_1 = tf.reduce_mean(out2, axis=-2, keep_dims=True)

  out3 = tf_util.conv2d(tf.concat([net_max_1, net_mean_1], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  adj = tf_util.pairwise_distance(tf.squeeze(out3, axis=-2))
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(out3, nn_idx=nn_idx, k=k)

  out4 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
  
  net_max_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)
  net_mean_2 = tf.reduce_mean(out4, axis=-2, keep_dims=True)

  out5 = tf_util.conv2d(tf.concat([net_max_2, net_mean_2], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)

  adj = tf_util.pairwise_distance(tf.squeeze(out5, axis=-2))
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(out5, nn_idx=nn_idx, k=k)

  out6 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv6', bn_decay=bn_decay, is_dist=True)
  
  net_max_3 = tf.reduce_max(out6, axis=-2, keep_dims=True)
  net_mean_3 = tf.reduce_mean(out6, axis=-2, keep_dims=True)

  out7 = tf_util.conv2d(tf.concat([net_max_3, net_mean_3], axis=-1), 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out8 = tf_util.conv2d(tf.concat([out3, out5, out7], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='adj_conv8', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out8, [num_point,1], padding='VALID', scope='maxpool')

  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand, 
                                     net_max_1,
                                     net_mean_1,
                                     out3,
                                     net_max_2,
                                     net_mean_2,
                                     out5,
                                     net_max_3,
                                     net_mean_3,
                                     out7,
                                     out8])

  # CONV 
  net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
  net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
  net = tf_util.conv2d(net, 13, [1,1], padding='VALID', stride=[1,1],
             activation_fn=None, scope='seg/conv3', is_dist=True)
  net = tf.squeeze(net, [2])

  return net

def get_loss(pred, label):
  """ pred: B,N,13; label: B,N """
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  return tf.reduce_mean(loss)
