import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--path_data', help='folder with train val data')
parser.add_argument('--cls', type=int, help='number of classes')
parser.add_argument('--num_gpu', type=int, default=1, help='the number of GPUs to use [default: 2]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training for each GPU [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parsed_args = parser.parse_args()

TOWER_NAME = 'tower'

path_data = parsed_args.path_data
cls = parsed_args.cls
BATCH_SIZE = parsed_args.batch_size
NUM_POINT = parsed_args.num_point
MAX_EPOCH = parsed_args.max_epoch
NUM_POINT = parsed_args.num_point
BASE_LEARNING_RATE = parsed_args.learning_rate
MOMENTUM = parsed_args.momentum
OPTIMIZER = parsed_args.optimizer
DECAY_STEP = parsed_args.decay_step
DECAY_RATE = parsed_args.decay_rate

LOG_DIR = parsed_args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp model.py %s' % (LOG_DIR)) 
os.system('cp train.py %s' % (LOG_DIR)) 
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(parsed_args)+'\n')

NUM_CLASSES = cls
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

path_train = os.path.join(path_data, 'train')
files_train = provider.getDataFiles(os.path.join(path_train, 'files.txt'))

data_batch_list = []
label_batch_list = []

for h5_filename in files_train:
    file_path = os.path.join(path_train, h5_filename)
    data_batch, label_batch = provider.loadDataFile(file_path)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
train_data = np.concatenate(data_batch_list, 0)
train_label = np.concatenate(label_batch_list, 0)
print(train_data.shape, train_label.shape)

path_val = os.path.join(path_data, 'val')
files_val = provider.getDataFiles(os.path.join(path_val, 'files.txt'))

data_batch_list = []
label_batch_list = []

for h5_filename in files_val:
    file_path = os.path.join(path_val, h5_filename)
    data_batch, label_batch = provider.loadDataFile(file_path)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)
val_data = np.concatenate(data_batch_list, 0)
val_label = np.concatenate(label_batch_list, 0)
print(val_data.shape, val_label.shape)

def log_string(out_str):
  LOG_FOUT.write(out_str+'\n')
  LOG_FOUT.flush()
  print(out_str)


def get_learning_rate(batch):
  learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            DECAY_STEP,          # Decay step.
            DECAY_RATE,          # Decay rate.
            staircase=True)
  learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
  return learning_rate        

def get_bn_decay(batch):
  bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            batch*BATCH_SIZE,
            BN_DECAY_DECAY_STEP,
            BN_DECAY_DECAY_RATE,
            staircase=True)
  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
  return bn_decay

def average_gradients(tower_grads):
  """Calculate average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been 
     averaged across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train():
  with tf.Graph().as_default(), tf.device('/gpu:0'):
    batch = tf.Variable(0, trainable=False)
    
    bn_decay = get_bn_decay(batch)
    tf.summary.scalar('bn_decay', bn_decay)

    learning_rate = get_learning_rate(batch)
    tf.summary.scalar('learning_rate', learning_rate)
    
    trainer = tf.train.AdamOptimizer(learning_rate)
    
    tower_grads = []
    pointclouds_phs = []
    labels_phs = []
    is_training_phs =[]

    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(parsed_args.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
      
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            pointclouds_phs.append(pointclouds_pl)
            labels_phs.append(labels_pl)
            is_training_phs.append(is_training_pl)
      
            pred = get_model(pointclouds_phs[-1], is_training_phs[-1], bn_decay=bn_decay)
            loss = get_loss(pred, labels_phs[-1])
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_phs[-1]))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            tf.get_variable_scope().reuse_variables()

            grads = trainer.compute_gradients(loss)

            tower_grads.append(grads)
    
    grads = average_gradients(tower_grads)

    train_op = trainer.apply_gradients(grads, global_step=batch)
    
    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=10)
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Add summary writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                  sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'val'))

    # Init variables for two GPUs
    init = tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer())
    sess.run(init)

    ops = {'pointclouds_phs': pointclouds_phs,
         'labels_phs': labels_phs,
         'is_training_phs': is_training_phs,
         'pred': pred,
         'loss': loss,
         'train_op': train_op,
         'merged': merged,
         'step': batch}

    loss_t_list = list()
    loss_val_list = list()
    acc_t_list = list()
    acc_val_list = list()

    for epoch in range(MAX_EPOCH):
      log_string('**** EPOCH %03d ****' % (epoch))
      sys.stdout.flush()
       
      loss_t, acc_t = train_one_epoch(sess, ops, train_writer)
      loss_t_list.append(loss_t)
      acc_t_list.append(acc_t)

      loss_val, acc_val = eval_one_epoch(sess, ops, val_writer)
      loss_val_list.append(loss_val)
      acc_val_list.append(acc_val)

      if loss_val == min(loss_val_list):
          best_sess = sess
          best_epoch = epoch

      stop = early_stopping(loss_t_list, loss_val_list, 1)
      if stop:
          log_string('early stopping')
          break

  log_string('save session at epoch %03d' % (best_epoch))
  # Save the variables to disk. 
  save_path = saver.save(best_sess, os.path.join(LOG_DIR, "model.ckpt"))
  log_string("Model saved in file: %s" % save_path)

def early_stopping(t_loss, v_loss, thr):

    stop = False

    if len(t_loss) > 9:

        a = 100 * ((v_loss[-1] / min(v_loss)) - 1)
        b = 1000 * ((sum(t_loss[-10:]) / (10 * min(t_loss[-10:]))) - 1)
        ab = a / b

        if ab > thr:
            stop = True

    return stop

def train_one_epoch(sess, ops, train_writer):
  """ ops: dict mapping from string to tf ops """
  is_training = True
  
  log_string('----')
  current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label) 
  
  file_size = current_data.shape[0]
  num_batches = file_size // (parsed_args.num_gpu * BATCH_SIZE) 
  
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  
  for batch_idx in range(num_batches):
    if batch_idx % 100 == 0:
      print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
    start_idx_0 = batch_idx * BATCH_SIZE
    end_idx_0 = (batch_idx+1) * BATCH_SIZE
    
    
    feed_dict = {ops['pointclouds_phs'][0]: current_data[start_idx_0:end_idx_0, :, :],
                 ops['labels_phs'][0]: current_label[start_idx_0:end_idx_0],
                 ops['is_training_phs'][0]: is_training}
    summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                     feed_dict=feed_dict)
    train_writer.add_summary(summary, step)
    pred_val = np.argmax(pred_val, 2)
    correct = np.sum(pred_val == current_label[start_idx_0:end_idx_0])
    total_correct += correct
    total_seen += (BATCH_SIZE*NUM_POINT)
    loss_sum += loss_val

  mean_loss = loss_sum / float(num_batches)
  accuracy = total_correct / float(total_seen)

  log_string('mean loss: %f' % (mean_loss))
  log_string('accuracy: %f' % (accuracy))

  return mean_loss, accuracy

def eval_one_epoch(sess, ops, val_writer):
  """ ops: dict mapping from string to tf ops """
  is_training = False
  total_correct = 0
  total_seen = 0
  loss_sum = 0
  total_seen_class = [0 for _ in range(NUM_CLASSES)]
  total_correct_class = [0 for _ in range(NUM_CLASSES)]
  
  log_string('----')
  current_data = val_data[:,0:NUM_POINT,:]
  current_label = np.squeeze(val_label)
  
  file_size = current_data.shape[0]
  num_batches = file_size // BATCH_SIZE
  
  for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = (batch_idx+1) * BATCH_SIZE

    feed_dict = {ops['pointclouds_phs'][0]: current_data[start_idx:end_idx, :, :],
                  ops['labels_phs'][0]: current_label[start_idx:end_idx],
                  ops['is_training_phs'][0]: is_training}
    summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
    val_writer.add_summary(summary, step)
    pred_val = np.argmax(pred_val, 2)
    correct = np.sum(pred_val == current_label[start_idx:end_idx])
    total_correct += correct
    total_seen += (BATCH_SIZE*NUM_POINT)
    loss_sum += (loss_val*BATCH_SIZE)
    for i in range(start_idx, end_idx):
      for j in range(NUM_POINT):
        l = current_label[i, j]
        total_seen_class[l] += 1
        total_correct_class[l] += (pred_val[i-start_idx, j] == l)
          
  
  mean_loss = loss_sum / float(total_seen/NUM_POINT)
  accuracy = total_correct / float(total_seen)
  
  
  log_string('eval mean loss: %f' % (mean_loss))
  log_string('eval accuracy: %f'% (accuracy))
  log_string('eval class acc: ' + str((np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))
  log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))))

  return mean_loss, accuracy
        

if __name__ == "__main__":
  train()
  LOG_FOUT.close()