import argparse
import subprocess
import tensorflow as tf
import numpy as np
from datetime import datetime
import json
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import provider
import part_seg_model as model

TOWER_NAME = 'tower'

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=2, help='The number of GPUs to use [default: 2]')
parser.add_argument('--batch', type=int, default=16, help='Batch Size per GPU during training [default: 32]')
parser.add_argument('--epoch', type=int, default=201, help='Epoch to run [default: 50]')
parser.add_argument('--point_num', type=int, default=2048, help='Point Number [256/512/1024/2048]')
parser.add_argument('--output_dir', type=str, default='train_results', help='Directory that stores all training logs and trained models')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
FLAGS = parser.parse_args()

hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')

# MAIN SCRIPT
point_num = FLAGS.point_num
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

# color_map_file = os.path.join(hdf5_data_dir, 'part_color_mapping.json')
# color_map = json.load(open(color_map_file, 'r'))

all_obj_cats_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
fin = open(all_obj_cats_file, 'r')
lines = [line.rstrip() for line in fin.readlines()]
all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
fin.close()

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_CATEGORIES = 16
NUM_PART_CATS = len(all_cats)

print('#### Batch Size Per GPU: {0}'.format(batch_size))
print('#### Point Number: {0}'.format(point_num))
print('#### Using GPUs: {0}'.format(FLAGS.num_gpu))

DECAY_STEP = 16881 * 20
DECAY_RATE = 0.5

LEARNING_RATE_CLIP = 1e-5

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.003
MOMENTUM = 0.9
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')

MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
  os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
  os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
  os.mkdir(SUMMARIES_FOLDER)

def printout(flog, data):
  print(data)
  flog.write(data + '\n')

def convert_label_to_one_hot(labels):
  label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
  for idx in range(labels.shape[0]):
    label_one_hot[idx, labels[idx]] = 1
  return label_one_hot

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
      if g is None:
        continue
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
  with tf.Graph().as_default(), tf.device('/cpu:0'):

    batch = tf.Variable(0, trainable=False)
    
    learning_rate = tf.train.exponential_decay(
            BASE_LEARNING_RATE,     # base learning rate
            batch * batch_size,     # global_var indicating the number of steps
            DECAY_STEP,             # step size
            DECAY_RATE,             # decay rate
            staircase=True          # Stair-case or continuous decreasing
            )
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP)
  
    bn_momentum = tf.train.exponential_decay(
          BN_INIT_DECAY,
          batch*batch_size,
          BN_DECAY_DECAY_STEP,
          BN_DECAY_DECAY_RATE,
          staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

    lr_op = tf.summary.scalar('learning_rate', learning_rate)
    batch_op = tf.summary.scalar('batch_number', batch)
    bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)

    trainer = tf.train.AdamOptimizer(learning_rate)

    # store tensors for different gpus
    tower_grads = []
    pointclouds_phs = []
    input_label_phs = []
    seg_phs =[]
    is_training_phs =[]

    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpu):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            pointclouds_phs.append(tf.placeholder(tf.float32, shape=(batch_size, point_num, 3))) # for points
            input_label_phs.append(tf.placeholder(tf.float32, shape=(batch_size, NUM_CATEGORIES))) # for one-hot category label
            seg_phs.append(tf.placeholder(tf.int32, shape=(batch_size, point_num))) # for part labels
            is_training_phs.append(tf.placeholder(tf.bool, shape=()))

            seg_pred = model.get_model(pointclouds_phs[-1], input_label_phs[-1], \
                is_training=is_training_phs[-1], bn_decay=bn_decay, cat_num=NUM_CATEGORIES, \
                part_num=NUM_PART_CATS, batch_size=batch_size, num_point=point_num, weight_decay=FLAGS.wd)


            loss, per_instance_seg_loss, per_instance_seg_pred_res  \
              = model.get_loss(seg_pred, seg_phs[-1])

            total_training_loss_ph = tf.placeholder(tf.float32, shape=())
            total_testing_loss_ph = tf.placeholder(tf.float32, shape=())

            seg_training_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_ph = tf.placeholder(tf.float32, shape=())
            seg_testing_acc_avg_cat_ph = tf.placeholder(tf.float32, shape=())

            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

        
            seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', seg_training_acc_ph)
            seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', seg_testing_acc_ph)
            seg_test_acc_avg_cat_op = tf.summary.scalar('seg_testing_acc_avg_cat', seg_testing_acc_avg_cat_ph)

            tf.get_variable_scope().reuse_variables()

            grads = trainer.compute_gradients(loss)

            tower_grads.append(grads)

    grads = average_gradients(tower_grads)

    train_op = trainer.apply_gradients(grads, global_step=batch)

    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=20)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    
    init = tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer())
    sess.run(init)

    train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

    train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    test_file_list = provider.getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)

    fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
    fcmd.write(str(FLAGS))
    fcmd.close()

    # write logs to the disk
    flog = open(os.path.join(LOG_STORAGE_PATH, 'log.txt'), 'w')

    def train_one_epoch(train_file_idx, epoch_num):
      is_training = True

      for i in range(num_train_file):
        cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[train_file_idx[i]])
        printout(flog, 'Loading train file ' + cur_train_filename)

        cur_data, cur_labels, cur_seg = provider.load_h5_data_label_seg(cur_train_filename)
        cur_data, cur_labels, order = provider.shuffle_data(cur_data, np.squeeze(cur_labels))
        cur_seg = cur_seg[order, ...]

        cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

        num_data = len(cur_labels)
        num_batch = num_data // (FLAGS.num_gpu * batch_size) # For all working gpus

        total_loss = 0.0
        total_seg_acc = 0.0

        for j in range(num_batch):
          begidx_0 = j * batch_size
          endidx_0 = (j + 1) * batch_size
          begidx_1 = (j + 1) * batch_size
          endidx_1 = (j + 2) * batch_size

          feed_dict = {
              # For the first gpu
              pointclouds_phs[0]: cur_data[begidx_0: endidx_0, ...], 
              input_label_phs[0]: cur_labels_one_hot[begidx_0: endidx_0, ...], 
              seg_phs[0]: cur_seg[begidx_0: endidx_0, ...],
              is_training_phs[0]: is_training, 
              # For the second gpu
              pointclouds_phs[1]: cur_data[begidx_1: endidx_1, ...], 
              input_label_phs[1]: cur_labels_one_hot[begidx_1: endidx_1, ...], 
              seg_phs[1]: cur_seg[begidx_1: endidx_1, ...],
              is_training_phs[1]: is_training, 
              }


          # train_op is for both gpus, and the others are for gpu_1
          _, loss_val, per_instance_seg_loss_val, seg_pred_val, pred_seg_res \
              = sess.run([train_op, loss, per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
              feed_dict=feed_dict)

          per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx_1: endidx_1, ...], axis=1)
          average_part_acc = np.mean(per_instance_part_acc)

          total_loss += loss_val
          total_seg_acc += average_part_acc

        total_loss = total_loss * 1.0 / num_batch
        total_seg_acc = total_seg_acc * 1.0 / num_batch

        lr_sum, bn_decay_sum, batch_sum, train_loss_sum, train_seg_acc_sum = sess.run(\
            [lr_op, bn_decay_op, batch_op, total_train_loss_sum_op, seg_train_acc_sum_op], \
            feed_dict={total_training_loss_ph: total_loss, seg_training_acc_ph: total_seg_acc})

        train_writer.add_summary(train_loss_sum, i + epoch_num * num_train_file)
        train_writer.add_summary(lr_sum, i + epoch_num * num_train_file)
        train_writer.add_summary(bn_decay_sum, i + epoch_num * num_train_file)
        train_writer.add_summary(train_seg_acc_sum, i + epoch_num * num_train_file)
        train_writer.add_summary(batch_sum, i + epoch_num * num_train_file)

        printout(flog, '\tTraining Total Mean_loss: %f' % total_loss)
        printout(flog, '\t\tTraining Seg Accuracy: %f' % total_seg_acc)

    def eval_one_epoch(epoch_num):
      is_training = False

      total_loss = 0.0
      total_seg_acc = 0.0
      total_seen = 0

      total_seg_acc_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.float32)
      total_seen_per_cat = np.zeros((NUM_CATEGORIES)).astype(np.int32)

      for i in range(num_test_file):
        cur_test_filename = os.path.join(hdf5_data_dir, test_file_list[i])
        printout(flog, 'Loading test file ' + cur_test_filename)

        cur_data, cur_labels, cur_seg = provider.load_h5_data_label_seg(cur_test_filename)
        cur_labels = np.squeeze(cur_labels)

        cur_labels_one_hot = convert_label_to_one_hot(cur_labels)

        num_data = len(cur_labels)
        num_batch = num_data // batch_size

        # Run on gpu_1, since the tensors used for evaluation are defined on gpu_1
        for j in range(num_batch):
          begidx = j * batch_size
          endidx = (j + 1) * batch_size
          feed_dict = {
              pointclouds_phs[1]: cur_data[begidx: endidx, ...], 
              input_label_phs[1]: cur_labels_one_hot[begidx: endidx, ...], 
              seg_phs[1]: cur_seg[begidx: endidx, ...],
              is_training_phs[1]: is_training}

          loss_val, per_instance_seg_loss_val, seg_pred_val, pred_seg_res \
              = sess.run([loss, per_instance_seg_loss, seg_pred, per_instance_seg_pred_res], \
              feed_dict=feed_dict)

          per_instance_part_acc = np.mean(pred_seg_res == cur_seg[begidx: endidx, ...], axis=1)
          average_part_acc = np.mean(per_instance_part_acc)

          total_seen += 1
          total_loss += loss_val
          
          total_seg_acc += average_part_acc

          for shape_idx in range(begidx, endidx):
            total_seen_per_cat[cur_labels[shape_idx]] += 1
            total_seg_acc_per_cat[cur_labels[shape_idx]] += per_instance_part_acc[shape_idx - begidx]

      total_loss = total_loss * 1.0 / total_seen
      total_seg_acc = total_seg_acc * 1.0 / total_seen

      test_loss_sum, test_seg_acc_sum = sess.run(\
          [total_test_loss_sum_op, seg_test_acc_sum_op], \
          feed_dict={total_testing_loss_ph: total_loss, \
          seg_testing_acc_ph: total_seg_acc})

      test_writer.add_summary(test_loss_sum, (epoch_num+1) * num_train_file-1)
      test_writer.add_summary(test_seg_acc_sum, (epoch_num+1) * num_train_file-1)

      printout(flog, '\tTesting Total Mean_loss: %f' % total_loss)
      printout(flog, '\t\tTesting Seg Accuracy: %f' % total_seg_acc)

      for cat_idx in range(NUM_CATEGORIES):
        if total_seen_per_cat[cat_idx] > 0:
          printout(flog, '\n\t\tCategory %s Object Number: %d' % (all_obj_cats[cat_idx][0], total_seen_per_cat[cat_idx]))
          printout(flog, '\t\tCategory %s Seg Accuracy: %f' % (all_obj_cats[cat_idx][0], total_seg_acc_per_cat[cat_idx]/total_seen_per_cat[cat_idx]))

    if not os.path.exists(MODEL_STORAGE_PATH):
      os.mkdir(MODEL_STORAGE_PATH)

    for epoch in range(TRAINING_EPOCHES):
      printout(flog, '\n<<< Testing on the test dataset ...')
      eval_one_epoch(epoch)

      printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, TRAINING_EPOCHES))

      train_file_idx = np.arange(0, len(train_file_list))
      np.random.shuffle(train_file_idx)

      train_one_epoch(train_file_idx, epoch)

      if epoch % 5 == 0:
        cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch)+'.ckpt'))
        printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

      flog.flush()

    flog.close()

if __name__=='__main__':
  train()
