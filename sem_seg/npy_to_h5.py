import os
import numpy as np
import argparse
from natsort import natsorted
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import data_prep_util
import indoor3d_util

'''
Call:

python npy_to_h5.py --path_in ../data/npy/ --path_out ../data/h5/
'''

parser = argparse.ArgumentParser()
parser.add_argument('--path_in', help='path to the npy data folder.')
parser.add_argument('--path_out', help='path to save h5 folder.')
parser.add_argument('--num_point', default=4096, help='num points per block.')


parsed_args = parser.parse_args(sys.argv[1:])

path_in = parsed_args.path_in
path_out = parsed_args.path_out
NUM_POINT = int(parsed_args.num_point)


if not os.path.exists(path_out):
    os.mkdir(path_out)

# Constants
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
for root, dirs, files in os.walk(path_in): break
data_label_files = [os.path.join(root, file) for file in files]

filelist = open(os.path.join(path_out, 'filelist.txt'), 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0  # state: the next h5 file to save

def insert_batch(data, label, last_batch=False):

    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename = os.path.join(path_out, 'data_' + str(h5_index) + '.h5')
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename = os.path.join(path_out, 'data_' + str(h5_index) + '.h5')
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):  # for npy
    print(data_label_filename)
    data, label = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=0.1, stride=0.1,
                                                 random_sample=False, sample_num=None)
    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        filelist.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(data_label_files)-1)

filelist.close()
print("Total samples: {0}".format(sample_cnt))

allfiles = open(os.path.join(path_out, 'files.txt'), 'w')

for i in range(h5_index):
    h5_filename = os.path.join('data_' + str(i) + '.h5')
    allfiles.write(os.path.basename(h5_filename)+'\n')

allfiles.close()

