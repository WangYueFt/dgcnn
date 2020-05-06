import os
import re
import numpy as np
import math
import argparse
import sys
from natsort import natsorted
from sklearn.preprocessing import normalize
'''
script to evaluate a model

execution example:
 - python3 evaluate.py --path_run "/home/miguel/Desktop/pipes/dgcnn/sem_seg/RUNS/valve_test/" --path_cls "/home/miguel/Desktop/pipes/data/valve_test/classes.txt"
'''

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def get_info_classes(cls_path):

    classes = []
    colors = []

    for line in open(cls_path):
        data = line.split()
        classes.append(data[0])
        colors.append([int(data[1]), int(data[2]), int(data[3])])

    labels = {cls: i for i, cls in enumerate(classes)}

    label2color = {classes.index(cls): colors[classes.index(cls)] for cls in classes}

    return classes, labels, label2color

def get_distance(p1,p2, diemnsions):
    if diemnsions == 2:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    if diemnsions == 3:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2)+((p2[2]-p1[2])**2))
    return d

def grow(data, idx, min_dist):

    new_idx = list()
    for n, i in enumerate(idx):

        progress(n, len(idx), status='growing')
        sys.stdout.write('\n')

        p1 = data[i,0:3]
        cls1 = data[i,3]
        for j, point2 in enumerate(data):
            p2 = data[j,0:3]
            cls2 = data[j, 3]

            if cls1 == cls2:
                d = get_distance(p1,p2,2)
                if d < min_dist:
                    new_idx.append(j)

    new_idx = list(set(new_idx)-set(idx))
    return new_idx

def main():

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the folder.')
    parser.add_argument('--path_cls', help='path to the folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    '''

    path_run = "/home/miguel/Desktop/pipes/dgcnn/sem_seg/RUNS/valve_test_test/"
    path_cls = "/home/miguel/Desktop/pipes/data/valve_test/classes.txt"

    path_infer = os.path.join(path_run, 'dump')

    classes, labels, label2color = get_info_classes(path_cls)

    files = natsorted(os.listdir(path_infer))
    cases = [s for s in files if s.endswith(".obj")]
    names = natsorted(set([re.split("[.\_]+",string)[0] for string in cases]))

    n_classes = len(classes)

    for name in names:

        print("evaluating case: " + name)
        path_gt = os.path.join(path_infer, name + "_gt.txt")
        path_pred = os.path.join(path_infer, name + "_pred.txt")

        gt = np.loadtxt(path_gt)
        pred = np.loadtxt(path_pred)

        pred = np.delete(pred,[3,4,5,6],1)
        gt = np.delete(np.c_[pred,gt],3,1)

        gt_instances = list()

        gt_aux = gt.copy()

        while gt_aux.size > 0:


            actual_idx = [0]
            actual_inst = gt_aux[actual_idx]
            np.delete(gt_aux, actual_idx, axis=0)

            while actual_idx:
                actual_idx = grow(gt_aux, actual_idx, 0.1)
                actual_inst = np.vstack([actual_inst, gt_aux[actual_idx]])
                np.delete(gt_aux, actual_idx, axis=0)
            gt_instances.append(actual_inst)










if __name__ == "__main__":
    main()
