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

def get_distance(p1,p2, dimensions):
    if dimensions == 2:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    if dimensions == 3:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2)+((p2[2]-p1[2])**2))
    return d

def grow(data, points, min_dist, dimensions):

    new_idx = list()
    for n, p in enumerate(points):

        progress(n, len(points), status='growing')

        p1 = p[0:3]
        cls1 = p[3]
        for j, point2 in enumerate(data):
            p2 = data[j,0:3]
            cls2 = data[j, 3]

            if cls1 == cls2:
                d = get_distance(p1,p2,dimensions)
                if d < min_dist:
                    new_idx.append(j)

    new_idx = list(set(new_idx))

    return new_idx


def get_color(instance):

    colors = [[0, 255, 0],
              [0, 0, 255],
              [255, 0, 0],
              [255, 255, 0],
              [255, 0, 255],
              [0, 255, 255],
              [0, 128, 0],
              [0, 0, 128],
              [128, 0, 0]]

    n = int(instance/len(colors))
    i = instance-(len(colors)*n)
    color = colors[i-1]
    return color


def write_ply(data, path_out):

    f = open(path_out, 'w')

    f.write("ply" + '\n')
    f.write("format ascii 1.0" + '\n')
    f.write("comment VCGLIB generated" + '\n')
    f.write("element vertex " + str(data.shape[0]) + '\n')
    f.write("property float x" + '\n')
    f.write("property float y" + '\n')
    f.write("property float z" + '\n')
    f.write("property uchar red" + '\n')
    f.write("property uchar green" + '\n')
    f.write("property uchar blue" + '\n')
    f.write("property int class" + '\n')
    f.write("property int inst" + '\n')
    f.write("element face 0" + '\n')
    f.write("property list uchar int vertex_indices" + '\n')
    f.write("end_header" + '\n')

    for row in range(data.shape[0]):
        color = get_color(int(data[row, 4]))
        line = ' '.join(map(str, data[row, :-2])) + ' ' + ' '.join(map(str, color)) + ' ' + str(int(data[row, 3]))+ ' ' + str(int(data[row, 4])) +'\n'
        f.write(line)

    f.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parser.add_argument('--dim', default=3, help='dimensions to calculate distance for growing (2 or 3).')
    parser.add_argument('--rad', default=0.03, help='max radius for growing (2 or 3).')
    parser.add_argument('--min_p', default=0, help='min points to not delete an instance')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    dimensions = int(parsed_args.dim)
    radius = float(parsed_args.rad)
    min_p = int(parsed_args.min_p)

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

        dels = [labels["floor"],labels["vessel"],labels["block"]]
        for i in (dels):
            gt = gt[gt[:,3] != i]
            pred = pred[pred[:,3] != i]

        gt_instances = list()
        pred_instances = list()

        gt_aux = gt.copy()
        pred_aux = pred.copy()

        n_inst = 1

        while gt_aux.size > 0:
            actual_idx = [0]
            actual_inst = gt_aux[actual_idx]
            inst = gt_aux[actual_idx]
            gt_aux = np.delete(gt_aux, actual_idx, axis=0)
            while actual_idx:
                actual_idx = grow(gt_aux, actual_inst, radius, dimensions)
                actual_inst = gt_aux[actual_idx]
                inst = np.vstack([inst, gt_aux[actual_idx]])
                gt_aux = np.delete(gt_aux, actual_idx, axis=0)
            if inst.shape[0] > min_p:
                inst = np.insert(inst, 4, n_inst, axis=1)
                gt_instances.append(inst)
                n_inst = n_inst + 1

        n_inst = 1

        while pred_aux.size > 0:
            actual_idx = [0]
            actual_inst = pred_aux[actual_idx]
            inst = pred_aux[actual_idx]
            pred_aux = np.delete(pred_aux, actual_idx, axis=0)
            while actual_idx:
                actual_idx = grow(pred_aux, actual_inst, radius, dimensions)
                actual_inst = pred_aux[actual_idx]
                inst = np.vstack([inst, pred_aux[actual_idx]])
                pred_aux = np.delete(pred_aux, actual_idx, axis=0)
            if inst.shape[0] > min_p:
                inst = np.insert(inst, 4, n_inst, axis=1)
                pred_instances.append(inst)
                n_inst = n_inst + 1

        sys.stdout.write('\n')
        # create plys

        data_gt = np.vstack(gt_instances)  # set instance number as data[...,5]
        file_path_out = os.path.join(path_infer, name + "_gt_inst.ply")
        write_ply(data_gt, file_path_out)

        data_pred = np.vstack(pred_instances)  # set instance number as data[...,5]
        file_path_out = os.path.join(path_infer, name + "_pred_inst.ply")
        write_ply(data_pred, file_path_out)













if __name__ == "__main__":
    main()
