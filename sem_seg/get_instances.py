import os
import re
import numpy as np
import argparse
import sys
from natsort import natsorted
from sklearn.preprocessing import normalize
'''
script to evaluate a model

execution example:
 - python3 evaluate.py --path_run "/home/miguel/Desktop/pipes/dgcnn/sem_seg/RUNS/valve_test/" --path_cls "/home/miguel/Desktop/pipes/data/valve_test/classes.txt"
'''

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


def main():

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the folder.')
    parser.add_argument('--path_cls', help='path to the folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    '''

    path_run = "/home/miguel/Desktop/pipes/dgcnn/sem_seg/RUNS/valve_test/"
    path_cls = "/home/miguel/Desktop/pipes/data/valve_test/classes.txt"

    path_infer = os.path.join(path_run, 'dump')

    classes, labels, label2color = get_info_classes(path_cls)

    files = natsorted(os.listdir(path_infer))
    cases = [s for s in files if s.endswith(".obj")]
    names = natsorted(set([re.split("[.\_]+",string)[0] for string in cases]))

    n_classes = len(classes)
    cnf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for name in names:

        print("evaluating case: " + name)
        path_gt = os.path.join(path_infer, name + "_gt.txt")
        path_pred = os.path.join(path_infer, name + "_pred.txt")

        gt = np.loadtxt(path_gt)
        pred = np.loadtxt(path_pred)

        pred = np.delete(pred,[3,4,5,6],1)
        gt = np.delete(np.c_[pred,gt],3,1)



if __name__ == "__main__":
    main()
