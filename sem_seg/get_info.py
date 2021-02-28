import os
import re
import numpy as np
import math
import argparse
import sys
from natsort import natsorted
import time
'''
script to evaluate a model

execution example:

INPUT
    PC:        X Y Z R G B C
OUTPUT
    INSTANCES: X Y Z R G B C I

 - python3 get_info.py --path_run "path/to/run/" --path_cls "path/to/classes.txt" --test_name name
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


def get_distance(p1,p2, dim):
    if dim == 2:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    if dim == 3:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2)+((p2[2]-p1[2])**2))
    return d


def get_info(instances, method, models=0):

    if method == "skeleton":
        info = get_info_skeleton(instances)
    elif method == "matching":
        info = get_info_matching(instances, models)



    return info
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parser.add_argument('--test_name', help='name of the test')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    test_name = parsed_args.test_name

    path_infer = os.path.join(path_run, 'dump_' + test_name)
    classes, labels, label2color = get_info_classes(path_cls)


    for file_name in natsorted(os.listdir(path_infer)):

        if "_pred_inst_ref" in file:

            print("evaluating case: " + file_name)

            inst_ref = np.loadtxt(file_name)   # TODO CAMBIAR PARA QUE ACABE SIENDO NUMPY CON X Y Z R G B C I


            instances_pipe = inst_ref[inst_ref[:,6] == [labels["pipe"]]]       # get data label pipe
            instances_valve = inst_ref[inst_ref[:,6] == [labels["valve"]]]     # get data label pipe
            models = 0 # MODELS HA DE SER UNA LISTA DE NUMPYS X Y Z R G B DE LOS DIFERENTES TIPOS DE VALVULAS CON QUE SE QUIERA HACER MATCHING

            info_pipe = get_info(inst_ref, method="skeleton")
            info_valve = get_info(inst_ref, method="matching", models)
