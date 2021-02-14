import os
import re
import numpy as np
import math
import argparse
import sys
from natsort import natsorted
'''
script to evaluate a model

execution example:

INPUT
    PC:        X Y Z R G B C
OUTPUT
    INSTANCES: X Y Z R G B C I

 - python3 get_instances.py --path_run "path/to/run/" --path_cls "path/to/classes.txt" --test_name name --dim 2 --rad 0.03 --min_p 10 --rad_v 0.1
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
        cls1 = p[6]
        for j, point2 in enumerate(data):
            p2 = data[j,0:3]
            cls2 = data[j, 6]

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
              [128, 0, 0],
              [0, 100, 0],
              [0, 0, 100],
              [100, 0, 0],
              [100, 0, 255],
              [0, 255, 100],
              [255, 100, 0]]

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
        color = get_color(int(data[row, 7]))
        line = ' '.join(map(str, data[row, :-5])) + ' ' + ' '.join(map(str, color)) + ' ' + str(int(data[row, 6]))+ ' ' + str(int(data[row, 7])) +'\n'
        f.write(line)

    f.close()


def refine_instances(instances, labels, dimensions, radius, min_p, rad_v = 0.1):

    repeat = 0
    del_list_inst = list()

    for i, inst in enumerate(instances):

        c_inst = inst[0,6]
        n_inst = inst[0,7]

        if inst[0,6] == labels["valve"]:
            central_point = np.mean(inst, axis=0)[0:3]

            for j, inst2 in enumerate(instances):
                del_list_point = list()

                if inst2[0,6] == labels["pipe"]:

                    for k, point in enumerate(inst2):
                        p = point[0:3]
                        d = get_distance(central_point,p,2)
                        
                        if d < rad_v:
                            point[6] = c_inst
                            point[7] = n_inst
                            inst = np.vstack(point)
                            del_list_point.append(k)               
                            repeat = 1

            inst2 = np.delete(inst2, del_list_point, 0)
            if inst2.shape[0] < min_p:
                del_list_inst.append(j)
    
    set_del_list_inst = set(del_list_inst)
    for index in sorted(set_del_list_inst, reverse=True):
        del instances[index]
    
    if repeat == 1:
        
        instances = np.vstack(instances)
        instances= np.delete(instances,[7],1)

        ref_instances = list()

        n_inst = 1

        while instances.size > 0:
            actual_idx = [0]
            actual_inst = instances[actual_idx]
            inst = actual_inst
            instances = np.delete(instances, actual_idx, axis=0)
            while actual_idx:
                actual_idx = grow(instances, actual_inst, radius, dimensions)
                actual_inst = instances[actual_idx]
                inst = np.vstack([inst, actual_inst])
                instances = np.delete(instances, actual_idx, axis=0)
            if inst.shape[0] > min_p:
                inst = np.insert(inst, 7, n_inst, axis=1)
                ref_instances.append(inst)
                n_inst = n_inst + 1

        sys.stdout.write('\n')
    
        return ref_instances
    else:
        return instances


def get_instances(data_label, labels, radius, dimensions, min_p, rad_v = 0.1, ref=False):

    dels = [labels["floor"]] # ,labels["vessel"],labels["block"]]
    for i in (dels):
        data_label = data_label[data_label[:,6] != i]

    instances = list()

    n_inst = 1

    while data_label.size > 0:
        actual_idx = [0]
        actual_inst = data_label[actual_idx]
        inst = actual_inst
        data_label = np.delete(data_label, actual_idx, axis=0)
        while actual_idx:
            actual_idx = grow(data_label, actual_inst, radius, dimensions)
            actual_inst = data_label[actual_idx]
            inst = np.vstack([inst, actual_inst])
            data_label = np.delete(data_label, actual_idx, axis=0)
        if inst.shape[0] > min_p:
            inst = np.insert(inst, 7, n_inst, axis=1)
            instances.append(inst)
            n_inst = n_inst + 1

    sys.stdout.write('\n')

    if ref:
        instances = refine_instances(instances, labels, dimensions, radius, min_p, rad_v)
    
    if len(instances)!= 0:
        instances = np.vstack(instances)  # QUITANDO ESTO SE SACA LISTA Y SE PUEDE AHCER VSTACK FUERA
    else:
        print("NO INSTANCES FOUND")
        instances = None

    return instances
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parser.add_argument('--dim', default=2, help='dimensions to calculate distance for growing (2 or 3).')
    parser.add_argument('--rad', default=0.02, help='max radius for growing (2 or 3).')
    parser.add_argument('--min_p', default=0, help='min points to not delete an instance')
    parser.add_argument('--rad_v', default=0.1, help='rad valve')
    parser.add_argument('--test_name', help='name of the test')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    dimensions = int(parsed_args.dim)
    radius = float(parsed_args.rad)
    min_p = int(parsed_args.min_p)
    rad_v = float(parsed_args.rad_v)
    test_name = parsed_args.test_name

    path_infer = os.path.join(path_run, 'dump_' + test_name)

    classes, labels, label2color = get_info_classes(path_cls)

    files = natsorted(os.listdir(path_infer))
    cases = [s for s in files if s.endswith(".obj")]
    names = natsorted(set([re.split("[.\_]+",string)[0] for string in cases]))

    null = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])

    for name in names:

        print("evaluating case: " + name)
        path_gt = os.path.join(path_infer, name + "_gt.txt")
        path_pred = os.path.join(path_infer, name + "_pred.txt")

        gt = np.loadtxt(path_gt)
        pred = np.loadtxt(path_pred)

        gt = gt.reshape(gt.shape[0],1)

        pred = np.delete(pred,[6],1)
        gt = np.hstack([pred[...,0:6],gt])  

        gt_inst = get_instances(gt, labels, radius, dimensions, min_p)
        pred_inst = get_instances(pred, labels, radius, dimensions, min_p)
        pred_inst_ref = get_instances(pred, labels, radius, dimensions, min_p, rad_v, ref=True)

        file_path_out = os.path.join(path_infer, name + "_gt_inst.ply")
        if pred_inst is not None:
            write_ply(gt_inst, file_path_out)
        else:
            write_ply(null, file_path_out)
        
        file_path_out = os.path.join(path_infer, name + "_pred_inst.ply")
        if pred_inst is not None:
            write_ply(pred_inst, file_path_out)
        else:
            write_ply(null, file_path_out)

        file_path_out = os.path.join(path_infer, name + "_pred_inst_ref.ply")
        if pred_inst_ref is not None:
            write_ply(pred_inst_ref, file_path_out)
        else:
            write_ply(null, file_path_out)