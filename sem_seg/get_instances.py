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

 - python3 get_instances.py --path_run "path/to/run/" --path_cls "path/to/classes.txt" --test_name name --dim 2 --rad 0.03 --min_points 10 --rad_ref 0.1
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


def get_distance(p1,p2, dim):
    if dim == 2:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2))
    if dim == 3:
        d = math.sqrt(((p2[0]-p1[0])**2)+((p2[1]-p1[1])**2)+((p2[2]-p1[2])**2))
    return d


def grow(data, points, min_dist, dim):

    new_idx = list()
    for n, p in enumerate(points):

        progress(n, len(points), status='growing')

        p1 = p[0:3]
        cls1 = p[6]
        for j, point2 in enumerate(data):
            p2 = data[j,0:3]
            cls2 = data[j, 6]

            if cls1 == cls2:
                d = get_distance(p1,p2,dim)
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


def refine_instances(instances_valve, data_label_pipe, rad_ref = 0.1):

    for i, inst in enumerate(instances_valve):  # for each valve
        c_inst = inst[0,6]                      # get class (could be fixed)
        n_inst = inst[0,7]                      # get inst number
        central_point = np.mean(inst, axis=0)[0:3] # get central_point
        
        del_list_point = list()

        for k, point in enumerate(data_label_pipe): # for each point of the remaining pointcloud not in valves
            p = point[0:3]                          # get point coordinates
            d = get_distance(central_point,p,2)     # calculate distance between point and central_point
            
            if d < rad_ref:                         # if distance lower than thr
                point[6] = c_inst                   # point converted to class valve
                point = np.append(point, n_inst)    # point converted to actual instance number
                inst = np.vstack([inst, point])     # append point to actual instance
                del_list_point.append(k)            # save point index from remaining pointcloud not in valves
        instances_valve[i] = inst                   # new valve instance
        data_label_pipe = np.delete(data_label_pipe, del_list_point, 0) # del point from remaining pointcloud not in valves

    return instances_valve, data_label_pipe


def get_instances(data_label, labels, dim_p, rad_p, dim_v, rad_v, min_points, rad_ref = 0.1, ref=False):

    data_label_pipe = data_label[data_label[:,6] == [labels["pipe"]]]       # get data label pipe
    data_label_valve = data_label[data_label[:,6] == [labels["valve"]]]     # get data label pipe

    n_inst = 1

    # get instances valves
    instances_valve = list()
    while data_label_valve.size > 0:                                        # while there are valve points
        actual_idx = [0]                                                    # init idx
        actual_inst = data_label_valve[actual_idx]                          # init actual inst
        inst = actual_inst                                                  # init inst
        data_label_valve = np.delete(data_label_valve, actual_idx, axis=0)  # delete valve points
        while actual_idx:                                                   # while idx exists
            actual_idx = grow(data_label_valve, actual_inst, rad_v, dim_v)  # idx grow
            actual_inst = data_label_valve[actual_idx]                      # get new actual inst
            inst = np.vstack([inst, actual_inst])                           # append to inst
            data_label_valve = np.delete(data_label_valve, actual_idx, axis=0) # delete valve points
        if inst.shape[0] > min_points:                                      # save instance if  bigger than min
            inst = np.insert(inst, 7, n_inst, axis=1)
            instances_valve.append(inst)
            n_inst = n_inst + 1

    sys.stdout.write('\n')

    if ref:  # if ref -> refine isntance
        start = time.time()
        instances_valve, data_label_pipe = refine_instances(instances_valve, data_label_pipe, rad_ref)
        end = time.time()
        time_ref = end - start

    # get instances pipe
    instances_pipe = list()
    while data_label_pipe.size > 0:
        actual_idx = [0]
        actual_inst = data_label_pipe[actual_idx]
        inst = actual_inst
        data_label_pipe = np.delete(data_label_pipe, actual_idx, axis=0)
        while actual_idx:
            actual_idx = grow(data_label_pipe, actual_inst, rad_p, dim_p)
            actual_inst = data_label_pipe[actual_idx]
            inst = np.vstack([inst, actual_inst])
            data_label_pipe = np.delete(data_label_pipe, actual_idx, axis=0)
        if inst.shape[0] > min_points:
            inst = np.insert(inst, 7, n_inst, axis=1)
            instances_pipe.append(inst)
            n_inst = n_inst + 1

    sys.stdout.write('\n')

    instances = instances_valve + instances_pipe
    if len(instances)!= 0:
        instances = np.vstack(instances)  # deleting this line, the ouput becomes a list of numpys
    else:
        print("NO INSTANCES FOUND")
        instances = None

    return instances
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parser.add_argument('--dim_v', default=2, help='dim to calculate distance for growing (2 or 3).')
    parser.add_argument('--rad_v', default=0.02, help='max rad for growing (2 or 3).')
    parser.add_argument('--dim_p', default=2, help='dim to calculate distance for growing (2 or 3).')
    parser.add_argument('--rad_p', default=0.02, help='max rad for growing (2 or 3).')
    parser.add_argument('--min_points', default=0, help='min points to not delete an instance')
    parser.add_argument('--rad_ref', default=0.1, help='rad valve')
    parser.add_argument('--test_name', help='name of the test')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    dim_v = int(parsed_args.dim_v)
    rad_v = float(parsed_args.rad_v)
    dim_p = int(parsed_args.dim_p)
    rad_p = float(parsed_args.rad_p)
    min_points = int(parsed_args.min_points)
    rad_ref = float(parsed_args.rad_ref)
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

        gt_inst = get_instances(gt, labels, dim_p, rad_p, dim_v, rad_v, min_points)
        pred_inst = get_instances(pred, labels, dim_p, rad_p, dim_v, rad_v, min_points)
        pred_inst_ref = get_instances(pred, labels, dim_p, rad_p, dim_v, rad_v, min_points, rad_ref, ref=True)

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