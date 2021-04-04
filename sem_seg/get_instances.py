import os
import re
import sys
import math
import time
import argparse
import numpy as np
import open3d as o3d
from natsort import natsorted

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
    for n, p in enumerate(points):              # for each point to grow from

        #progress(n, len(points), status='growing')

        p1 = p[0:3]
        cls1 = p[6]
        for j, point2 in enumerate(data):       # for each point of data to grow over
            p2 = data[j,0:3]
            cls2 = data[j, 6]

            if cls1 == cls2:                    # if same class
                d = get_distance(p1,p2,dim)     # get distance
                if d < min_dist:                # if dist < thr 
                    new_idx.append(j)           # add idx

    new_idx = list(set(new_idx))

    return new_idx


def get_color(instance):

    colors = [[255, 255, 0],
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
              [255, 100, 0],
              [0, 255, 0],
              [0, 0, 255],
              [255, 0, 0],]

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


def refine_instances(instances, ref_data, rad_ref = 0.1):

    stolen_list = list()

    for i, inst in enumerate(instances):  # for each instnace

        stolen_points_list = list()
        stolen_idx = inst.shape[0]

        c_inst = inst[0,6]                      # get class 
        n_inst = inst[0,7]                      # get inst number
        central_point = np.mean(inst, axis=0)[0:3] # get central_point
        
        del_list_point = list()

        for k, point in enumerate(ref_data):        # for each point of ref_data
            p = point[0:3]                          # get point coordinates
            d = get_distance(central_point,p,2)     # calculate distance between point and central_point
            
            if d < rad_ref:                         # if distance lower than thr
                stolen_point = np.array([stolen_idx, point[6]]) # annotate idx from new instance and original class
                stolen_points_list.append(stolen_point)
                stolen_idx = stolen_idx +1
                point[6] = c_inst                   # point converted to actual class
                point = np.append(point, n_inst)    # point converted to actual instance number
                inst = np.vstack([inst, point])     # append point to actual instance
                del_list_point.append(k)            # save point index from ref_data to delete later

        stolen_list.append(stolen_points_list)
        instances[i] = inst                   # new  instance
        ref_data = np.delete(ref_data, del_list_point, 0) # del point from remaining pointcloud not in valves

    return instances, ref_data, stolen_list


def get_instances(data_label, dim, rad, min_points, ref=False, ref_data=0, ref_rad=0.1):

    n_inst = 1
    stolen_list = list()

    # get instances 
    instances = list()
    while data_label.size > 0:                                        # while there are  points
        actual_idx = [0]                                              # init idx
        actual_inst = data_label[actual_idx]                          # init actual inst
        inst = actual_inst                                            # init inst
        data_label = np.delete(data_label, actual_idx, axis=0)        # delete  points
        while actual_idx:                                             # while idx exists
            actual_idx = grow(data_label, actual_inst, rad, dim)      # idx grow
            actual_inst = data_label[actual_idx]                      # get new actual inst
            inst = np.vstack([inst, actual_inst])                     # append to inst
            data_label = np.delete(data_label, actual_idx, axis=0)    # delete  points
        if inst.shape[0] > min_points:                                # save instance if  bigger than min
            inst = np.insert(inst, 7, n_inst, axis=1)
            instances.append(inst)
            n_inst = n_inst + 1

    if ref:  # if ref -> refine isntance
        start = time.time()
        instances, ref_data, stolen_list = refine_instances(instances, ref_data, ref_rad)
        end = time.time()
        time_ref = end - start

    #sys.stdout.write('\n')

    #if len(instances)== 0:
        #print("NO INSTANCES FOUND")
    return instances, ref_data, stolen_list
        

def get_instances_o3d(data_label, dim, rad, min_points, ref=False, ref_data=0, ref_rad=0.1):

    data_label_copy = data_label.copy()
    if dim == 2:
        data_label_copy[...,2] = 0

    data_label_o3d = o3d.geometry.PointCloud()
    data_label_o3d.points = o3d.utility.Vector3dVector(data_label_copy[:,0:3])
    data_label_o3d.colors = o3d.utility.Vector3dVector(data_label_copy[:,3:6])

    labels = np.array(data_label_o3d.cluster_dbscan(eps=rad, min_points=10, print_progress=False))
    labels = labels +1

    instances_np = np.insert(data_label, 7, labels, axis=1)
    instances_np = instances_np[instances_np[:, -1] != 0]

    instances = list()
    for i in set(instances_np[..., 7]):
        inst = instances_np[np.where(instances_np[..., 7] == float(i))]
        if inst.shape[0]>min_points:
            instances.append(inst)

    stolen_list = list()
    if ref:  # if ref -> refine isntance
        instances, ref_data, stolen_list = refine_instances(instances, ref_data, ref_rad)

    #sys.stdout.write('\n')

    if len(instances)== 0:
        print("NO INSTANCES FOUND")
    return instances, ref_data, stolen_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parser.add_argument('--dim_v', default=2, help='dim to calculate distance for growing (2 or 3).')
    parser.add_argument('--rad_v', default=0.03, help='max rad for growing (2 or 3).')
    parser.add_argument('--dim_p', default=2, help='dim to calculate distance for growing (2 or 3).')
    parser.add_argument('--rad_p', default=0.03, help='max rad for growing (2 or 3).')
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


        gt_pipe = gt[gt[:,6] == [labels["pipe"]]]       # get data label pipe
        gt_valve = gt[gt[:,6] == [labels["valve"]]]     # get data label pipe
        pred_pipe = pred[pred[:,6] == [labels["pipe"]]]       # get data label pipe
        pred_valve = pred[pred[:,6] == [labels["valve"]]]     # get data label pipe


        gt_inst_valve_list, _, _ = get_instances(gt_valve, dim_v, rad_v, min_points)
        gt_inst_pipe_list, _, _  = get_instances(gt_pipe, dim_p, rad_p, min_points)
        i = len(gt_inst_valve_list)

        if len(gt_inst_valve_list)>0:
            gt_inst_valve = np.vstack(gt_inst_valve_list)
        if len(gt_inst_pipe_list):
            gt_inst_pipe = np.vstack(gt_inst_pipe_list)
            gt_inst_pipe[:,7] = gt_inst_pipe[:,7]+i

        if len(gt_inst_valve_list)>0 and len(gt_inst_pipe_list)>0:
            gt_inst = np.concatenate((gt_inst_valve, gt_inst_pipe), axis=0)
        elif len(gt_inst_valve_list)==0 and len(gt_inst_pipe_list)>0:
            gt_inst = gt_inst_pipe
        elif len(gt_inst_valve_list)>0 and len(gt_inst_pipe_list)==0:
            gt_inst = gt_inst_valve
        else:
            gt_inst = None

        pred_inst_valve_list, _, _  = get_instances(pred_valve, dim_v, rad_v, min_points)
        pred_inst_pipe_list, _, _  = get_instances(pred_pipe, dim_p, rad_p, min_points)
        i = len(pred_inst_valve_list)

        if len(pred_inst_valve_list)>0:
            pred_inst_valve = np.vstack(pred_inst_valve_list)
        if len(pred_inst_pipe_list)>0:
            pred_inst_pipe = np.vstack(pred_inst_pipe_list)
            pred_inst_pipe[:,7] = pred_inst_pipe[:,7]+i

        if len(pred_inst_valve_list)>0 and len(pred_inst_pipe_list)>0:
            pred_inst = np.concatenate((pred_inst_valve, pred_inst_pipe), axis=0)
        elif len(pred_inst_valve_list)==0 and len(pred_inst_pipe_list)>0:
            pred_inst = pred_inst_pipe
        elif len(pred_inst_valve_list)>0 and len(pred_inst_pipe_list)==0:
            pred_inst = pred_inst_valve
        else:
            pred_inst = None

        pred_pipe2 = np.copy(pred_pipe)

        pred_inst_ref_valve_list, pred_pipe_ref, stolen_list = get_instances(pred_valve, dim_v, rad_v, min_points, ref=True, ref_data = pred_pipe, ref_rad = 0.1)
        pred_inst_ref_pipe_list, _, _  = get_instances(pred_pipe_ref, dim_p, rad_p, min_points)
        i = len(pred_inst_ref_valve_list)

        if len(pred_inst_ref_valve_list)>0:
            pred_inst_ref_valve = np.vstack(pred_inst_ref_valve_list)
        if len(pred_inst_ref_pipe_list)>0:
            pred_inst_ref_pipe = np.vstack(pred_inst_ref_pipe_list)
            pred_inst_ref_pipe[:,7] = pred_inst_ref_pipe[:,7]+i

        if len(pred_inst_ref_valve_list)>0 and len(pred_inst_ref_pipe_list)>0:
            pred_inst_ref = np.concatenate((pred_inst_ref_valve, pred_inst_ref_pipe), axis=0)
        elif len(pred_inst_ref_valve_list)==0 and len(pred_inst_ref_pipe_list)>0:
            pred_inst_ref = pred_inst_ref_pipe
        elif len(pred_inst_ref_valve_list)>0 and len(pred_inst_ref_pipe_list)==0:
            pred_inst_ref = pred_inst_ref_valve
        else:
            pred_inst_ref = None


        pred_inst_ref2_valve_list, pred_pipe_ref2, stolen_list = get_instances(pred_valve, dim_v, rad_v, min_points, ref=True, ref_data = pred_pipe2, ref_rad = 0.1)

        matches_list = [None, 2, 1, 2, 2, 1]
        descart_list = [i for i, x in enumerate(matches_list) if x == None]

        for i, idx in enumerate(descart_list):
            descarted_points = np.vstack(pred_inst_ref2_valve_list[idx])
            stolen_idx = list(np.vstack(stolen_list[idx])[:,0].astype(int))
            stolen_cls = np.vstack(stolen_list[idx])[:,1].astype(int)
            stolen_cls = stolen_cls.reshape(stolen_cls.shape[0],1)
            if len(stolen_idx)>0:
                stolen_points = descarted_points[stolen_idx, :-2]
                stolen_points = np.concatenate((stolen_points,stolen_cls),axis=1)
                pred_pipe_ref2 = np.concatenate((pred_pipe_ref2,stolen_points),axis=0)

        for index in sorted(descart_list, reverse=True):
            del pred_inst_ref2_valve_list[index]

        pred_inst_ref2_pipe_list, _, _  = get_instances(pred_pipe_ref2, dim_p, rad_p, min_points)
        i = len(pred_inst_ref2_valve_list)

        if len(pred_inst_ref2_valve_list)>0:
            pred_inst_ref2_valve = np.vstack(pred_inst_ref2_valve_list)
        if len(pred_inst_ref2_pipe_list)>0:
            pred_inst_ref2_pipe = np.vstack(pred_inst_ref2_pipe_list)
            pred_inst_ref2_pipe[:,7] = pred_inst_ref2_pipe[:,7]+i

        if len(pred_inst_ref2_valve_list)>0 and len(pred_inst_ref2_pipe_list)>0:
            pred_inst_ref2 = np.concatenate((pred_inst_ref2_valve, pred_inst_ref2_pipe), axis=0)
        elif len(pred_inst_ref2_valve_list)==0 and len(pred_inst_ref2_pipe_list)>0:
            pred_inst_ref2 = pred_inst_ref2_pipe
        elif len(pred_inst_ref2_valve_list)>0 and len(pred_inst_ref2_pipe_list)==0:
            pred_inst_ref2 = pred_inst_ref2_valve
        else:
            pred_inst_ref2 = None




        file_path_out = os.path.join(path_infer, name + "_gt_inst.ply")
        if gt_inst is not None:
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

        file_path_out = os.path.join(path_infer, name + "_pred_inst_ref2.ply")
        if pred_inst_ref2 is not None:
            write_ply(pred_inst_ref2, file_path_out)
        else:
            write_ply(null, file_path_out)


        # TODO print de ref2