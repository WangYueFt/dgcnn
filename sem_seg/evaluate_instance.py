
import os
import re
import numpy as np
import argparse
import sys
from natsort import natsorted
from plyfile import PlyData, PlyElement

'''
script to evaluate a model

execution example:
 - python3 evaluate.py --path_run "/home/miguel/Desktop/pipes/dgcnn/sem_seg/RUNS/valve_test/" --path_cls "/home/miguel/Desktop/pipes/data/valve_test/classes.txt"
'''


def read_ply(filename):

    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, c, i] for x,y,z,r,g,b,c,i in pc])



    ply = PlyData.read(filename)
    data = np.column_stack((ply.elements[0].data['x'], ply.elements[0].data['y'], ply.elements[0].data['z']))
    col = np.column_stack((ply.elements[0].data['red'], ply.elements[0].data['green'], ply.elements[0].data['blue']))


    return pc_array

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
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parser.add_argument('--iou_thr', help='min iou.')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    iou_thr = parsed.args.iou_thr
    '''

    path_run = "/home/miguel/Desktop/pipes/pointnet/sem_seg/RUNS/numpoints256_show2/"
    path_cls = "/home/miguel/Desktop/pipes/data/valve/classes.txt"
    iou_thr = 0.5

    path_infer = os.path.join(path_run, 'dump')

    classes, labels, label2color = get_info_classes(path_cls)

    files = natsorted(os.listdir(path_infer))
    cases = [s for s in files if s.endswith(".obj")]
    names = natsorted(set([re.split("[.\_]+", string)[0] for string in cases]))

    tp = 0
    fp = 0
    n_gt = 0

    for name in names:

        print("evaluating case: " + name)
        path_gt = os.path.join(path_infer, name + "_gt_inst.ply")
        path_pred = os.path.join(path_infer, name + "_pred_inst.ply")

        gt = read_ply(path_gt)
        pred = read_ply(path_pred)

        gt_list = list()
        for i in range(len(set(gt[..., 4]))):
            inst = gt[np.where(gt[..., 4] == float(i+1))]
            gt_list.append(inst)

        pred_list = list()
        for j in range(len(set(pred[..., 4]))):
            inst = pred[np.where(pred[..., 4] == float(i+1))]
            pred_list.append(inst)

        for i, pred_inst in enumerate(pred_list):
            iou_list = list()
            for j, gt_inst in enumerate(gt_list):
                iou = get_iou(pred_inst,gt_inst)
                iou_list.append(iou)
            iou_max = max(iou_list)
            if iou_max > iou_thr:
                tp += 1
            else:
                fp += 1
        n_gt = n_gt + len(gt_list)

    fn = n_gt - tp

    f = open(path_run + '/evaluation.txt', 'w')
    f.write('\n')
    f.write('Confusion matrix \n\n')
    f.write(cnf_matrix_s + '\n\n')
    f.write('Normalized confusion matrix \n\n')
    f.write(cnf_norm_s + '\n\n')
    f.write('Global accuracy: ' + str(acc_global) + '\n\n')

    for i in range(n_classes):
        str_acc = list(labels.keys())[list(labels.values()).index(i)] + ' accuracy: ' + str(acc_classes[i])
        str_prec = list(labels.keys())[list(labels.values()).index(i)] + ' precision: ' + str(prec_calsses[i])
        str_rec = list(labels.keys())[list(labels.values()).index(i)] + ' recall: ' + str(rec_classes[i])
        f.write(str_acc + '\n')
        f.write(str_prec + '\n')
        f.write(str_rec + '\n\n')
    f.close()


if __name__ == "__main__":
    main()
