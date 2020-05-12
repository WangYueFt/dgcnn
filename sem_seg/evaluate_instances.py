
import os
import re
import numpy as np
import argparse
import sys
from natsort import natsorted
from plyfile import PlyData, PlyElement
import pandas as pd

'''
script to evaluate a model

execution example:
 - python3 evaluate.py --path_run "/home/miguel/Desktop/pipes/dgcnn/sem_seg/RUNS/valve_test/" --path_cls "/home/miguel/Desktop/pipes/data/valve_test/classes.txt"
'''

def get_iou(inst1,inst2):

    inst1 = inst1[:, 0:3].tolist()
    inst2 = inst2[:, 0:3].tolist()
    intersection = 0

    for i in inst1:
        if i in inst2:
            intersection += 1
    union = len(inst1) + len(inst2) - intersection
    iou = intersection/union
    return iou


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, c, i] for x,y,z,r,g,b,c,i in pc])
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', help='path to the run folder.')
    parser.add_argument('--path_cls', help='path to the class file.')
    parser.add_argument('--iou_thr', help='min iou.')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    iou_thr = float(parsed_args.iou_thr)

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
        n_gt = n_gt + len(gt_list)

        pred_list = list()
        for j in range(len(set(pred[..., 4]))):
            inst = pred[np.where(pred[..., 4] == float(j+1))]
            pred_list.append(inst)

        for i, pred_inst in enumerate(pred_list):
            iou_list = list()
            for j, gt_inst in enumerate(gt_list):
                if pred_inst[0, 3] == gt_inst[0, 3]:
                    iou = get_iou(pred_inst,gt_inst)
                else:
                    iou = 0
                iou_list.append(iou)
            iou_max = max(iou_list)
            if iou_max > iou_thr:
                tp += 1
            else:
                fp += 1
    fn = n_gt - tp

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = (2*recall*precision)/(recall+precision)

    filepath = os.path.join(path_run, "evaluation_instance.xlsx")

    header = ['Recall.', 'Precision', 'F1']
    spine_csv = ({header[0]: recall, header[1]: precision, header[2]: f1})
    df = pd.DataFrame.from_records(spine_csv, index=[''])
    df.to_excel(filepath)





if __name__ == "__main__":
    main()

