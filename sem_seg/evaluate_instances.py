
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
 - python3 evaluate_instances.py --path_run /home/uib/Desktop/test_evaluate_instances/ --path_cls /home/uib/Desktop/test_evaluate_instances/classes.txt --iou_thr 0.5 --test_name test --ref 0
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
    pc_array = np.array([[x, y, z, r, g, b, c, i] for x,y,z,r,g,b,c,i in pc])
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
    parser.add_argument('--iou_thr', default=0.5, help='min iou.')
    parser.add_argument('--test_name', help='name of the test')
    parser.add_argument('--ref', default=0, help='name of the test')

    parsed_args = parser.parse_args(sys.argv[1:])

    path_run = parsed_args.path_run
    path_cls = parsed_args.path_cls  # get class txt path
    iou_thr = float(parsed_args.iou_thr)
    test_name = parsed_args.test_name
    ref = int(parsed_args.ref)

    path_infer = os.path.join(path_run, 'dump_' + test_name)

    classes, labels, label2color = get_info_classes(path_cls)

    files = natsorted(os.listdir(path_infer))
    cases = [s for s in files if s.endswith(".obj")]
    names = natsorted(set([re.split("[.\_]+", string)[0] for string in cases]))

    tp = np.zeros((len(classes),), dtype=int)
    fp = np.zeros((len(classes),), dtype=int)
    n_gt = np.zeros((len(classes),), dtype=int)
    n_pred = np.zeros((len(classes),), dtype=int)
    iou_max_sum = np.zeros((len(classes),), dtype=float)

    for name in names:

        print("evaluating case: " + name)
        path_gt = os.path.join(path_infer, name + "_gt_inst.ply")
        path_pred = os.path.join(path_infer, name + "_pred_inst.ply")
        if ref==1:
            path_pred = os.path.join(path_infer, name + "_pred_inst_ref.ply")

        gt = read_ply(path_gt)
        pred = read_ply(path_pred)

        if (gt.shape[0]>2) and (pred.shape[0]>2):  # IN CASE GT OR PRED ARE "EMPTY" - LOK AT GET_INSTANCES OUTPUT WHEN NO INSTANCES ORE FOUND (THEY SAVE "NULL" A TWO ROW NUMPY)

            gt_list = list()
            instances_gt = set(gt[..., 7])
            instances_pred = set(pred[..., 7])

            for i in instances_gt:
                inst = gt[np.where(gt[..., 7] == float(i))]
                gt_list.append(inst) 
                n_gt[int(inst[0, 6])] += 1 

            pred_list = list()
            for j in instances_pred:
                inst = pred[np.where(pred[..., 7] == float(j))]
                pred_list.append(inst)
                n_pred[int(inst[0, 6])] += 1

            for i, pred_inst in enumerate(pred_list):
                iou_list = list()
                for j, gt_inst in enumerate(gt_list):
                    if pred_inst[0, 6] == gt_inst[0, 6]:
                        iou = get_iou(pred_inst,gt_inst)
                    else:
                        iou = 0
                    iou_list.append(iou)
                iou_max = max(iou_list)
                iou_max_sum[int(pred_inst[0, 6])]+= iou_max
                if  iou_max >= iou_thr:
                    tp[int(pred_inst[0, 6])] += 1 
                else:
                    fp[int(pred_inst[0, 6])] += 1 

    fn = n_gt - tp 
    iou_max_mean = iou_max_sum / n_pred

    # hacer cambios para sacar una fila de excel por cada clase con su nombre
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f1 = (2*recall*precision)/(recall+precision)

    filepath = os.path.join(path_run, "evaluation_instance_" + test_name + ".xlsx")
    if ref==1:
        filepath = os.path.join(path_run, "evaluation_instance_ref_" + test_name + ".xlsx")
    
    header = ['Recall', 'Precision', 'F1', 'mean_IoU']
    csv = ({header[0]: recall, header[1]: precision, header[2]: f1, header[3]: iou_max_mean})
    df = pd.DataFrame.from_records(csv, index=classes)
    df.to_excel(filepath)





if __name__ == "__main__":
    main()

