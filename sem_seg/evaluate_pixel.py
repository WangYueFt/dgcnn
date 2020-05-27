import os
import re
import numpy as np
from os import listdir
import argparse
import sys
from natsort import natsorted
from sklearn.preprocessing import normalize
import pandas as pd

'''
script to evaluate a model

execution example:
 - python3 evaluate_pixel.py --path_run "/home/miguel/Desktop/pipes/dgcnn/sem_seg/RUNS/valve_test/" --path_cls "/home/miguel/Desktop/pipes/data/valve_test/classes.txt"
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_runs', help='path to the folder.')
    parser.add_argument('--path_cls', help='path to the folder.')
    parsed_args = parser.parse_args(sys.argv[1:])

    path_runs = parsed_args.path_runs
    path_cls = parsed_args.path_cls  # get class txt path

    run_folders = listdir(path_runs)

    for run in run_folders:

        print("evaluating run: " + run)

        path_run = os.path.join(path_runs,run)

        path_infer = os.path.join(path_run, 'dump')

        classes, labels, label2color = get_info_classes(path_cls)

        files = natsorted(os.listdir(path_infer))
        cases = [s for s in files if s.endswith(".txt")]
        names = natsorted(set([re.split("[.\_]+",string)[0] for string in cases]))
        names = names[:-2]

        n_classes = len(classes)
        cnf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        for name in names:

            print("---evaluating case: " + name)
            path_gt = os.path.join(path_infer, name + "_gt.txt")
            path_pred = os.path.join(path_infer, name + "_pred.txt")

            gt = np.loadtxt(path_gt, dtype=int)
            pred = np.loadtxt(path_pred)[..., 7].astype(int)

            for i, label in enumerate(gt):
                cnf_matrix[label,pred[i]] += 1

        cnf_norm_h = normalize(cnf_matrix, norm='l1', axis=1)*100
        cnf_norm_v = normalize(cnf_matrix, norm='l1', axis=0)*100


        acc_global = cnf_matrix.diagonal().sum()/cnf_matrix.sum()
        prec_calsses = list()
        rec_classes = list()
        acc_classes = list()

        for i in range(n_classes):
            if np.sum(cnf_matrix, axis=0)[i] != 0:
                prec_class = cnf_matrix[i,i] / np.sum(cnf_matrix, axis=0)[i]
            else:
                prec_class = 'nan'

            if np.sum(cnf_matrix, axis=1)[i] != 0:
                rec_class = cnf_matrix[i, i] / np.sum(cnf_matrix, axis=1)[i]
            else:
                rec_class = 'nan'

            if np.sum(cnf_matrix, axis=0)[i] != 0 and np.sum(cnf_matrix, axis=1)[i] != 0:
                acc_class = (cnf_matrix[i, i] + np.sum(cnf_matrix) - np.sum(cnf_matrix, axis=0)[i] - np.sum(cnf_matrix, axis=1)[i] + cnf_matrix[i, i]) / np.sum(cnf_matrix)
            else:
                acc_class = 'nan'

            prec_calsses.append(prec_class)
            rec_classes.append(rec_class)
            acc_classes.append(acc_class)

        cnf_matrix_s = np.array_str(cnf_matrix)
        cnf_norm_h_s = np.array_str(cnf_norm_h)
        cnf_norm_v_s = np.array_str(cnf_norm_v)

        f = open(path_run + '/evaluation.txt', 'w')
        f.write('\n')
        f.write('Confusion matrix \n\n')
        f.write(cnf_matrix_s + '\n\n')
        f.write('Normalized confusion matrix (horizontal)\n\n')
        f.write(cnf_norm_h_s + '\n\n')
        f.write('Normalized confusion matrix (vertical)\n\n')
        f.write(cnf_norm_v_s + '\n\n')
        f.write('Global accuracy: ' + str(acc_global) + '\n\n')

        for i in range(n_classes):
            str_acc = list(labels.keys())[list(labels.values()).index(i)] + ' accuracy: ' + str(acc_classes[i])
            str_prec = list(labels.keys())[list(labels.values()).index(i)] + ' precision: ' + str(prec_calsses[i])
            str_rec = list(labels.keys())[list(labels.values()).index(i)] + ' recall: ' + str(rec_classes[i])
            f.write(str_acc + '\n')
            f.write(str_prec + '\n')
            f.write(str_rec + '\n\n')
        f.close()

        filepath = path_run + '/evaluation_pixel.xlsx'
        df = pd.DataFrame.from_records(cnf_matrix, index=classes)
        df.to_excel(filepath, header = classes, index_label = 'gt\pred')

if __name__ == "__main__":
    main()
