import os
import sys
import argparse
from natsort import natsorted
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import indoor3d_util

'''
Call:

python txt_to_npy.py --path_in ../data/txt/ --path_out ../data/npy/ --path_cls meta/class_names.txt
'''

parser = argparse.ArgumentParser()
parser.add_argument('--path_in', help='path to the txt data folder.')
parser.add_argument('--path_out', help='path to save ply folder.')
parser.add_argument('--path_cls', help='path to classes txt.')

parsed_args = parser.parse_args(sys.argv[1:])

path_in = parsed_args.path_in
path_out = parsed_args.path_out
path_cls = parsed_args.path_cls

if not os.path.exists(path_out):
    os.mkdir(path_out)

for folder in natsorted(os.listdir(path_in)):

    path_annotation = os.path.join(path_in, folder, "annotations")
    print(path_annotation)

    elements = path_annotation.split('/')
    out_filename = os.path.join(path_out, elements[-2]+'.npy')
    indoor3d_util.collect_point_label(path_annotation, out_filename, path_cls, 'numpy')
