#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import argparse
import glob
import numpy as np
import pandas as pd
import csv

from evtools import evaluate_CE
from evtools import evaluate_CA
from evtools import evaluate_EAG
from evtools import evaluate_VE
from evtools import evaluate_OE

from evtools import EV_converter
from evtools import bitmap_tools


def main(traj_dir_path, gt_path=F'dataset{os.sep}gt{os.sep}', gis_path=F'dataset{os.sep}gis{os.sep}',
         est_weight=[0.25, 0.2, 0.25, 0.15, 0.15], floor = None,
         est_timerange = [], ALIP_timerange = [],
         draw = False, output_path = './output/'):
    """
    Evaluate all trajectories in traj_dir
    
    traj_dir_path : folder path
            folder path of estimated trajectories
    gt_path : folder path
            floder path of ground truth
    gis_path : folder path
            folder path of GIS images
    est_weight : [float, ...] : list length = 5
            weight to calculate total score
    est_timerange : [timestamp(float), timestamp(float)], default=[gt.start, gt.end]
            estimated trajctory timerange to use evaluation
    ALIP_timerange : [timestamp(float), timestamp(float)], default=[gt.start, gt.end]
            estimated pdr-trajctory timerange to use EAG evaluation
    draw : boolean
            output graph and image of specific evaluation result
    output_path : folder path
            folder path to output graph, image and csv
    """
    results_list = []
    traj_pdr_filename_list = glob.glob(traj_dir_path + '*_pdr_est.csv')

    for traj_pdr_filename in traj_pdr_filename_list:
        # identify filenmae 
        traj_name = traj_pdr_filename.split(F'{os.sep}')[-1].split('_pdr_est.csv')[0]
        traj_filename = traj_dir_path + traj_name + '_est.csv'
        gt_filename = gt_path + traj_name + '_gt.csv'

        # read csv
        df_est = pd.read_csv(traj_filename, header=None, names=['ts', 'x', 'y', 'floor'], index_col=0)
        df_est_pdr = pd.read_csv(traj_pdr_filename, header=None, names=['ts', 'x', 'y', 'floor'], index_col=0)
        df_gt = pd.read_csv(gt_filename, index_col=0); df_gt.index.name = 'ts'

        # calc score
        results = evaluate_traj(df_est, df_est_pdr, df_gt, gis_path, est_weight,
                                floor, draw, est_timerange, ALIP_timerange, output_path + F'{traj_name}{os.sep}')
        results.insert(0, traj_name)
        results_list.append(results)

    # output results to csv
    os.makedirs(output_path, exist_ok=True)
    output_csv(results_list, output_path)


def evaluate_traj(df_est, df_est_pdr, df_gt, gis_path, est_weight=[0.25, 0.2, 0.25, 0.15, 0.15],
         floor = None, draw = False,
         est_timerange = [], ALIP_timerange = [],
         output_path = './output/'):
    """
    Estimate trajectory on CE, CA, EAG, VE and OE
    and Calculate normalized Score
    """
    
    if draw: os.makedirs(output_path, exist_ok=True)
    
    # bitmap load
    if floor is None: floor = np.unique(df_est['floor'])[0]
    bitmap = bitmap_tools.load_bitmap_to_ndarray(gis_path + F"{floor}_0.01_0.01.bmp")
    # calc mapsize
    x = len(bitmap[0])/100; y = len(bitmap)/100
    mapsize = (y, x)
    
    # timerange
    if len(est_timerange) < 1: est_timerange = (df_gt.index[0], df_gt.index[-1])
    if len(ALIP_timerange) < 1: ALIP_timerange = (df_gt.index[0], df_gt.index[-1])

    # positive check
    CE = evaluate_CE.main(df_gt, df_est, est_timerange)
    CA = evaluate_CA.main(df_gt, df_est, est_timerange, draw_flg=draw, output_path=output_path)
    EAG = evaluate_EAG.main(df_gt, df_est_pdr, ALIP_timerange, draw_flg=draw, output_path=output_path)
    VE = evaluate_VE.main(df_est, est_timerange = est_timerange, df_gt = df_gt)

    # negative check
    OE = evaluate_OE.main(df_est, bitmap, mapsize, draw_flg=draw, output_path=output_path)

    # convert to normalized Score
    I_CE, I_CA, I_EAG, I_VE, I_OE = EV_converter.main(CE, CA, EAG, VE, OE)

    # calc Result
    score = I_CE * est_weight[0] + I_CA * est_weight[1] + I_EAG * est_weight[2] + \
            I_VE * est_weight[3] + I_OE * est_weight[4]
    
#     print('===================================')
#     print(F'CE  : {I_CA}')
#     print(F'CA  : {I_CA}')
#     print(F'EAG : {I_EAG}')
#     print(F'VE  : {I_VE}')
#     print(F'OE  : {I_OE}')
#     print('-----------------------------------')
#     print(F'Score : {score}')
#     print(F'==================================={os.linesep}')

    return [I_CE, I_CA, I_EAG, I_VE, I_OE, score]


def output_csv(results_list, output_path = './output/'):
    """
    output Results to csv
    """
    with open(output_path + 'result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'I_ce', 'I_ca', 'I_eag', 'I_ve', 'I_obstacle', 'Score'])
        for results in results_list:
            writer.writerow(results)
        
        results_narray = np.array(results_list, dtype=object)
        writer.writerow(['Avg Results', np.mean(results_narray[:,1]), np.mean(results_narray[:,2]), np.mean(results_narray[:,3]),
                         np.mean(results_narray[:,4]), np.mean(results_narray[:,5]), np.mean(results_narray[:,6])])


def main_cl(args):
    """
    fetch comand line arguments
    """
    args.traj_dir = normalize_path_end(args.traj_dir)
    args.gt_path = normalize_path_end(args.gt_path)
    args.gis_path = normalize_path_end(args.gis_path)

    main(args.traj_dir, args.gt_path, args.gis_path, args.est_weight, args.floor,
         args.est_timerange, args.ALIP_timerange, args.draw, args.output_path)

def normalize_path_end(txt):
    if txt[-1] == os.sep: return txt

    if txt[-1] == '"':
        txt[-1] = os.sep
    else:
        txt += os.sep
    return txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required params
    parser.add_argument('-traj_dir', '-t', type=str, help='directory of trajectory path.')
    parser.add_argument('-gt_path', type=str, help='gt folder path.', default=F'dataset{os.sep}gt{os.sep}')
    parser.add_argument('-gis_path', type=str, help='gis folder path.', default=F'dataset{os.sep}gis{os.sep}')

    # optional params
    parser.add_argument('--est_weight', nargs="*", type=float, default=[0.25, 0.2, 0.25, 0.15, 0.15])
    parser.add_argument('--floor', type=str, default=None)
    parser.add_argument('--est_timerange', nargs="*", type=float, default=[])
    parser.add_argument('--ALIP_timerange', nargs="*", type=float, default=[])
    parser.add_argument('--draw', action='store_true', help='output CA, EAG and OE graph-image')
    # parser.add_argument('--output_csv', action='store_true', help='output result to csv file')
    parser.add_argument('--output_path', type=str, default='./output/', help='output folder path')

    # parser.add_argument('--EAG_WCS_flg', action='store_true')
    # parser.add_argument('--EAG_hist2d_flg', action='store_true')
    # parser.add_argument('--EAG_mode', type=str, default='T')
    # parser.add_argument('--VE_Valid_vel', type=float, default=1.5)

    # parser.add_argument('--extension', type=str, default='.csv')

    args = parser.parse_args()
    main_cl(args)
