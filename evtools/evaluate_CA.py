#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as kde
import seaborn as sns
from scipy.spatial.transform import Rotation

import warnings
warnings.simplefilter('ignore')


def main(df_gt_data, df_est, est_timerange = [], RCS_flg = True, hist2d_flg = False, draw_flg = False, output_path = './output/'):
    if RCS_flg:
        # df_est['yaw'] = np.zeros(len(df_est)) # ACS化
        x_error, y_error = make_error_xy_local(df_est, df_gt_data, est_timerange)
    else:
        x_error, y_error = make_error_xy(df_est, df_gt_data, est_timerange)
    

    # 2dhist
    if hist2d_flg:
        if draw_flg: CA = calc_2D_histgram_mod(x_error, y_error) # 描画付き
        else: CA = calc_2dhist_mod(x_error, y_error)
        
    # kernel
    else:
        if draw_flg: CA = calc_kernel_density_mod(x_error, y_error, output_path) # 描画付き
        else: CA = calc_kernel_mod(x_error, y_error)
        

    return CA


def calc_2D_histgram_mod(x_error_list, y_error_list):
            
    fig = plt.figure()
    
    plt.rcParams['font.size'] = 12
    xmax = max(np.abs(x_error_list))
    ymax = max(np.abs(y_error_list))

    xbin = math.floor(xmax * 2/0.5)
    ybin = math.floor(ymax * 2/0.5)

    counts, xedges, yedges, _ = plt.hist2d(x_error_list,y_error_list, bins=(xbin, ybin))
    x_delta = xedges[1] - xedges[0]
    y_delta = yedges[1] - yedges[0]
    
    idx = np.unravel_index(np.argmax(counts), counts.shape)
    
    x_mod = xedges[idx[0]] + x_delta/2
    y_mod = yedges[idx[1]] + y_delta/2
    
    plt.plot(x_mod, y_mod, marker='^', color='forestgreen', 
            markerfacecolor='white', markeredgewidth=2, markersize=12)
    plt.xlabel('X error')
    plt.ylabel('Y error')
    plt.close()

    fig.savefig('CA_2dhist.png')

    CA = math.hypot(x_mod, y_mod)
    
    return CA


def calc_2dhist_mod(x_error_list, y_error_list):
    """
    位置誤差のx成分・y成分毎に分布を生成しその平均値と原点(=GT)との距離を評価値として返す。
    """
    xmax = max(np.abs(x_error_list))
    ymax = max(np.abs(y_error_list))

    xbin = math.floor(xmax * 2/0.5)
    ybin = math.floor(ymax * 2/0.5)

    counts, xedges, yedges, _ = plt.hist2d(x_error_list,y_error_list, bins=(xbin, ybin))
    x_delta = xedges[1] - xedges[0]
    y_delta = yedges[1] - yedges[0]
    
    # print(np.argmax(counts))
    idx = np.unravel_index(np.argmax(counts), counts.shape)
    
    x_mod = xedges[idx[0]] + x_delta/2
    y_mod = yedges[idx[1]] + y_delta/2

    x_mod, y_mod

    CA = math.hypot(x_mod, y_mod)

    return CA


def make_error_xy(df_est, df_gt_data, est_timerange = []):
    """
    ACS version
    """
    # timerange
    if len(est_timerange) < 1: est_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ts_start = est_timerange[0]; ts_end = est_timerange[1]
    df_est = df_est[ts_start:ts_end]

    if "yaw" not in df_gt_data.keys():
        df_gt_data["yaw"] = [Rotation.from_quat(q).as_euler("XYZ")[0] for q in df_gt_data[["q0", "q1", "q2", "q3"]].values]

    if "floor_ble_mode" in df_est.columns:
        df_est["floor"] = df_est["floor_ble_mode"]
    df_gt_data = df_gt_data.dropna(subset=("x", "y")); df_est = df_est.dropna(subset=("x", "y"))
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]

    x_error = df_eval_FC['x_gt'] - df_eval_FC['x_est']
    y_error = df_eval_FC['y_gt'] - df_eval_FC['y_est']

    return (x_error, y_error)


def make_error_xy_local(df_est, df_gt_data, est_timerange = []):
    """
    RCS version
    """
    # timerange
    if len(est_timerange) < 1: est_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ts_start = est_timerange[0]; ts_end = est_timerange[1]
    df_est = df_est[ts_start:ts_end]

    if "yaw" not in df_gt_data.keys():
        df_gt_data["yaw"] = [Rotation.from_quat(q).as_euler("XYZ")[0] for q in df_gt_data[["q0", "q1", "q2", "q3"]].values]
    if "yaw" not in df_est.keys():
        df_est["yaw"] = calc_yaw_from_xy(df_est)

    if "floor_ble_mode" in df_est.columns:
        df_est["floor"] = df_est["floor_ble_mode"]
    df_gt_data = df_gt_data.dropna(subset=("x", "y")); df_est = df_est.dropna(subset=("x", "y"))
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]

    y_error = df_eval_FC.apply(calc_y_error, axis=1)
    df_eval_FC['y_error'] = y_error
    x_error = df_eval_FC.apply(calc_x_error, axis=1)

    return (x_error, y_error)


def calc_y_error(row):
    """
    orthographic vector
    y_error = {(vector_a ・ vector_b) / |ventor_a|**2}|vector_a|

    vector_a : unit vector to yaw direction
    vector_b : vector from estimated position to ground truth position
    y_error = (vector_a ・ vector_b)
    """
    vector_b = [row['x_gt'] - row['x_est'], row['y_gt'] - row['y_est']]
    vector_a = [np.sin(row['yaw_est']), np.cos(row['yaw_est'])]

    return (np.dot(vector_a, vector_b))


def calc_x_error(row):
    """
    vector_b : vector from estimated position to ground truth position
    vector_c : unit vector normal to yaw direction
    x_error = (vector_c ・ vector_b)
    """
    vector_b = [row['x_gt'] - row['x_est'], row['y_gt'] - row['y_est']]
    vector_c = [np.cos(row['yaw_est']), -np.sin(row['yaw_est'])]

    return (np.dot(vector_c, vector_b))


def calc_yaw_from_xy(df_est):
    """
    +y = 0; -y = 2pi
    +y → +x → -y = (0 <= Θ < 2pi)
    +y → -x → -y = (-2pi < Θ <= 0)
    """
    return np.arctan(df_est['x']/df_est['y'])


def calc_kernel_density_mod(x, y, output_path, bw_method=None):
    fig = plt.figure()
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 12
    nbins=300
    k = kde.gaussian_kde([x,y], bw_method=bw_method)
    xi, yi = np.mgrid[min(x)-2:max(x)+2:nbins*1j, min(y)-2:max(y)+2:nbins*1j]
    try:
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    except:
        print('Unable to calculate inverse matrix, return mean value')
        return np.mean(x), np.mean(y), fig
    row_idx = np.argmax(zi) // len(xi)
    col_idx = np.argmax(zi) % len(yi)
    x_mod = xi[:, 0][row_idx].round(2)
    y_mod = yi[0][col_idx].round(2)
    
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap='jet')
    plt.plot(x_mod, y_mod, marker='^', color='forestgreen', 
            markerfacecolor='white', markeredgewidth=2, markersize=12)
    plt.title('x: {:.2f} y: {:.2f}'.format(x_mod, y_mod))
    plt.xlabel('X error')
    plt.ylabel('Y error')

    fig.savefig(output_path + 'CA_kernel.png')
    plt.close()

    CA = math.hypot(x_mod, y_mod)
    
    return CA


def calc_kernel_mod(x, y, bw_method=None):
    fig = plt.figure()
    nbins=300
    k = kde.gaussian_kde([x,y], bw_method=bw_method)
    xi, yi = np.mgrid[min(x)-2:max(x)+2:nbins*1j, min(y)-2:max(y)+2:nbins*1j]
    try:
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    except:
        print('Unable to calculate inverse matrix, return mean value')
        return np.mean(x), np.mean(y), fig
    row_idx = np.argmax(zi) // len(xi)
    col_idx = np.argmax(zi) % len(yi)
    x_mod = xi[:, 0][row_idx].round(2)
    y_mod = yi[0][col_idx].round(2)

    CA = math.hypot(x_mod, y_mod)
    
    return CA


