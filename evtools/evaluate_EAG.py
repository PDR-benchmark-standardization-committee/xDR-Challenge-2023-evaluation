#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def main(df_gt_data, df_est, ALIP_timerange = [], mode='T', draw_flg = False, output_path='./output/'):
    """
    Calculate Error Accumulation Gradient

    df_gt_data  : pandas.DataFrame [timestamp, x, y, theta, floor]
            ground truth
    df_est      : pandas.DataFrame [timestamp, x, y, floor, ...]
            estimated position
    ALIP_timerange : [float, float]
    mode : str ['T', 'D', 'A']
            EAG based on Time, Distance or Angle
    draw_flg : boolean
            output EAG graph flag
    output_path : str (path)
            output graph folder path
    """
    try:
        if mode == 'T':
            EAG = calc_T_EAG(df_gt_data, df_est, ALIP_timerange, draw_flg, output_path)
        elif mode == 'D':
            EAG = calc_D_EAG(df_gt_data, df_est, ALIP_timerange, draw_flg, output_path)
        elif mode == 'A':
            EAG = calc_A_EAG(df_gt_data, df_est, ALIP_timerange, draw_flg, output_path)
    except Exception as e:
        print(e)
        EAG = None
    
    return EAG

###
def calc_T_EAG(df_gt_data, df_est, ALIP_timerange = [], draw_flg = False, output_path = './output/'):
    """
    Calculate EAG based on Time
    """
    # set timerange
    if len(ALIP_timerange) < 1: ALIP_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ALIP_start = ALIP_timerange[0]
    ALIP_end = ALIP_timerange[-1]

    df_gt_data = df_gt_data[ALIP_start:ALIP_end]
    df_est = df_est[ALIP_start:ALIP_end]

    # convert to relative time near ALIP-start time and ALIP-end time
    df_gt_data['delta_ts'] = df_gt_data.index
    df_gt_data['delta_ts'][ALIP_start:ALIP_start + (ALIP_end - ALIP_start)/2] -= ALIP_start
    df_gt_data['delta_ts'][ALIP_start + (ALIP_end - ALIP_start)/2:ALIP_end] -= ALIP_end
    df_gt_data['delta_ts'][ALIP_start + (ALIP_end - ALIP_start)/2:ALIP_end] *= -1

    df_gt_data = df_gt_data.dropna(subset=("x", "y")); df_est = df_est.dropna(subset=("x", "y"))

    df_gt_data['ts_gt'] = df_gt_data.index
    df_est['ts_est'] = df_est.index

    # merge with timestamp
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]

    df_EAG = np.hypot(df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est']) / df_eval_FC['delta_ts']
    
    if not draw_flg: return np.percentile(df_EAG.values, 50)

    ### graph
    mask = (df_eval_FC['delta_ts'] > 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_xlabel('elapsed time [s]')
    ax1.set_ylabel('error distance form gt [m]')
    ax1.scatter(df_eval_FC['delta_ts'], np.hypot(df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est']), s=1)

    ax2 = fig.add_subplot(2,1,2)
    ax2.scatter(df_eval_FC['delta_ts'][mask], df_EAG[mask], s=1)
    ax2.set_xlabel('elapsed time [s]')
    ax2.set_ylabel('T-EAG [m/s]')

    p50 = np.full(len(df_eval_FC['delta_ts'][mask]), np.percentile(df_EAG.values, 50))

    plt.tight_layout()
    fig.savefig(output_path + 'T_EAG.png')
    plt.close()

    return np.percentile(df_EAG.values, 50)

###
def calc_D_EAG(df_gt_data, df_est, ALIP_timerange = [], draw_flg = False, output_path = './output/'):
    """
    Calculate EAG based on Distance
    """
    if len(ALIP_timerange) < 1: ALIP_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ALIP_start = ALIP_timerange[0]
    ALIP_end = ALIP_timerange[-1]

    df_gt_data = df_gt_data[ALIP_start:ALIP_end]
    df_est = df_est[ALIP_start:ALIP_end]

    s_point = (df_gt_data['x'].iloc[0], df_gt_data['y'].iloc[0])
    e_point = (df_gt_data['x'].iloc[-1], df_gt_data['y'].iloc[-1])

    if "floor_ble_mode" in df_est.columns:
        df_est["floor"] = df_est["floor_ble_mode"]
    df_gt_data = df_gt_data.dropna(subset=("x", "y")); df_est = df_est.dropna(subset=("x", "y"))

    df_gt_data['ts_gt'] = df_gt_data.index
    df_est['ts_est'] = df_est.index

    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]

    df_eval_FC['x_diff'] = df_eval_FC['x_est'].diff()
    df_eval_FC['y_diff'] = df_eval_FC['y_est'].diff()
    df_eval_FC.drop(index=df_eval_FC.index[0], inplace=True)
    df_eval_FC['dist_diff'] = np.hypot(df_eval_FC['x_diff'], df_eval_FC['y_diff'])

    def calc_Sdistance(row):
        idx = row['ts_gt']
        dist_s = np.sum(df_eval_FC[ALIP_start:idx]['dist_diff'].values)
        dist_e = np.sum(df_eval_FC[idx:ALIP_end]['dist_diff'].values)
        
        return np.min([dist_s, dist_e])
    
    dist_from_ALIPpoint = df_eval_FC.apply(calc_Sdistance, axis=1)
    error_from_gt = np.hypot(df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est'])

    df_EAG = error_from_gt/dist_from_ALIPpoint

    if not draw_flg: return np.percentile(df_EAG, 50)

    ### graph
    mask = (dist_from_ALIPpoint > 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_xlabel('cumlative distance [m]')
    ax1.set_ylabel('error distance form gt [m]')
    ax1.scatter(dist_from_ALIPpoint, error_from_gt, s=1)

    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xlabel('elapsed time [s]')
    ax2.set_ylabel('D-EAG [m/m]')
    ax2.scatter(dist_from_ALIPpoint[mask], df_EAG[mask], s=1)
    plt.tight_layout()
    fig.savefig(output_path + 'D_EAG.png')
    plt.close()

    return np.percentile(df_EAG, 50)

###
def calc_A_EAG(df_gt_data, df_est, ALIP_timerange = [], draw_flg = False, output_path = './output/'):
    """
    Calculate EAG based on Angle
    """
    if len(ALIP_timerange) < 1: ALIP_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ALIP_start = ALIP_timerange[0]
    ALIP_end = ALIP_timerange[-1]

    df_gt_data = df_gt_data[ALIP_start:ALIP_end]
    df_est = df_est[ALIP_start:ALIP_end]

    if "yaw" not in df_gt_data.keys():
        df_gt_data["yaw"] = [Rotation.from_quat(q).as_euler("XYZ")[0] for q in df_gt_data[["q0", "q1", "q2", "q3"]].values]
    
    if "floor_ble_mode" in df_est.columns:
        df_est["floor"] = df_est["floor_ble_mode"]
    df_gt_data = df_gt_data.dropna(subset=("x", "y")); df_est = df_est.dropna(subset=("x", "y"))

    df_gt_data['ts_gt'] = df_gt_data.index
    df_est['ts_est'] = df_est.index

    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]

    df_eval_FC['yaw_gt'] = np.abs(df_eval_FC['yaw_gt'].diff())
    df_eval_FC['yaw_est'] = np.abs(df_eval_FC['yaw_est'].diff())
    df_eval_FC.drop(index=df_eval_FC.index[0], inplace=True)

    def calc_Srad(row):
        idx = row['ts_gt']
        Srad_s = np.sum(df_eval_FC[ALIP_start:idx]['yaw_est'].values)
        Srad_e = np.sum(df_eval_FC[idx:ALIP_end]['yaw_est'].values)

        return np.min([Srad_s, Srad_e])

    Srad_from_ALIP = df_eval_FC.apply(calc_Srad, axis=1)
    error_from_gt = np.hypot(df_eval_FC['x_gt'] - df_eval_FC['x_est'], df_eval_FC['y_gt'] - df_eval_FC['y_est'])

    df_EAG = error_from_gt/Srad_from_ALIP

    if not draw_flg: return np.percentile(df_EAG, 50)

    ### graph
    mask = (Srad_from_ALIP > 1)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_xlabel('cumlative angle [rad]')
    ax1.set_ylabel('error distance form gt [m]')
    ax1.scatter(Srad_from_ALIP, error_from_gt, s=1)
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xlabel('cumlative angle [rad]')
    ax2.set_ylabel('A-EAG [m/rad]')
    ax2.scatter(Srad_from_ALIP[mask], df_EAG[mask], s=1)
    plt.tight_layout()
    fig.savefig(output_path + 'A_EAG.png')
    plt.close()

    return np.percentile(df_EAG, 50)

