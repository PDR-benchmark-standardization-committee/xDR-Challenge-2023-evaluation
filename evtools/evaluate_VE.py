#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd


def main(df_est, valid_vel = 1.5, est_timerange = [], df_gt = None, PE_flg = True, window_time = 0.5):
    """
    Calc Velocity Error

    df_est      : pandas.DataFrame [timestamp, x, y, floor, ...]
            estimated position
    est_timerange : [timestamp(float), timestamp(float)], default=[gt.start, gt.end]
            estimated trajctory timerange to use evaluation
    df_gt  : pandas.DataFrame [timestamp, x, y, theta, floor]
            ground truth
    PE_flg : boolean
            True : calc_velocity_error_Median, False : calc_velocity_error_Avg
    diff_rate : int
            frame interval to calculate velocity from x-y
    """
    if df_gt is not None:
        if PE_flg: VE = calc_velocity_error_Median(df_est, df_gt, window_time, est_timerange)
        else: VE = calc_velocity_error_Avg(df_est, df_gt, est_timerange, window_time)
        
    else:
        VE = main_negative_xy(df_est, valid_vel, est_timerange)
    
    return VE


def calc_velocity_error_Median(df_est, df_gt, window_time=0.5, est_timerange=[]):
    """
    Calculate error of (gt_vel - est_vel) and
    return median of error
    """
    df_est = df_est.dropna(subset=('x', 'y'))

    # set timerange
    if len(est_timerange) > 0:
        if type(df_est.index) == str: df_est = df_est[str(est_timerange[0]):str(est_timerange[1])]
        else: df_est = df_est[(est_timerange[0]):(est_timerange[1])]
    
    # merge with timestamp
    df_eval = pd.merge_asof(df_gt, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    # calc velocity
    est_velocity = calc_velocity_from_xy2(df_eval, 'x_est', 'y_est', window_time)
    gt_velocity = calc_velocity_from_xy2(df_eval, 'x_gt', 'y_gt', window_time)
    
    return round(np.percentile(abs((gt_velocity - est_velocity)), 50), 4)


def calc_velocity_from_xy2(df_est, x_column='x', y_column='y', window_time = 0.5):
    """
    Calc velocity from sum of move distance from -0.5sec to +0.5sec
    """
    if 'ts' not in df_est.columns: df_est['ts'] = df_est.index
    
    dx = df_est[x_column].diff(); dx.iloc[0] = 0
    dy = df_est[y_column].diff(); dy.iloc[0] = 0
    ddist = np.sqrt(dx.values**2 + dy.values**2)

    dt = df_est['ts'].diff(); dt.iloc[0] = 1 # avoiding 0-division error
    df_vel = pd.DataFrame(data={'vel':ddist/dt, 'ts':df_est['ts'].values}, index=df_est['ts'].values)

    def calc_velocity_window(row):
        df_target = df_vel[row['ts'] - window_time:row['ts'] + window_time]
        return np.average(df_target.values)
    
    velocity_list = df_vel.apply(calc_velocity_window, axis=1)

    return velocity_list


####################
def main_negative_vxvy(df_est, valid_vel = 1.5, est_timerange = []):
    """
    check abnormal walking velocity (= valid_vel)
    and calculate rate of abnormal velocity.

    calculate velocity from vx-vy version
    """
    df_est = df_est.dropna(subset=("vx", "vy"))

    # calc velocity from vx-vy
    velocity = np.sqrt(df_est['vx']**2 + df_est['vy']**2)

    # set timerange
    if len(est_timerange) > 0:
        velocity = velocity[str(est_timerange[0]):str(est_timerange[1])]
        
    # calc rate
    err_vel = velocity[velocity > valid_vel]
    return round(np.sum(err_vel)/len(df_est), 4)


def main_negative_xy(df_est, valid_vel = 1.5, est_timerange = [], window_time=0.5):
    """
    check abnormal walking velocity (= valid_vel)
    and calculate rate of abnormal velocity.

    calculate velocity from x-y version
    """
    df_est = df_est.dropna(subset=('x', 'y'))

    # set timerange
    if len(est_timerange) > 0:
        if type(df_est.index) == str: df_est = df_est[str(est_timerange[0]):str(est_timerange[1])]
        else: df_est = df_est[(est_timerange[0]):(est_timerange[1])]

    # calc velocity from x-y
    velocity = calc_velocity_from_xy2(df_est, 'x', 'y', window_time)

    # calc rate
    err_vel = velocity[velocity < valid_vel]
    return round(len(err_vel)/len(velocity), 4)


def calc_velocity_error_Avg(df_est, df_gt, est_timerange=[], window_time=0.5):
    """
    速度正解値(lidarデータ)との比較によるpositive check版
    """
    df_est = df_est.dropna(subset=('x', 'y'))

    # set timerange
    if len(est_timerange) > 0:
        if type(df_est.index) == str: df_est = df_est[str(est_timerange[0]):str(est_timerange[1])]
        else: df_est = df_est[(est_timerange[0]):(est_timerange[1])]
    
    # merge with timestamp
    df_eval = pd.merge_asof(df_gt, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    est_velocity = calc_velocity_from_xy2(df_eval, 'x_est', 'y_est', window_time)
    gt_velocity = calc_velocity_from_xy2(df_eval, 'x_gt', 'y_gt', window_time)
    
    
    return round(np.average(abs((gt_velocity - est_velocity))), 4)


def calc_velocity_from_xy(df_est, x_column='x', y_column='y', diff_rate = 1):
    """
    Calc velocity from x-y
    """
    if 'ts' not in df_est.columns: df_est['ts'] = df_est.index

    dx = df_est[x_column].diff(diff_rate); dx = dx.iloc[diff_rate:]
    dy = df_est[y_column].diff(diff_rate); dy = dy.iloc[diff_rate:]
    dt = df_est['ts'].diff(diff_rate); dt = dt.iloc[diff_rate:]

    return np.sqrt(dx.values**2 + dy.values**2)/dt.values

