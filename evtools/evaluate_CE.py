#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import pandas as pd


def main(df_gt_data, df_est, est_timerange = [], quantile = 50):
    """
    Calculate Circular-Error 50 Percentile

    df_gt_data  : pandas.DataFrame [timestamp, x, y, theta, floor]
            ground truth
    df_est      : pandas.DataFrame [timestamp, x, y, floor, ...]
            estimated position
    """
    # timerange
    if len(est_timerange) < 1: est_timerange = (df_gt_data.index[0], df_gt_data.index[-1])
    ts_start = est_timerange[0]; ts_end = est_timerange[1]
    df_est = df_est[ts_start:ts_end]


    df_gt_data = df_gt_data.dropna(subset=("x", "y")); df_est = df_est.dropna(subset=("x", "y"))
    df_eval = pd.merge_asof(df_gt_data, df_est,
                            left_index=True, right_index=True, tolerance=0.5,
                            suffixes=["_gt", "_est"])
    df_eval = df_eval.dropna(subset=("x_gt", "y_gt", "x_est", "y_est"))

    df_eval["floor_correct"] = (df_eval["floor_est"] == df_eval["floor_gt"])
    df_eval_FC = df_eval[df_eval['floor_correct']]
    
    # calc error distance
    err_dst_FC = np.sqrt((df_eval_FC['x_gt'] - df_eval_FC['x_est'])**2 + (df_eval_FC['y_gt'] - df_eval_FC['y_est'])**2)

    return np.percentile(err_dst_FC, quantile)

