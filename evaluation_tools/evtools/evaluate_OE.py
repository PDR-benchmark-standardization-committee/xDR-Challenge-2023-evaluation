#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import pandas as pd
import numpy as np
import shapely
# from tqdm import tqdm
# tqdm.pandas()

import matplotlib.pyplot as plt


def main(df_est, obstacle, map_size, draw_flg=False, output_path='./output/'):
    """
    Calculate Obstacle Error

    df_est      : pandas.DataFrame [timestamp, x, y, floor, ...]
            estimated position
    obstacle : bitmap
            0 : movable, 1 : obstacle
    map_size : (x(float), y(float))
            map x-meter and y-meter
    draw_flg : boolean
            output image of obstacle check
    output_path : folder path
            foder path to output image of obstacle check
    """
    obstacle = np.logical_not(np.transpose(obstacle))
    x_block_m = len(obstacle[0]) / map_size[0]
    y_block_m = len(obstacle) / map_size[1]

    # print(x_block_m, y_block_m)

    # Convert bitmap coordinate to mapsize coordinate
    df_est.dropna(subset=['x', 'y'], inplace=True)

    df_est['x_block_num'] = np.abs((df_est['x'] - 0) * x_block_m).map(lambda x: int(x) -1 if int(x)!=0 else 0)
    df_est['y_block_num'] = np.abs((df_est['y'] - 0) * y_block_m).map(lambda x: int(x) -1 if int(x)!=0 else 0)

    # print(df_est.iloc[0])
    df_est['unixtime'] = df_est.index
    df_est = df_est[['unixtime', 'x', 'y', 'x_block_num', 'y_block_num']]

    x_block_num_dif = df_est['x_block_num'].diff()
    y_block_num_dif = df_est['y_block_num'].diff()
    x_block_num_dif[0] = 0; y_block_num_dif[0] = 0
    x_block_num_dif.dropna(inplace=True)
    y_block_num_dif.dropna(inplace=True)


    df_est['x_block_num_dif'] = (x_block_num_dif.astype(int).values)
    df_est['y_block_num_dif'] = (y_block_num_dif.astype(int).values)

    
    # Auxiliary function to calculate E_obstacle
    # check_pattern, is_inside_map, is_obstacle, is_obstacle_around, is_obstacle_exist, 
    # ObstacleCordinate_count, CheckCordinate_count
    def check_pattern(row):
        '''
        Fucntion to calculate obstacle error
        Appoint pattern to trajection point
        Parameters
        ----------
        row : pd.Series
            trajection file row data
        Return
        ------
        pattern : str
            'A' or 'B' or 'C' or 'D'
        '''

        x_dif = row['x_block_num_dif']
        y_dif = row['y_block_num_dif']
        
        if x_dif == 0:
            if y_dif  > 0:
                return 'A'
            else:
                return 'B'
    
        else:
            if x_dif > 0:
                return 'C'
            elif x_dif < 0:
                return 'D'

    def is_inside_map(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather input cordinate is inside bitmap data or not
        Parameters
        ----------
        x, y : int
            Cordinates
        
        Returns
        -------
        boolean : bool
            if cordinate is inside bitmap : True, else :  False
        '''
        if  0 <= x < obstacle.shape[1]  and 0 <= y < obstacle.shape[0]:
            return True
        else:
            return False
        
    def is_obstacle(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather obstacle exsits on input cordinates in bitmap data or not
        
        Parameters
        ----------
        x, y : int
            Cordinates
        
        Returns
        -------
        boolean : bool
            if obstacle exists on input cordinates : True, else :  False
        '''
        if obstacle[y][x] == 1:
            return True
        else:
            return False

    def is_obstacle_around(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather all area around input cordinates are filled with obstacle or not
        
        Parameters
        ----------
        x, y : int
            Cordinates
        
        Returns
        -------
        boolean : bool
            if no empty point exist : True, else :  False
        '''

        for x_i in range(-3, 4):
            for y_i in range(-3, 4):
                if is_inside_map(x + x_i, y + y_i):
                    if not is_obstacle(x + x_i, y+y_i):
                        return False          
        return True

    def is_obstacle_exist(x, y):
        '''
        Fucntion to calculate obstacle error
        Check wheather obstacle exist on input cordinate including around area
        Parameters
        ----------
        x, y : int
            Cordinates
        
        Returns
        -------
        boolean : bool
            if obstacle exist on input cordinates: True, else :  False
        '''
        if is_inside_map(x, y):
            if is_obstacle(x, y):
                if is_obstacle_around(x, y):
                    return True
        return False
    
    def ObstacleCordinate_count(row):
        '''
        Fucntion to calculate obstacle error
        Count total cordinates where obstacle exist in trajection data
        Parameters
        ----------
        row : pd.Series
            trajection file row data
        
        Returns
        -------
        obstacle_count : int
            number of total cordinates where obstacle exist
        '''

        y_block_num = row['y_block_num']
        y_block_num_t1 = y_block_num + row['y_block_num_dif']
        
        x_block_num = row['x_block_num']
        x_block_num_t1 = x_block_num + row['x_block_num_dif']
        
        obstacle_count = 0
        
        if row['pattern'] == 'A':
            for y in range(y_block_num, y_block_num_t1):
                if is_obstacle_exist(x_block_num, y):
                    obstacle_count += 1

        elif row['pattern'] == 'B':
            for y in range(y_block_num, y_block_num_t1, -1):
                if is_obstacle_exist(x_block_num, y):
                    obstacle_count += 1
                
        elif row['pattern'] == 'C':
            a = int((y_block_num - y_block_num_t1) / (x_block_num - x_block_num_t1))
            b = y_block_num - (a * x_block_num)
            for x in range(x_block_num, x_block_num_t1):
                y = int(a * x + b)
                if is_obstacle_exist(x, y):
                    obstacle_count += 1
                                    
        elif row['pattern'] == 'D':
            a = int((y_block_num - y_block_num_t1) / (x_block_num - x_block_num_t1))
            b = y_block_num - (a * x_block_num)
            for x in range(x_block_num, x_block_num_t1, -1):
                y = int(a * x + b)
                if is_obstacle_exist(x, y):
                    obstacle_count += 1
            
        return obstacle_count

    def CheckCordinate_count(row):
        '''
        Fucntion to calculate obstacle error
        Count total codinates checked wheather obstacle exist or not
        Parameters
        ----------
        row : pd.Series
            trajection file row data
        
        Returns
        -------
        check_cordinate_count : int
            number of total cordinates checked wheather obstacle exist or not
        '''

        pattern = row['pattern']
        if pattern  == 'A' or pattern  == 'B':
            return abs(row['y_block_num_dif'])
        else:
            return abs(row['x_block_num_dif'])
            
    df_est['pattern'] = df_est.apply(check_pattern, axis=1)
    
    # obstacle_cordinate_count =  df_est.progress_apply(ObstacleCordinate_count, axis=1)
    obstacle_cordinate_count =  df_est.apply(ObstacleCordinate_count, axis=1)
    check_cordinate_count = df_est.apply(CheckCordinate_count, axis=1)

    obstacle_check = pd.DataFrame({'check_cordinate_count': list(check_cordinate_count),
                                  'obstacle_cordinate_count': list(obstacle_cordinate_count)})
    

    if draw_flg: check_error(df_est, obstacle, obstacle_cordinate_count, output_path)

    # return np.sum(list(obstacle_cordinate_count))/np.sum(list(check_cordinate_count))
    return (np.sum(list(check_cordinate_count)) - np.sum(list(obstacle_cordinate_count)))/np.sum(list(check_cordinate_count))


def calc_mapsize(geom):
    x_min, y_min, x_max, y_max = shapely.bounds(geom)
    return ((x_max - x_min) *1, (y_max - y_min) *1)


def check_error(df_est, obstacle, obstacle_cordinate_count, output_path='./output/'):
    mask_1 = (0 < obstacle_cordinate_count)
    mask_0 = (obstacle_cordinate_count == 0)

    plt.rcParams['image.cmap'] = 'viridis'

    fig, ax = plt.subplots()
    
    ax.pcolor(obstacle)
    ax.scatter(df_est['x_block_num'][mask_0].values, df_est['y_block_num'][mask_0].values, s=1, color='black', label='movable')
    ax.scatter(df_est['x_block_num'][mask_1].values, df_est['y_block_num'][mask_1].values, s=1, color='red', label='in obstacle')
    x_ticks = ax.get_xticks(); y_ticks = ax.get_yticks()
    ax.set_xticklabels(x_ticks/100); ax.set_yticklabels(y_ticks/100)
    ax.set_xlabel('(m)'); ax.set_ylabel('(m)')
    ax.legend()

    fig.savefig(output_path + 'OE.png')
    plt.close()


