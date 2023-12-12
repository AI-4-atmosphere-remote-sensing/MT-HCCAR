import os
import math
import logging
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logit
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path


"""
Read dataset file
"""
def load_data(filename):
     ext = splitext(filename)[1]
     if ext =='.nc':
        ds = xr.open_dataset(filename)
        data = {var_name: ds[var_name].values for var_name in ds.variables}
        # for var_name, var_data in data.items():
        #     logging.info(f"{var_name},{var_data.shape}")
        return data
     else:
        raise ValueError('Please have data file that we provided in the data folder')
     
"""
MinMax scaler
"""
def minmaxscaler(reflectances):
    if len(reflectances.shape) == 1:
        return (reflectances - reflectances.min()) / (reflectances.max() - reflectances.min())
    else:
        min_vals = reflectances.min(axis=1, keepdims=True)
        max_vals = reflectances.max(axis=1, keepdims=True)
        return (reflectances - min_vals) / (max_vals - min_vals)

"""
Define labels for albedo types: [land, snow, desert, ocean_water]
Output 1-hot arraies of each albedo type
"""
def albedo_category(albedo_type):
    land = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,18])
    snow = np.array([15,19,20])
    desert = np.array([16, 16])
    ocean_water = np.array([17, 17])
    albedo = np.array([land, snow, desert, ocean_water], dtype=object)

    albedo_type_sc = np.array([[1 if val in category else 0 for val in albedo_type] for category in albedo])
    
    return albedo_type_sc

"""
scene_type: 1 cloud free; 2 single cloud layer
cloud_type: 1 liquid phase; 2 ice crystal
1-hot array: [0]: cloud-free, [1]: liquid, [2]: ice
"""
def transfer_1hot_array(scene_type, cloud_type):
    scene_type_1hot = np.zeros((len(scene_type), 3), dtype=int)
    # cloud-free
    scene_type_1hot[scene_type == 1, 0] = 1
    # cloud-liquid
    scene_type_1hot[(scene_type == 2) & (cloud_type == 1), 1] = 1
    # cloud-ice
    scene_type_1hot[(scene_type == 2) & (cloud_type == 2), 2] = 1
    return scene_type_1hot



def Preprocessing(data_dir: str):
    data_dir = Path(data_dir)
    all_variable_arrays = load_data(data_dir)
    X = np.concatenate((
        all_variable_arrays['oci_reflectances'],
        all_variable_arrays['h2o'][None, :],
        all_variable_arrays['o3'][None, :],
        all_variable_arrays['spress'][None, :],
        albedo_category(all_variable_arrays['albedo_type']),
        np.cos(all_variable_arrays['solar_zenith_angle'][None, :]),
        np.cos(all_variable_arrays['viewing_zenith_angle'][None, :]),
        np.cos(all_variable_arrays['relative_azimuth_angle'][None, :])
    ), axis=0).astype(np.float32)
    X = minmaxscaler(X)
    X = np.transpose(X)

    Y_cls = transfer_1hot_array(all_variable_arrays['scene_type'], all_variable_arrays['cloud_type'])
    Preprocessing.min_value = np.floor(np.min(all_variable_arrays['log10_cloud_optical_thickness'][~np.isnan(all_variable_arrays['log10_cloud_optical_thickness'])]))
    Preprocessing.max_value = np.max(all_variable_arrays['log10_cloud_optical_thickness'][~np.isnan(all_variable_arrays['log10_cloud_optical_thickness'])])
    Y_reg = np.nan_to_num(all_variable_arrays['log10_cloud_optical_thickness'], nan=Preprocessing.min_value)
    Y = np.concatenate((Y_cls, np.transpose(Y_reg[None, :])), axis=1).astype(np.float32)
    Y = minmaxscaler(np.transpose(Y))
    Y = np.transpose(Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    torch.save(X_train_t, f'{Path(data_dir).parent}/X_train.pt')
    torch.save(y_train_t, f'{Path(data_dir).parent}/y_train.pt')
    torch.save(X_val_t, f'{Path(data_dir).parent}/X_val.pt')
    torch.save(y_val_t, f'{Path(data_dir).parent}/y_val.pt')
    torch.save(X_test_t, f'{Path(data_dir).parent}/X_test.pt')
    torch.save(y_test_t, f'{Path(data_dir).parent}/y_test.pt')

    return X_train_t, y_train_t, X_val_t, y_val_t