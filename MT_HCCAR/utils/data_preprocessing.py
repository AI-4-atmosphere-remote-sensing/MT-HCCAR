import os
import math
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
from time import perf_counter

dir_data = './data/'

file_path ='data/rt_nn_cloud_training_data_20230518.nc'
ds = xr.open_dataset(file_path)

oci_reflectances = ds["oci_reflectances"].values 
modis_reflectances = ds["modis_reflectances"].values 
viirs_reflectances = ds["viirs_reflectances"].values 
nbands_oci = ds["nbands_oci"].values
nbands_modis = ds["nbands_modis"].values
nbands_viirs = ds["nbands_viirs"].values
angles = ds[["solar_zenith_angle", "viewing_zenith_angle", "relative_azimuth_angle"]].to_array().values
h2o = ds["h2o"].values
o3 = ds["o3"].values
scene_type = ds["scene_type"].values
albedo_type = ds["albedo_type"].values
spress = ds["spress"].values
cot = ds["log10_cloud_optical_thickness"].values
cloud_type = ds["cloud_type"].values

def minmaxscaler(reflectances):
        if len(reflectances.shape) == 1:
                reflectances_sc = (reflectances-reflectances.min())/(reflectances.max() - reflectances.min())
        else:
                reflectances_sc = []
                for i in range(len(reflectances)):
                        reflectances_sc_tem = (reflectances[i]-reflectances[i].min())/(reflectances[i].max()-reflectances[i].min())
                        reflectances_sc.append(reflectances_sc_tem)
        return np.array(reflectances_sc)

# Define labels for albedo type
land = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,18]
snow = [15,19,20]
desert = [16, 16]
ocean_water = [17, 17]
albedo = [land, snow, desert, ocean_water]

def albedo_category(albedo_type):
    albedo_type_sc = []
    for k in range(len(albedo)):
        i = 0
        albedo_1hot = []
        for i in range(len(albedo_type)):
            if albedo_type[i] in albedo[k]:
                albedo_1hot.append(1)
            else:
                albedo_1hot.append(0)
        albedo_type_sc.append(albedo_1hot)
    return np.array(albedo_type_sc)


"""
scene_type: 1 cloud free; 2 single cloud layer
cloud_type: 1 liquid phase; 2 ice crystal
1-hot array: [0]: cloud-free, [1]: liquid, [2]: ice
"""
scene_type_1hot = []
for i in range(len(scene_type)):
    # cloud-free
    if scene_type[i] == 1:
        scene_type_1hot.append([1,0,0])
    elif scene_type[i] == 2:
        # cloud-liquid
        if cloud_type[i] == 1:
            scene_type_1hot.append([0,1,0])
        # cloud-ice
        elif cloud_type[i] == 2:
            scene_type_1hot.append([0,0,1])

"""
Reflectances, h2o, o3, spress: minmax normalize to [0,1]
Angles: take cosine and normalize to [0,1]
albedo_type: break down into 4 categories

y1 - scen_type: change values from [1, 2] to [0,1]
y2 - cot: set 0 as -2
"""
oci_reflectances_sc = minmaxscaler(oci_reflectances)
h2o_sc = minmaxscaler(h2o)
o3_sc = minmaxscaler(o3)
spress_sc = minmaxscaler(spress)
angles_sc = minmaxscaler(np.cos(angles))
albedo_type_sc = np.float32(albedo_category(albedo_type))
scene_type_sc = minmaxscaler(np.array(scene_type_1hot))
scene_type_sc = np.float32(scene_type_sc)

cot_sc = np.nan_to_num(cot, nan=-2) 
max_value = cot_sc.max()
min_value = cot_sc.min()
cot_sc = minmaxscaler(cot_sc)

X = np.concatenate((oci_reflectances_sc, h2o_sc[None, :], o3_sc[None, :], spress_sc[None, :], albedo_type_sc, angles_sc), axis=0) # n x m, where n = 22
X = np.float32(X)
Y_cls = scene_type_sc
Y_reg = cot_sc
Y = np.concatenate((Y_cls, np.transpose(Y_reg[None, :])), axis=1)
X = np.transpose(X)
print(X.shape, Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

X_train_t = torch.from_numpy(X_train.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.float32))

X_val_t = torch.from_numpy(X_val.astype(np.float32))
y_val_t = torch.from_numpy(y_val.astype(np.float32))

X_test_t = torch.from_numpy(X_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.float32))

torch.save(X_train_t, f'{dir_data}/X_train_t.pt')
torch.save(y_train_t, f'{dir_data}/y_train_t.pt')
torch.save(X_val_t, f'{dir_data}/X_val_t.pt')
torch.save(y_val_t, f'{dir_data}/y_val_t.pt')
torch.save(X_test_t, f'{dir_data}/X_test_t.pt')
torch.save(y_test_t, f'{dir_data}/y_test_t.pt')