import os


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar, seq_missing, block_missing
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae, calc_mse, calc_mre
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
from sklearn.impute import KNNImputer
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib.widgets import Slider
import ipywidgets as widgets
from ipywidgets import interact
from scipy.fft import fft
from scipy.ndimage import gaussian_filter


# Load JSON dictionaries for training and testing
instances_dict_path_train = './instances_dict_train.json'
with open(instances_dict_path_train, 'r') as f:
    instances_dict_train = json.load(f)
    
instances_dict_path_test = './instances_dict_test.json'
with open(instances_dict_path_test, 'r') as f:
    instances_dict_test = json.load(f)

# Load numpy arrays (processed and original)
smoothpur_1_4_train = np.load('Data_in/Train/SmoothPur_1_4_Ar.npy')
smoothpur_5_8_train = np.load('Data_in/Train/SmoothPur_5_8_Ar.npy')
smoothpur_9_10_train = np.load('Data_in/Train/SmoothPur_9_10_Ar.npy')
smoothpur_11_12_train = np.load('Data_in/Train/SmoothPur_11_12_Ar.npy')

smoothpur_1_4_test = np.load('Data_in/Test/SmoothPur_1_4_Ar.npy')
smoothpur_5_8_test = np.load('Data_in/Test/SmoothPur_5_8_Ar.npy')
smoothpur_9_10_test = np.load('Data_in/Test/SmoothPur_9_10_Ar.npy')
smoothpur_11_12_test = np.load('Data_in/Test/SmoothPur_11_12_Ar.npy')

smoothpur_1_4_train_ori = np.load('Data_in/Train/SmoothPur_1_4.npy')
smoothpur_5_8_train_ori = np.load('Data_in/Train/SmoothPur_5_8.npy')
smoothpur_9_10_train_ori = np.load('Data_in/Train/SmoothPur_9_10.npy')
smoothpur_11_12_train_ori = np.load('Data_in/Train/SmoothPur_11_12.npy')

smoothpur_1_4_test_ori = np.load('Data_in/Test/SmoothPur_1_4.npy')
smoothpur_5_8_test_ori = np.load('Data_in/Test/SmoothPur_5_8.npy')
smoothpur_9_10_test_ori = np.load('Data_in/Test/SmoothPur_9_10.npy')
smoothpur_11_12_test_ori = np.load('Data_in/Test/SmoothPur_11_12.npy')

# Print information
print('TRAIN')
print(instances_dict_train)
print('length of instances_dict : ', len(instances_dict_train))
print('-' * 100)
print('length of smoothpur_1_4 : ', len(smoothpur_1_4_train))
print('length of smoothpur_5_8 : ', len(smoothpur_5_8_train))
print('length of smoothpur_9_10 : ', len(smoothpur_9_10_train))
print('length of smoothpur_11_12 : ', len(smoothpur_11_12_train))
print('-' * 100)
print('type of smoothpur_1_4 : ', type(smoothpur_1_4_train))
print('type of smoothpur_5_8 : ', type(smoothpur_5_8_train))
print('type of smoothpur_9_10 : ', type(smoothpur_9_10_train))
print('type of smoothpur_11_12 : ', type(smoothpur_11_12_train))
print('-' * 100)
print('shape of smoothpur_1_4 : ', smoothpur_1_4_train.shape)
print('shape of smoothpur_5_8 : ', smoothpur_5_8_train.shape)
print('shape of smoothpur_9_10 : ', smoothpur_9_10_train.shape)
print('shape of smoothpur_11_12 : ', smoothpur_11_12_train.shape)

print('TEST')
print(instances_dict_test)
print('length of instances_dict : ', len(instances_dict_test))
print('-' * 100)
print('length of smoothpur_1_4 : ', len(smoothpur_1_4_test))
print('length of smoothpur_5_8 : ', len(smoothpur_5_8_test))
print('length of smoothpur_9_10 : ', len(smoothpur_9_10_test))
print('length of smoothpur_11_12 : ', len(smoothpur_11_12_test))
print('-' * 100)
print('type of smoothpur_1_4 : ', type(smoothpur_1_4_test))
print('type of smoothpur_5_8 : ', type(smoothpur_5_8_test))
print('type of smoothpur_9_10 : ', type(smoothpur_9_10_test))
print('type of smoothpur_11_12 : ', type(smoothpur_11_12_test))
print('-' * 100)
print('shape of smoothpur_1_4 : ', smoothpur_1_4_test.shape)
print('shape of smoothpur_5_8 : ', smoothpur_5_8_test.shape)
print('shape of smoothpur_9_10 : ', smoothpur_9_10_test.shape)
print('shape of smoothpur_11_12 : ', smoothpur_11_12_test.shape)

# Function definitions
def downsample(signals, new_len):
    downsampled_signals = []
    for signal in signals:
        signal = signal.reshape(-1)  # Flatten the signal
        downsample_factor = len(signal) // new_len
        indices = np.arange(0, len(signal), downsample_factor)
        downsampled_signal = signal[indices[:new_len]]  # Ensure the length matches new_len
        downsampled_signals.append(downsampled_signal)
    return np.array(downsampled_signals)

def flatten_smoothpur_1_8(original_arr):
    arr_shape = original_arr.shape
    flattened_list = []
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            flattened_list.append(original_arr[i][j][0])
            flattened_list.append(original_arr[i][j][1])
    flattened_arr = np.array(flattened_list)
    return flattened_arr

def reform_smoothpur_1_8(flattened_arr, original_arr):
    arr_shape = original_arr.shape
    reshaped_arr = flattened_arr.reshape((154, 4, 2, 15000))
    target_values = np.empty((154, 4, 1, 15000))
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            target = original_arr[i][j][2]
            target_values[i][j][0] = target
    reformed_arr = np.concatenate((reshaped_arr, target_values), axis=2)
    return reformed_arr

def flatten_Target_1_8(original_arr):
    arr_shape = original_arr.shape
    flattened_list = []
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            flattened_list.append(original_arr[i][j][2])
            flattened_list.append(original_arr[i][j][2])
    flattened_arr = np.array(flattened_list)
    return flattened_arr

def flatten_Target_9_12(original_arr):
    arr_shape = original_arr.shape
    flattened_x_list = []
    flattened_y_list = []
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            flattened_x_list.append(original_arr[i][j][2][0])
            flattened_x_list.append(original_arr[i][j][2][0])
            flattened_y_list.append(original_arr[i][j][2][1])
            flattened_y_list.append(original_arr[i][j][2][1])
    flattened_x_arr = np.array(flattened_x_list)
    flattened_y_arr = np.array(flattened_y_list)
    return flattened_x_arr, flattened_y_arr

def reform_smoothpur_1_8_500(flattened_arr):
    reshaped_arr = flattened_arr.reshape((154, 4, 2, 500))
    return reshaped_arr

def flatten_smoothpur_9_12(original_arr):
    arr_shape = original_arr.shape
    flattened_x_list = []
    flattened_y_list = []
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            flattened_x_list.append(original_arr[i][j][0][0])
            flattened_x_list.append(original_arr[i][j][0][1])
            flattened_y_list.append(original_arr[i][j][1][0])
            flattened_y_list.append(original_arr[i][j][1][1])
    flattened_x_arr = np.array(flattened_x_list)
    flattened_y_arr = np.array(flattened_y_list)
    return flattened_x_arr, flattened_y_arr

def reform_smoothpur_9_12(flattened_x_arr, flattened_y_arr, original_arr):
    arr_shape = original_arr.shape
    reshaped_x_arr = flattened_x_arr.reshape((154, 2, 1, 2, 15000))
    reshaped_y_arr = flattened_y_arr.reshape((154, 2, 1, 2, 15000))
    reformed_signal_arr = np.concatenate((reshaped_x_arr, reshaped_y_arr), axis=2)
    target_x_values = np.empty((154, 2, 1, 1, 15000))
    target_y_values = np.empty((154, 2, 1, 1, 15000))
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            targetx = original_arr[i][j][2][0]
            targety = original_arr[i][j][2][1]
            target_x_values[i][j][0][0] = targetx
            target_y_values[i][j][0][0] = targety
    reformed_target_arr = np.concatenate((target_x_values, target_y_values), axis=3)
    total_reformed_arr = np.concatenate((reformed_signal_arr, reformed_target_arr), axis=2)
    return total_reformed_arr

def reform_smoothpur_9_12_500(flattened_x_arr, flattened_y_arr):
    reshaped_x_arr = flattened_x_arr.reshape((154, 2, 1, 2, 500))
    reshaped_y_arr = flattened_y_arr.reshape((154, 2, 1, 2, 500))
    reformed_signal_arr = np.concatenate((reshaped_x_arr, reshaped_y_arr), axis=2)
    return reformed_signal_arr

def calculate_metrics(predicted_signals, original_signals, indicating_mask):
    masked_predicted = predicted_signals[indicating_mask]
    masked_original = original_signals[indicating_mask]
    mae = np.mean(np.abs(masked_predicted - masked_original))
    mse = np.mean((masked_predicted - masked_original) ** 2)
    mre = np.sum(np.abs(masked_predicted - masked_original)) / np.sum(np.abs(masked_original) + 1e-12)
    return mae, mse, mre

def compute_all_metrics(imputed_signal, original_signal, indicating_mask):
    num_signals, signal_length = original_signal.shape
    high_freq = 30
    low_freq = 0.1
    sampling_rate = 1000
    MAE_list = []
    MRE_list = []
    RMSE_list = []
    Sim_list = []
    FSD_list = []
    RMSE_F_list = []
    RMSE_F_low_list = []
    RMSE_F_high_list = []
    freq_bins = np.fft.rfftfreq(signal_length, d=1/sampling_rate)
    low_freq_indices = np.where(freq_bins <= low_freq)[0]
    high_freq_indices = np.where(freq_bins >= high_freq)[0]
    for i in range(num_signals):
        orig_sig = np.ravel(original_signal[i, :])
        imp_sig = np.ravel(imputed_signal[i, :])
        mask = np.ravel(indicating_mask[i, :])
        if not np.any(mask):
            continue
        orig_values_at_imputed = orig_sig[mask]
        imp_values_at_imputed = imp_sig[mask]
        mae = np.mean(np.abs(orig_values_at_imputed - imp_values_at_imputed))
        MAE_list.append(mae)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_errors = np.abs((orig_values_at_imputed - imp_values_at_imputed) / orig_values_at_imputed)
            relative_errors = np.nan_to_num(relative_errors, nan=0.0, posinf=0.0, neginf=0.0)
        mre = np.mean(relative_errors)
        MRE_list.append(mre)
        mse = np.mean((orig_values_at_imputed - imp_values_at_imputed) ** 2)
        rmse = np.sqrt(mse)
        RMSE_list.append(rmse)
        orig_mean = np.mean(orig_values_at_imputed)
        imp_mean = np.mean(imp_values_at_imputed)
        numerator = np.sum((orig_values_at_imputed - orig_mean) * (imp_values_at_imputed - imp_mean))
        denominator = np.sqrt(np.sum((orig_values_at_imputed - orig_mean) ** 2) * np.sum((imp_values_at_imputed - imp_mean) ** 2))
        sim = numerator / denominator if denominator != 0 else 0
        Sim_list.append(sim)
        std_orig = np.std(orig_values_at_imputed)
        fsd = rmse / std_orig if std_orig != 0 else 0
        FSD_list.append(fsd)
        orig_sig_fft = fft(orig_sig)
        imp_sig_fft = fft(imp_sig)
        mse_f = np.mean(np.abs(orig_sig_fft - imp_sig_fft) ** 2)
        rmse_f = np.sqrt(mse_f)
        RMSE_F_list.append(rmse_f)
        orig_sig_fft_low = orig_sig_fft[low_freq_indices]
        imp_sig_fft_low = imp_sig_fft[low_freq_indices]
        mse_f_low = np.mean(np.abs(orig_sig_fft_low - imp_sig_fft_low) ** 2)
        rmse_f_low = np.sqrt(mse_f_low)
        RMSE_F_low_list.append(rmse_f_low)
        orig_sig_fft_high = orig_sig_fft[high_freq_indices]
        imp_sig_fft_high = imp_sig_fft[high_freq_indices]
        mse_f_high = np.mean(np.abs(orig_sig_fft_high - imp_sig_fft_high) ** 2)
        rmse_f_high = np.sqrt(mse_f_high)
        RMSE_F_high_list.append(rmse_f_high)
    metrics = {
        'MAE_mean': np.mean(MAE_list) if MAE_list else None,
        'MRE_mean': np.mean(MRE_list) if MRE_list else None,
        'RMSE_mean': np.mean(RMSE_list) if RMSE_list else None,
        'Sim_mean': np.mean(Sim_list) if Sim_list else None,
        'FSD_mean': np.mean(FSD_list) if FSD_list else None,
        'RMSE_F_mean': np.mean(RMSE_F_list) if RMSE_F_list else None,
        'RMSE_F_Low_mean': np.mean(RMSE_F_low_list) if RMSE_F_low_list else None,
        'RMSE_F_High_mean': np.mean(RMSE_F_high_list) if RMSE_F_high_list else None
    }
    return metrics

print("TRAIN")

flat_1_4_train = flatten_smoothpur_1_8(smoothpur_1_4_train)
flat_5_8_train = flatten_smoothpur_1_8(smoothpur_5_8_train)
flat_x_9_10_train, flat_y_9_10_train = flatten_smoothpur_9_12(smoothpur_9_10_train)
flat_x_11_12_train, flat_y_11_12_train = flatten_smoothpur_9_12(smoothpur_11_12_train)

mean_1_4_train = np.nanmean(flat_1_4_train)
std_1_4_train = np.nanstd(flat_1_4_train)
flat_1_4_normalized_train = (flat_1_4_train - mean_1_4_train) / std_1_4_train
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_train = np.nanmean(flat_5_8_train)
std_5_8_train = np.nanstd(flat_5_8_train)
flat_5_8_normalized_train = (flat_5_8_train - mean_5_8_train) / std_5_8_train
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_train = np.nanmean(flat_x_9_10_train)
std_x_9_10_train = np.nanstd(flat_x_9_10_train)
flat_x_9_10_normalized_train = (flat_x_9_10_train - mean_x_9_10_train) / std_x_9_10_train
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_train = np.nanmean(flat_y_9_10_train)
std_y_9_10_train = np.nanstd(flat_y_9_10_train)
flat_y_9_10_normalized_train = (flat_y_9_10_train - mean_y_9_10_train) / std_y_9_10_train
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_train = np.nanmean(flat_x_11_12_train)
std_x_11_12_train = np.nanstd(flat_x_11_12_train)
flat_x_11_12_normalized_train = (flat_x_11_12_train - mean_x_11_12_train) / std_x_11_12_train
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_train = np.nanmean(flat_y_11_12_train)
std_y_11_12_train = np.nanstd(flat_y_11_12_train)
flat_y_11_12_normalized_train = (flat_y_11_12_train - mean_y_11_12_train) / std_y_11_12_train
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_train.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_train.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_train.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_train.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_train.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_train.shape)

print("TEST")

flat_1_4_test = flatten_smoothpur_1_8(smoothpur_1_4_test)
flat_5_8_test = flatten_smoothpur_1_8(smoothpur_5_8_test)
flat_x_9_10_test, flat_y_9_10_test = flatten_smoothpur_9_12(smoothpur_9_10_test)
flat_x_11_12_test, flat_y_11_12_test = flatten_smoothpur_9_12(smoothpur_11_12_test)

mean_1_4_test = np.nanmean(flat_1_4_test)
std_1_4_test = np.nanstd(flat_1_4_test)
flat_1_4_normalized_test = (flat_1_4_test - mean_1_4_test) / std_1_4_test
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_test = np.nanmean(flat_5_8_test)
std_5_8_test = np.nanstd(flat_5_8_test)
flat_5_8_normalized_test = (flat_5_8_test - mean_5_8_test) / std_5_8_test
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_test = np.nanmean(flat_x_9_10_test)
std_x_9_10_test = np.nanstd(flat_x_9_10_test)
flat_x_9_10_normalized_test = (flat_x_9_10_test - mean_x_9_10_test) / std_x_9_10_test
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_test = np.nanmean(flat_y_9_10_test)
std_y_9_10_test = np.nanstd(flat_y_9_10_test)
flat_y_9_10_normalized_test = (flat_y_9_10_test - mean_y_9_10_test) / std_y_9_10_test
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_test = np.nanmean(flat_x_11_12_test)
std_x_11_12_test = np.nanstd(flat_x_11_12_test)
flat_x_11_12_normalized_test = (flat_x_11_12_test - mean_x_11_12_test) / std_x_11_12_test
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_test = np.nanmean(flat_y_11_12_test)
std_y_11_12_test = np.nanstd(flat_y_11_12_test)
flat_y_11_12_normalized_test = (flat_y_11_12_test - mean_y_11_12_test) / std_y_11_12_test
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_test.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_test.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_test.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_test.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_test.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_test.shape)


print("TRAIN")

flat_1_4_train_ori = flatten_smoothpur_1_8(smoothpur_1_4_train_ori)
flat_5_8_train_ori = flatten_smoothpur_1_8(smoothpur_5_8_train_ori)
flat_x_9_10_train_ori, flat_y_9_10_train_ori = flatten_smoothpur_9_12(smoothpur_9_10_train_ori)
flat_x_11_12_train_ori, flat_y_11_12_train_ori = flatten_smoothpur_9_12(smoothpur_11_12_train_ori)

mean_1_4_train_ori = np.nanmean(flat_1_4_train_ori)
std_1_4_train_ori = np.nanstd(flat_1_4_train_ori)
flat_1_4_normalized_train_ori = (flat_1_4_train_ori - mean_1_4_train_ori) / std_1_4_train_ori
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_train_ori = np.nanmean(flat_5_8_train_ori)
std_5_8_train_ori = np.nanstd(flat_5_8_train_ori)
flat_5_8_normalized_train_ori = (flat_5_8_train_ori - mean_5_8_train_ori) / std_5_8_train_ori
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_train_ori = np.nanmean(flat_x_9_10_train_ori)
std_x_9_10_train_ori = np.nanstd(flat_x_9_10_train_ori)
flat_x_9_10_normalized_train_ori = (flat_x_9_10_train_ori - mean_x_9_10_train_ori) / std_x_9_10_train_ori
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_train_ori = np.nanmean(flat_y_9_10_train_ori)
std_y_9_10_train_ori = np.nanstd(flat_y_9_10_train_ori)
flat_y_9_10_normalized_train_ori = (flat_y_9_10_train_ori - mean_y_9_10_train_ori) / std_y_9_10_train_ori
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_train_ori = np.nanmean(flat_x_11_12_train_ori)
std_x_11_12_train_ori = np.nanstd(flat_x_11_12_train_ori)
flat_x_11_12_normalized_train_ori = (flat_x_11_12_train_ori - mean_x_11_12_train_ori) / std_x_11_12_train_ori
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_train_ori = np.nanmean(flat_y_11_12_train_ori)
std_y_11_12_train_ori = np.nanstd(flat_y_11_12_train_ori)
flat_y_11_12_normalized_train_ori = (flat_y_11_12_train_ori - mean_y_11_12_train_ori) / std_y_11_12_train_ori
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_train_ori.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_train_ori.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_train_ori.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_train_ori.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_train_ori.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_train_ori.shape)

print("TEST")

flat_1_4_test_ori = flatten_smoothpur_1_8(smoothpur_1_4_test_ori)
flat_5_8_test_ori = flatten_smoothpur_1_8(smoothpur_5_8_test_ori)
flat_x_9_10_test_ori, flat_y_9_10_test_ori = flatten_smoothpur_9_12(smoothpur_9_10_test_ori)
flat_x_11_12_test_ori, flat_y_11_12_test_ori = flatten_smoothpur_9_12(smoothpur_11_12_test_ori)

mean_1_4_test_ori = np.nanmean(flat_1_4_test_ori)
std_1_4_test_ori = np.nanstd(flat_1_4_test_ori)
flat_1_4_normalized_test_ori = (flat_1_4_test_ori - mean_1_4_test_ori) / std_1_4_test_ori
# flat_1_4_normalized = min_max_normalize(flat_1_4)

mean_5_8_test_ori = np.nanmean(flat_5_8_test_ori)
std_5_8_test_ori = np.nanstd(flat_5_8_test_ori)
flat_5_8_normalized_test_ori = (flat_5_8_test_ori - mean_5_8_test_ori) / std_5_8_test_ori
# flat_5_8_normalized = min_max_normalize(flat_5_8)


mean_x_9_10_test_ori = np.nanmean(flat_x_9_10_test_ori)
std_x_9_10_test_ori = np.nanstd(flat_x_9_10_test_ori)
flat_x_9_10_normalized_test_ori = (flat_x_9_10_test_ori - mean_x_9_10_test_ori) / std_x_9_10_test_ori
# flat_x_9_10_normalized = min_max_normalize(flat_x_9_10)

mean_y_9_10_test_ori = np.nanmean(flat_y_9_10_test_ori)
std_y_9_10_test_ori = np.nanstd(flat_y_9_10_test_ori)
flat_y_9_10_normalized_test_ori = (flat_y_9_10_test_ori - mean_y_9_10_test_ori) / std_y_9_10_test_ori
# flat_y_9_10_normalized = min_max_normalize(flat_y_9_10)

mean_x_11_12_test_ori = np.nanmean(flat_x_11_12_test_ori)
std_x_11_12_test_ori = np.nanstd(flat_x_11_12_test_ori)
flat_x_11_12_normalized_test_ori = (flat_x_11_12_test_ori - mean_x_11_12_test_ori) / std_x_11_12_test_ori
# flat_x_11_12_normalized = min_max_normalize(flat_x_11_12)

mean_y_11_12_test_ori = np.nanmean(flat_y_11_12_test_ori)
std_y_11_12_test_ori = np.nanstd(flat_y_11_12_test_ori)
flat_y_11_12_normalized_test_ori = (flat_y_11_12_test_ori - mean_y_11_12_test_ori) / std_y_11_12_test_ori
# flat_y_11_12_normalized = min_max_normalize(flat_y_11_12)

print('shape of flattened smoothpur_1_4 :', flat_1_4_normalized_test_ori.shape)
print('shape of flattened smoothpur_5_8 :', flat_5_8_normalized_test_ori.shape)
print('shape of flattened smoothpur_x_9_10 :', flat_x_9_10_normalized_test_ori.shape)
print('shape of flattened smoothpur_y_9_10 :', flat_y_9_10_normalized_test_ori.shape)
print('shape of flattened smoothpur_x_11_12 :', flat_x_11_12_normalized_test_ori.shape)
print('shape of flattened smoothpur_y_11_12 :', flat_y_11_12_normalized_test_ori.shape)



flats_normalized_train = np.concatenate([
    flat_1_4_normalized_train,
    flat_5_8_normalized_train,
    flat_x_9_10_normalized_train,
    flat_y_9_10_normalized_train,
    flat_x_11_12_normalized_train,
    flat_y_11_12_normalized_train
], axis=0)

flats_normalized_test = np.concatenate([
    flat_1_4_normalized_test,
    flat_5_8_normalized_test,
    flat_x_9_10_normalized_test,
    flat_y_9_10_normalized_test,
    flat_x_11_12_normalized_test,
    flat_y_11_12_normalized_test
], axis=0)

flats_normalized_train_ori = np.concatenate([
    flat_1_4_normalized_train_ori,
    flat_5_8_normalized_train_ori,
    flat_x_9_10_normalized_train_ori,
    flat_y_9_10_normalized_train_ori,
    flat_x_11_12_normalized_train_ori,
    flat_y_11_12_normalized_train_ori
], axis=0)

flats_normalized_test_ori = np.concatenate([
    flat_1_4_normalized_test_ori,
    flat_5_8_normalized_test_ori,
    flat_x_9_10_normalized_test_ori,
    flat_y_9_10_normalized_test_ori,
    flat_x_11_12_normalized_test_ori,
    flat_y_11_12_normalized_test_ori
], axis=0)

print("TRAIN")
newL = 500
down_1_4_train = downsample(flat_1_4_normalized_train, newL)
down_5_8_train = downsample(flat_5_8_normalized_train, newL)
down_x_9_10_train = downsample(flat_x_9_10_normalized_train, newL)
down_y_9_10_train = downsample(flat_y_9_10_normalized_train, newL)
down_x_11_12_train = downsample(flat_x_11_12_normalized_train, newL)
down_y_11_12_train = downsample(flat_y_11_12_normalized_train, newL)
downs_train = downsample(flats_normalized_train, newL)

down_1_4_train_ori = downsample(flat_1_4_normalized_train_ori, newL)
down_5_8_train_ori = downsample(flat_5_8_normalized_train_ori, newL)
down_x_9_10_train_ori = downsample(flat_x_9_10_normalized_train_ori, newL)
down_y_9_10_train_ori = downsample(flat_y_9_10_normalized_train_ori, newL)
down_x_11_12_train_ori = downsample(flat_x_11_12_normalized_train_ori, newL)
down_y_11_12_train_ori = downsample(flat_y_11_12_normalized_train_ori, newL)
downs_train_ori = downsample(flats_normalized_train_ori, newL)

print('shape of downsampled smoothpur_1_4 :', down_1_4_train.shape)
print('shape of downsampled smoothpur_5_8 :', down_5_8_train.shape)
print('shape of downsampled smoothpur_x_9_10 :', down_x_9_10_train.shape)
print('shape of downsampled smoothpur_y_9_10 :', down_y_9_10_train.shape)
print('shape of downsampled smoothpur_x_11_12 :', down_x_11_12_train.shape)
print('shape of downsampled smoothpur_y_11_12 :', down_y_11_12_train.shape)
print('shape of downsampled All smoothpurs :', downs_train.shape)

print("TEST")
down_1_4_test = downsample(flat_1_4_normalized_test, newL)
down_5_8_test = downsample(flat_5_8_normalized_test, newL)
down_x_9_10_test = downsample(flat_x_9_10_normalized_test, newL)
down_y_9_10_test = downsample(flat_y_9_10_normalized_test, newL)
down_x_11_12_test = downsample(flat_x_11_12_normalized_test, newL)
down_y_11_12_test = downsample(flat_y_11_12_normalized_test, newL)
downs_test = downsample(flats_normalized_test, newL)

down_1_4_test_ori = downsample(flat_1_4_normalized_test_ori, newL)
down_5_8_test_ori = downsample(flat_5_8_normalized_test_ori, newL)
down_x_9_10_test_ori = downsample(flat_x_9_10_normalized_test_ori, newL)
down_y_9_10_test_ori = downsample(flat_y_9_10_normalized_test_ori, newL)
down_x_11_12_test_ori = downsample(flat_x_11_12_normalized_test_ori, newL)
down_y_11_12_test_ori = downsample(flat_y_11_12_normalized_test_ori, newL)
downs_test_ori = downsample(flats_normalized_test_ori, newL)

print('shape of downsampled smoothpur_1_4 :', down_1_4_test.shape)
print('shape of downsampled smoothpur_5_8 :', down_5_8_test.shape)
print('shape of downsampled smoothpur_x_9_10 :', down_x_9_10_test.shape)
print('shape of downsampled smoothpur_y_9_10 :', down_y_9_10_test.shape)
print('shape of downsampled smoothpur_x_11_12 :', down_x_11_12_test.shape)
print('shape of downsampled smoothpur_y_11_12 :', down_y_11_12_test.shape)
print('shape of downsampled All smoothpurs :', downs_test.shape)

print("TRAIN")
exp_1_4_train = np.expand_dims(down_1_4_train, axis=-1)
exp_5_8_train = np.expand_dims(down_5_8_train, axis=-1)
exp_x_9_10_train = np.expand_dims(down_x_9_10_train, axis=-1)
exp_y_9_10_train = np.expand_dims(down_y_9_10_train, axis=-1)
exp_x_11_12_train = np.expand_dims(down_x_11_12_train, axis=-1)
exp_y_11_12_train = np.expand_dims(down_y_11_12_train, axis=-1)
exps_train = np.expand_dims(downs_train, axis=-1)

print("TRAIN")
exp_1_4_train_ori = np.expand_dims(down_1_4_train_ori, axis=-1)
exp_5_8_train_ori = np.expand_dims(down_5_8_train_ori, axis=-1)
exp_x_9_10_train_ori = np.expand_dims(down_x_9_10_train_ori, axis=-1)
exp_y_9_10_train_ori = np.expand_dims(down_y_9_10_train_ori, axis=-1)
exp_x_11_12_train_ori = np.expand_dims(down_x_11_12_train_ori, axis=-1)
exp_y_11_12_train_ori = np.expand_dims(down_y_11_12_train_ori, axis=-1)
exps_train_ori = np.expand_dims(downs_train_ori, axis=-1)

print('shape of expanded smoothpur_1_4 :', exp_1_4_train.shape)
print('shape of expanded smoothpur_5_8 :', exp_5_8_train.shape)
print('shape of expanded smoothpur_x_9_10 :', exp_x_9_10_train.shape)
print('shape of expanded smoothpur_y_9_10 :', exp_y_9_10_train.shape)
print('shape of expanded smoothpur_x_11_12 :', exp_x_11_12_train.shape)
print('shape of expanded smoothpur_y_11_12 :', exp_y_11_12_train.shape)
print('shape of expanded All smoothpur :', exps_train.shape)

print("TEST")
exp_1_4_test = np.expand_dims(down_1_4_test, axis=-1)
exp_5_8_test = np.expand_dims(down_5_8_test, axis=-1)
exp_x_9_10_test = np.expand_dims(down_x_9_10_test, axis=-1)
exp_y_9_10_test = np.expand_dims(down_y_9_10_test, axis=-1)
exp_x_11_12_test = np.expand_dims(down_x_11_12_test, axis=-1)
exp_y_11_12_test = np.expand_dims(down_y_11_12_test, axis=-1)
exps_test = np.expand_dims(downs_test, axis=-1)

exp_1_4_test_ori = np.expand_dims(down_1_4_test_ori, axis=-1)
exp_5_8_test_ori = np.expand_dims(down_5_8_test_ori, axis=-1)
exp_x_9_10_test_ori = np.expand_dims(down_x_9_10_test_ori, axis=-1)
exp_y_9_10_test_ori = np.expand_dims(down_y_9_10_test_ori, axis=-1)
exp_x_11_12_test_ori = np.expand_dims(down_x_11_12_test_ori, axis=-1)
exp_y_11_12_test_ori = np.expand_dims(down_y_11_12_test_ori, axis=-1)
exps_test_ori = np.expand_dims(downs_test_ori, axis=-1)

print('shape of expanded smoothpur_1_4 :', exp_1_4_test.shape)
print('shape of expanded smoothpur_5_8 :', exp_5_8_test.shape)
print('shape of expanded smoothpur_x_9_10 :', exp_x_9_10_test.shape)
print('shape of expanded smoothpur_y_9_10 :', exp_y_9_10_test.shape)
print('shape of expanded smoothpur_x_11_12 :', exp_x_11_12_test.shape)
print('shape of expanded smoothpur_y_11_12 :', exp_y_11_12_test.shape)
print('shape of expanded All smoothpur :', exps_test.shape)

mask_rate = 0.0001

from pypots.optim import Adam
custom_optimizer = Adam(lr=0.0004)

X_ori = exps_train_ori
np.random.seed(42)
torch.manual_seed(42)
dataset = {"X": exps_train}

# saits = SAITS(n_steps=newL, n_features=1, n_layers=4, d_model=512, n_heads=8,
#               d_k=64, d_v=64, d_ffn=256, dropout=0.2, epochs=300,
#               optimizer=custom_optimizer, model_saving_strategy='better')

saits = SAITS(n_steps=newL, n_features=1, n_layers=2, d_model=256, n_heads=4,
               d_k=64, d_v=64, d_ffn=128, dropout=0.2, epochs=500, 
               optimizer=custom_optimizer, model_saving_strategy='better')

saits.fit(dataset)
imputation = saits.impute(dataset)

indicating_mask = np.isnan(exps_train) ^ np.isnan(X_ori)

saits.save("saits_weights/saits_all_artificial.pypots", overwrite=True)
saits.load("saits_weights/saits_all_artificial.pypots")

mae, mse, mre = calculate_metrics(imputation, np.nan_to_num(X_ori), indicating_mask)
print("Trainning metrics")
print(f'MAE: {mae:.3f}, MSE: {mse:.3f}, MRE: {mre:.3f}')



# Preparar el dataset de prueba
X_test_ori = exps_test_ori  # Datos originales sin alteraciones
X_test = exps_test  # Aplicar el mismo patrón de enmascaramiento

test_dataset = {"X": X_test}
imputation_test = saits.impute(test_dataset)
indicating_mask_test = np.isnan(X_test) ^ np.isnan(X_test_ori)
mae_test, mse_test, mre_test = calculate_metrics(imputation_test, np.nan_to_num(X_test_ori), indicating_mask_test)

# Mostrar resultados de la evaluación
print('Testing Results:')
print(f'MAE: {mae_test:.3f}, MSE: {mse_test:.3f}, MRE: {mre_test:.3f}')


from sklearn.model_selection import KFold
import numpy as np
import torch

# Datos originales
X_ori = exps_train_ori
np.random.seed(42)
torch.manual_seed(42)

def train_and_evaluate_kfold(X, X_ori, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"mae": [], "mse": [], "mre": []}

    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"Training fold {fold}/{n_splits}")
        
        # Crear conjuntos de entrenamiento y prueba
        X_train, X_test = X[train_index], X[test_index]
        X_ori_train, X_ori_test = X_ori[train_index], X_ori[test_index]

        dataset_train = {"X": X_train,"X_ori": X_ori_train }
        dataset_test = {"X": X_test,"X_ori": X_ori_test }

        # Crear y entrenar el modelo SAITS
        saits = SAITS(n_steps=newL, n_features=1, n_layers=2, d_model=256, n_heads=4, 
                      d_k=64, d_v=64, d_ffn=128, dropout=0.2, epochs=500, 
                     optimizer=custom_optimizer, model_saving_strategy='better')
                     
        saits.fit(train_set=dataset_train,val_set=dataset_test)

        # Imputar los valores faltantes
        imputation = saits.impute(dataset_test)

        # Máscara de valores imputados
        indicating_mask = np.isnan(X_test) ^ np.isnan(X_ori_test)

        # Calcular métricas
        mae, mse, mre = calculate_metrics(imputation, np.nan_to_num(X_ori_test), indicating_mask)
        metrics["mae"].append(mae)
        metrics["mse"].append(mse)
        metrics["mre"].append(mre)
        saits.save(F'saits_weights/saits_all_artificial_4_kfolds_{fold}.pypots', overwrite=True)
        print(f"Fold {fold} metrics: MAE: {mae:.3f}, MSE: {mse:.3f}, MRE: {mre:.3f}")

        print(f"Fold {fold} metrics: MAE: {mae:.3f}, MSE: {mse:.3f}, MRE: {mre:.3f}")
        fold += 1

    # Promediar las métricas
    mean_mae = np.mean(metrics["mae"])
    mean_mse = np.mean(metrics["mse"])
    mean_mre = np.mean(metrics["mre"])

    print("\nK-Fold Cross Validation Results")
    print(f"Average MAE: {mean_mae:.3f}, Average MSE: {mean_mse:.3f}, Average MRE: {mean_mre:.3f}")

    return metrics

# Llamar la función con K-Folds
n_splits = 10
metrics = train_and_evaluate_kfold(exps_train, exps_train_ori, n_splits=n_splits)

# Guardar los pesos finales
saits.save("saits_weights/saits_all_artificial_kfolds.pypots", overwrite=True)
saits.load("saits_weights/saits_all_artificial_kfolds.pypots")


# Crear boxplots
plt.figure(figsize=(12, 6))  # <-- Ensure this line has no leading whitespace

plt.subplot(1, 3, 1)
plt.boxplot(metrics["mae"], vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
plt.title("MAE Boxplot")
plt.ylabel("MAE")

plt.subplot(1, 3, 2)
plt.boxplot(metrics["mse"], vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.title("MSE Boxplot")
plt.ylabel("MSE")

plt.subplot(1, 3, 3)
plt.boxplot(metrics["mre"], vert=True, patch_artist=True, boxprops=dict(facecolor="lightcoral"))
plt.title("MRE Boxplot")
plt.ylabel("MRE")

plt.tight_layout()
plt.savefig('Box_Plot_SAITS.png', dpi=300, bbox_inches='tight')  # Save the figure BEFORE plt.show()
plt.show()


# Evaluate on test dataset
test_dataset = {"X": exps_test}
imputation_test = saits.impute(test_dataset)
indicating_mask_test = np.isnan(exps_test) ^ np.isnan(exps_test_ori)
mae_test, mse_test, mre_test = calculate_metrics(imputation_test, np.nan_to_num(exps_test_ori), indicating_mask_test)
print('Testing Results:')
print(f'MAE: {mae_test:.3f}, MSE: {mse_test:.3f}, MRE: {mre_test:.3f}')




np.random.seed(42)  # For reproducibility
X_test = exps_test       # Test data with the same masking applied
test_dataset = {"X": X_test}

# Number of folds
num_folds = 10

# Lists to store metrics from each fold
mae_list = []
mse_list = []
mre_list = []

for fold in range(1, num_folds + 1):
    print(f"Processing fold {fold}...")  
    # Construct the file path for the current fold
    model_path = f"saits_weights/saits_all_artificial_4_kfolds_{fold}.pypots"
    
    # Load the saved model weights for the current fold
    saits.load(model_path)
    
    # Perform imputation on the test dataset
    imputation_test = saits.impute(test_dataset)
    
    # Calculate the indicating mask: True where exactly one of the original or test is NaN
    indicating_mask_test = np.isnan(X_test) ^ np.isnan(X_test_ori)
    
    # Compute metrics using the provided function
    mae, mse, mre = calculate_metrics(imputation_test, np.nan_to_num(X_test_ori), indicating_mask_test)
    
    # Save metrics from this fold
    mae_list.append(mae)
    mse_list.append(mse)
    mre_list.append(mre)
    
    print(f"Fold {fold} - MAE: {mae:.3f}, MSE: {mse:.3f}, MRE: {mre:.3f}")

# Compute the average of the metrics over all folds
avg_mae = np.mean(mae_list)
avg_mse = np.mean(mse_list)
avg_mre = np.mean(mre_list)

print("\nAverage Metrics over 10 folds:")
print(f"Avg MAE: {avg_mae:.3f}, Avg MSE: {avg_mse:.3f}, Avg MRE: {avg_mre:.3f}")



reshaped_results_all = np.squeeze(imputation_test)
reshaped_x_test = np.squeeze(exps_test_ori)

plt.figure(figsize=(10, 6))
plt.plot(reshaped_x_test[30], label='Original', linewidth=3)
plt.plot(reshaped_results_all[30], label='Imputed', linestyle='-', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig('Imputation_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

