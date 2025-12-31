# Author: Guy Damari
# Date: December 15, 2025


import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import random
from torch import nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from sklearn.model_selection import train_test_split
from math import cos, radians
from lc_ins_dvl_real import lc_ins_dvl_real
from read_profile import read_profile
from plot_errors import plot_errors_with_std, plot_results


# Constants
deg_to_rad = 0.01745329252
rad_to_deg = 1 / deg_to_rad
micro_g_to_meters_per_second_squared = 9.80665e-6


# ============================================================================
# CONFIGURATION
# ============================================================================

# Input truth motion profile filename
input_profile_name = 'Profile_1.csv'
# Output motion profile and error filenames
output_profile_name = 'INS_GNSS_Demo_3_Profile.csv'
output_errors_name = 'INS_GNSS_Demo_3_Errors.csv'

# ============================================================================
# INITIALIZATION ERRORS
# ============================================================================

initialization_errors = {
    # Attitude initialization error (deg, converted to rad; @N,E,D)
    'delta_eul_nb_n': np.array([-0.0, 0.0, 0.0]) * deg_to_rad,  # rad
    # Position and velocity errors (set to zero if not specified)
    'delta_r_eb_n': np.array([0.0, 0.0, 0.0]),  # m
    'delta_v_eb_n': np.array([0.0, 0.0, 0.0])   # m/s
}

# ============================================================================
# IMU ERRORS
# ============================================================================

IMU_errors = {
    # Accelerometer biases (micro-g, converted to m/s^2; body axes)
    'b_a': np.array([900, -1300, 800]) * micro_g_to_meters_per_second_squared,

    # Gyro biases (deg/hour, converted to rad/sec; body axes)
    'b_g': np.array([-9, 13, -8]) * deg_to_rad / 3600,

    # Accelerometer scale factor and cross coupling errors (ppm, converted to unitless; body axes)
    'M_a': np.array([
        [500, -300, 200],
        [-150, -600, 250],
        [-250, 100, 450]
    ]) * 1e-6,

    # Gyro scale factor and cross coupling errors (ppm, converted to unitless; body axes)
    'M_g': np.array([
        [400, -300, 250],
        [0, -300, -150],
        [0, 0, -350]
    ]) * 1e-6,

    # Gyro g-dependent biases (deg/hour/g, converted to rad-sec/m; body axes)
    'G_g': np.array([
        [0.9, -1.1, -0.6],
        [-0.5, 1.9, -1.6],
        [0.3, 1.1, -1.3]
    ]) * deg_to_rad / (3600 * 9.80665),

    # Accelerometer noise root PSD (micro-g per root Hz, converted to m s^-1.5)
    'accel_noise_root_PSD': 100 * micro_g_to_meters_per_second_squared,

    # Gyro noise root PSD (deg per root hour, converted to rad s^-0.5)
    'gyro_noise_root_PSD': 0.01 * deg_to_rad / 60,

    # Accelerometer quantization level (m/s^2)
    'accel_quant_level': 1e-2,

    # Gyro quantization level (rad/s)
    'gyro_quant_level': 2e-4
}

# ============================================================================
# DVL CONFIG
# ============================================================================
#Interval between DVL epochs (s)
DVL_config = {
        'epoch_interval': 1.002506
}

# ============================================================================
# KALMAN FILTER CONFIGURATION
# ============================================================================

LC_KF_config = {
    # # Initial attitude uncertainty per axis (deg, converted to rad)
    # 'init_att_unc': np.deg2rad(np.sqrt(5)),
    #
    # # Initial velocity uncertainty per axis (m/s)
    # 'init_vel_unc': np.sqrt(0.2),
    #
    # # Initial position uncertainty per axis (m)
    # 'init_pos_unc': np.sqrt(20),
    #
    # # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
    # 'init_b_a_unc': np.sqrt(30000) * micro_g_to_meters_per_second_squared,
    #
    # # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
    # 'init_b_g_unc': np.sqrt(30) * deg_to_rad / 3600,

    # # Initial attitude uncertainty per axis (deg, converted to rad)
    # 'init_att_unc': np.deg2rad(5),
    #
    # # Initial velocity uncertainty per axis (m/s)
    # 'init_vel_unc': 0.2,
    #
    # # Initial position uncertainty per axis (m)
    # 'init_pos_unc': 20,
    #
    # # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
    # 'init_b_a_unc': 1000 * micro_g_to_meters_per_second_squared,
    #
    # # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
    # 'init_b_g_unc': 10 * deg_to_rad / 3600,
    #
    # # Gyro noise PSD (deg^2 per hour, converted to rad^2/s)
    # 'gyro_noise_PSD': (0.02 * deg_to_rad / 60) ** 2,
    #
    # # Accelerometer noise PSD (micro-g^2 per Hz, converted to m^2 s^-3)
    # 'accel_noise_PSD': (200 * micro_g_to_meters_per_second_squared) ** 2,
    #
    # # Accelerometer bias random walk PSD (m^2 s^-5)
    # 'accel_bias_PSD': 1.0e-7,
    #
    # # Gyro bias random walk PSD (rad^2 s^-3)
    # 'gyro_bias_PSD': 2.0e-12,
    #
    # # Position measurement noise SD per axis (m)
    # 'pos_meas_SD': 2.5,
    #
    # # Velocity measurement noise SD per axis (m/s)
    # 'vel_meas_SD': 0.5


    # Initial attitude uncertainty per axis (deg, converted to rad)
    'init_att_unc': np.deg2rad(0.1),

    # Initial velocity uncertainty per axis (m/s)
    'init_vel_unc': 0.2,

    # Initial position uncertainty per axis (m)
    'init_pos_unc': 0.1,

    # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
    'init_b_a_unc': 10 * micro_g_to_meters_per_second_squared,

    # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
    'init_b_g_unc': 0.001 * deg_to_rad / 3600,

    # # Gyro noise PSD (deg^2 per hour, converted to rad^2/s)
    # 'gyro_noise_PSD': (1.0 * deg_to_rad / 60) ** 2,
    #
    # # Accelerometer noise PSD (micro-g^2 per Hz, converted to m^2 s^-3)
    # 'accel_noise_PSD': (100 * micro_g_to_meters_per_second_squared) ** 2,
    #
    # # Accelerometer bias random walk PSD (m^2 s^-5)
    # 'accel_bias_PSD': 1.0e-7,
    #
    # # Gyro bias random walk PSD (rad^2 s^-3)
    # 'gyro_bias_PSD': 2.0e-12,

    # Moderate process noise (despite zero in simulation)
    'gyro_noise_PSD': (0.5 * deg_to_rad / 60) ** 2,
    'accel_noise_PSD': (0 * micro_g_to_meters_per_second_squared) ** 2,

    # Accelerometer bias random walk PSD (m^2 s^-5)
    'accel_bias_PSD': 1.0e-1,

    # Gyro bias random walk PSD (rad^2 s^-3)
    'gyro_bias_PSD': 2.0e-6,

    # # Position measurement noise SD per axis (m)
    # 'pos_meas_SD': 0.5,

    # Velocity measurement noise SD per axis (m/s)
    'vel_meas_SD': 0.05
}


def main(config):
    # Set random seed for reproducibility
    np.random.seed(1)

    # # Read input truth motion profile from CSV file
    # print(f"\nReading input profile: {input_profile_name}")
    # try:
    #     # Simple CSV reading with numpy
    #     # Assumes CSV has no header, 10 columns as specified in motion profile format
    #     in_profile, no_epochs = read_profile(input_profile_name)
    #     print(f"Successfully loaded {no_epochs} epochs")
    #     print(f"Time span: {in_profile[0, 0]:.2f}s to {in_profile[-1, 0]:.2f}s")
    #
    # except FileNotFoundError:
    #     print(f"Error: Could not find file '{input_profile_name}'")
    #     print("Please ensure the input profile CSV file is in the current directory.")
    #     return
    # except Exception as e:
    #     print(f"Error reading profile: {e}")
    #     return

    data_path = config['data_path']
    real_data_trajectory_index = config['real_data_trajectory_index']
    gt_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'GT_{real_data_trajectory_index}.csv'),
                        header=0, names=None)
    imu_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'IMU_{real_data_trajectory_index}.csv'),
                         header=0, names=None)
    dvl_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'DVL_{real_data_trajectory_index}.csv'),
                         header=0, names=None)

    in_gt_profile = np.array(gt_pd.iloc[:, 0:10])
    in_imu_profile = np.array(imu_pd.iloc[:, 0:7])
    # in_imu_profile[:, 1:7] *= 100
    in_dvl_profile = np.array(dvl_pd.iloc[:, 0:4])

    # Determine size of file
    no_epochs, no_columns = in_imu_profile.shape

    # Run loosely coupled ECEF Inertial navigation and GNSS integrated navigation simulation

    out_profile, out_errors, out_IMU_bias_est, out_KF_SD = lc_ins_dvl_real(
        in_imu_profile,
        in_dvl_profile,
        in_gt_profile,
        no_epochs,
        initialization_errors,
        IMU_errors,
        DVL_config,
        LC_KF_config,
        config
    )


    # Plot results
    # plot_results(out_errors, out_KF_SD)

    # Plot errors with standard deviations
    fig_pos, fig_vel, fig_att, fig_bias_a, fig_bias_g = plot_errors_with_std(
        out_errors,
        out_KF_SD,
        out_IMU_bias_est
    )

    # train model ##################################################################
    if config['train_model']:
        print('Training model...')

    # Test Model section ##################################################################
    if config['test_model']:
        print('Testing model...')


if __name__ == '__main__':

    # User-defined configuration (can be read from a config file or command-line arguments)
    user_config = {
        'convex_dataset_len': 2054, # - turn pattern simulated for convex -  important!! you have to update it, from the data output file, and every time you change dataset
        'data_path': "C:\\Users\\damar\\PycharmProjects\\EKF_CalibrateNet",
        'test_type': 'transformed_real_data',  # Set to "convex_data" or "simulated_data" or "transformed_real_data" or "simulated_imu_from_real_gt_data" or "real_data"
        'train_model': True,
        'test_model': False,
        'test_baseline_model': True,
        'trained_model_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work\\trained_model",
        #'simulated_data_file_name': 'simulated_data_output.csv',
        'simulated_data_file_name': 'simulated_data_output_long_turn_17_+0_3125_ba_real_bg_10.csv',
        'real_data_trajectory_index': 'static', # 1-13
        # 'simulated_data_file_name': 'simulated_data_output_long_turn_17_+2_8125_ba_real_bg_10.csv',
        # imu_dvl_model_simulated_data_straight_line_17_ + 2_8125_ba_real_bg_10_window_75
        # 'simulated_data_file_name': 'simulated_data_output.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj7_17_+0_3125.csv',
        'transformed_real_data_file_name': 'transformed_real_data_output_traj11_16_+0_3333.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj11_26_+1.csv',
    }

    # orzi_euler_config = {
    #     'roll_gt_deg': -179.9,
    #     'pitch_gt_deg': 0.2,
    #     'yaw_gt_deg': -44.3,
    # }

    # Merge default and user configurations
    config = {**user_config}

    main(config)

    plt.show()


