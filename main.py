# Author: Guy Damari
# Date: December 15, 2025


import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from lc_ins_dvl_real import lc_ins_dvl_real
from lc_ins_dvl_sim import lc_ins_dvl_sim
from plot_errors import plot_errors_with_std, plot_results


# Constants
deg_to_rad = 0.01745329252
rad_to_deg = 1 / deg_to_rad
micro_g_to_meters_per_second_squared = 9.80665e-6


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

    # Initial attitude uncertainty per axis (deg, converted to rad)
    'init_att_unc': np.deg2rad(0.1),

    # Initial velocity uncertainty per axis (m/s)
    'init_vel_unc': 0.2,

    # Initial position uncertainty per axis (m)
    'init_pos_unc': 0.01,

    # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
    'init_b_a_unc': 10 * micro_g_to_meters_per_second_squared,

    # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
    'init_b_g_unc': 0.01 * deg_to_rad / 3600,

    # Moderate process noise (despite zero in simulation)
    'gyro_noise_PSD': (0.005 * deg_to_rad / 60) ** 2,
    'accel_noise_PSD': (0.01 * micro_g_to_meters_per_second_squared) ** 2,

    # Accelerometer bias random walk PSD (m^2 s^-5)
    'accel_bias_PSD': 1.0e-8,

    # Gyro bias random walk PSD (rad^2 s^-3)
    'gyro_bias_PSD': 2.0e-9,

    # # Position measurement noise SD per axis (m)
    # 'pos_meas_SD': 0.5,

    # Velocity measurement noise SD per axis (m/s)
    'vel_meas_SD': 0.5
}


def main(config):
    # Set random seed for reproducibility
    np.random.seed(1)

    data_path = config['data_path']

    if config['data_type'] == "real":
        real_data_trajectory_index = config['real_data_trajectory_index']
        gt_pd = pd.read_csv(os.path.join(data_path, f'real_data', f'GT_trajectory{real_data_trajectory_index}.csv'),
                            header=0, names=None)
        imu_pd = pd.read_csv(os.path.join(data_path, f'real_data', f'IMU_trajectory{real_data_trajectory_index}.csv'),
                             header=0, names=None)
        dvl_pd = pd.read_csv(os.path.join(data_path, f'real_data', f'DVL_trajectory{real_data_trajectory_index}.csv'),
                             header=0, names=None)
        in_gt_profile = np.array(gt_pd.iloc[:, 0:10])
        in_imu_profile = np.array(imu_pd.iloc[:, 0:7])
        in_imu_profile[:, 1:7] *= 100
        in_dvl_profile = np.array(dvl_pd.iloc[:, 0:4])

        # Determine size of file
        no_epochs, no_columns = in_imu_profile.shape

        # Run loosely coupled NED Inertial navigation and DVL integrated navigation system
        out_profile, out_errors, out_IMU_bias_est, out_KF_SD = lc_ins_dvl_real(
            in_imu_profile,
            in_dvl_profile,
            in_gt_profile,
            no_epochs,
            DVL_config,
            LC_KF_config,
        )

    elif config['data_type'] == "sim":
        simulated_data_file_name = config['simulated_data_file_name']
        gt_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'GT_{simulated_data_file_name}.csv'),
                            header=0, names=None)
        imu_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'IMU_{simulated_data_file_name}.csv'),
                             header=0, names=None)
        dvl_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'DVL_{simulated_data_file_name}.csv'),
                             header=0, names=None)
        in_gt_profile = np.array(gt_pd.iloc[:, 0:10])
        in_imu_profile = np.array(imu_pd.iloc[:, 0:7])
        in_dvl_profile = np.array(dvl_pd.iloc[:, 0:4])

        # Determine size of file
        no_epochs, no_columns = in_imu_profile.shape

        # Run loosely coupled NED Inertial navigation and DVL integrated navigation system
        out_profile, out_errors, out_IMU_bias_est, out_KF_SD = lc_ins_dvl_sim(
            in_imu_profile,
            in_dvl_profile,
            in_gt_profile,
            no_epochs,
            DVL_config,
            LC_KF_config,
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
        'data_path': "C:\\Users\\damar\\PycharmProjects\\EKF_CalibrateNet",
        'data_type': 'sim',  # "sim" or "real"
        'simulated_data_file_name': 'static',
        'real_data_trajectory_index': '1',  # 1-13
        'train_model': False,
        'test_model': False,
        'test_baseline_model': False,
        'trained_model_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work\\trained_model",
    }


    # Merge default and user configurations
    config = {**user_config}

    main(config)

    plt.show()


