"""
Author: Guy Damari
Date: December 15, 2025
Description:

Loosely Coupled INS/GNSS Integration using Extended Kalman Filter

Simulates inertial navigation using ECEF navigation equations and kinematic model,
GNSS using a least-squares positioning algorithm, and loosely-coupled INS/GNSS integration.

Converted from MATLAB code by Paul Groves
Original: "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems," Second Edition.
"""

import numpy as np
import os
import pandas as pd
from typing import Tuple, Dict, Any
import sys
from lc_ekf_epoch import lc_ekf_epoch, lc_ekf_epoch_joint, update
from euler_to_ctm import euler_to_ctm
from initialize_ned_attitude import initialize_ned_attitude
from ctm_to_euler import ctm_to_euler
from calculate_errors_ned import calculate_errors_ned
from initialize_lc_p_matrix import initialize_lc_p_matrix
from kinematics_ned import kinematics_ned
from nav_equations_ned import nav_equations_ned
from body_to_ned import body_to_ned
from P_predict import P_predict


def lc_ins_dvl_real(
        in_imu_profile: np.ndarray,
        in_dvl_profile: np.ndarray,
        in_gt_profile: np.ndarray,
        no_epochs: int,
        dvl_config: Dict[str, Any],
        lc_kf_config: Dict[str, float],
        collect_data: bool = False,
        compensator=None,
        update_P_after_dnn: bool = False,
        dnn_mode: str = 'sequential',
) -> Tuple:
    """
    Loosely coupled INS/DVL integration using Extended Kalman Filter.

    Parameters
    ----------
    in_profile : np.ndarray
        True motion profile array (no_epochs x 10)
        Columns: time, lat, lon, height, v_n, v_e, v_d, roll, pitch, yaw
    no_epochs : int
        Number of epochs of profile data
    initialization_errors : dict
        - 'delta_r_eb_n': position error resolved along NED (m)
        - 'delta_v_eb_n': velocity error resolved along NED (m/s)
        - 'delta_eul_nb_n': attitude error as NED Euler angles (rad)
    imu_errors : dict
        - 'b_a': Accelerometer biases (m/s^2)
        - 'b_g': Gyro biases (rad/s)
        - 'M_a': Accelerometer scale factor and cross coupling errors
        - 'M_g': Gyro scale factor and cross coupling errors
        - 'G_g': Gyro g-dependent biases (rad-sec/m)
        - 'accel_noise_root_PSD': Accelerometer noise root PSD (m s^-1.5)
        - 'gyro_noise_root_PSD': Gyro noise root PSD (rad s^-0.5)
        - 'accel_quant_level': Accelerometer quantization level (m/s^2)
        - 'gyro_quant_level': Gyro quantization level (rad/s)
    lc_kf_config : dict
        - 'init_att_unc': Initial attitude uncertainty per axis (rad)
        - 'init_vel_unc': Initial velocity uncertainty per axis (m/s)
        - 'init_pos_unc': Initial position uncertainty per axis (m)
        - 'init_b_a_unc': Initial accel. bias uncertainty (m/s^2)
        - 'init_b_g_unc': Initial gyro. bias uncertainty (rad/s)
        - 'gyro_noise_PSD': Gyro noise PSD (rad^2/s)
        - 'accel_noise_PSD': Accelerometer noise PSD (m^2 s^-3)
        - 'accel_bias_PSD': Accelerometer bias random walk PSD (m^2 s^-5)
        - 'gyro_bias_PSD': Gyro bias random walk PSD (rad^2 s^-3)
        - 'pos_meas_SD': Position measurement noise SD per axis (m)
        - 'vel_meas_SD': Velocity measurement noise SD per axis (m/s)

    Returns
    -------
    out_profile : np.ndarray
        Navigation solution as a motion profile array
    out_errors : np.ndarray
        Navigation solution error array
    out_imu_bias_est : np.ndarray
        Kalman filter IMU bias estimate array
    out_clock : np.ndarray
        GNSS Receiver clock estimate array
    out_kf_sd : np.ndarray
        Output Kalman filter state uncertainties
    """

    if dnn_mode == 'joint' and update_P_after_dnn:
        print("[lc_ins_dvl_real] note: update_P_after_dnn is ignored in joint mode "
              "(P is updated jointly with the stacked DVL+DNN measurement).")

    # Initialize true navigation solution
    old_time = in_gt_profile[0, 0]
    true_L_b = in_gt_profile[0, 1]
    true_lambda_b = in_gt_profile[0, 2]
    true_h_b = in_gt_profile[0, 3]
    true_v_eb_n = in_gt_profile[0, 4:7].copy()    # GT cols 4:7 are V_N, V_E, V_D (already NED)
    true_eul_nb = in_gt_profile[0, 7:10].copy()
    true_C_b_n = euler_to_ctm(true_eul_nb).T

    # old_true_r_eb_e, old_true_v_eb_e, old_true_C_b_e = ned_to_ecef(
    #     true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
    # )

    # # Determine satellite positions and velocities
    # sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(
    #     old_time, gnss_config
    # )
    #
    # # Initialize GNSS biases
    # gnss_biases = initialize_gnss_biases(
    #     sat_r_es_e, old_true_r_eb_e, true_L_b, true_lambda_b, gnss_config
    # )

    # # Generate GNSS measurements
    # gnss_measurements, no_gnss_meas = generate_gnss_measurements(
    #     old_time, sat_r_es_e, sat_v_es_e, old_true_r_eb_e,
    #     true_L_b, true_lambda_b, old_true_v_eb_e, gnss_biases, gnss_config
    # )

    # # Determine Least-squares GNSS position solution
    # gnss_r_eb_e, gnss_v_eb_e, est_clock = gnss_ls_position_velocity(
    #     gnss_measurements, no_gnss_meas, gnss_config['init_est_r_ea_e'],
    #     np.array([0, 0, 0])
    # )
    #
    # old_est_r_eb_e = gnss_r_eb_e.copy()
    # old_est_v_eb_e = gnss_v_eb_e.copy()
    #
    # old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n = pv_ecef_to_ned(
    #     old_est_r_eb_e, old_est_v_eb_e
    # )
    # est_L_b = old_est_L_b

    # Initialize estimated attitude solution


    # _, _, old_est_C_b_e = ned_to_ecef(
    #     old_est_L_b, old_est_lambda_b, old_est_h_b,
    #     old_est_v_eb_n, old_est_C_b_n
    # )

    old_est_L_b = true_L_b
    old_est_lambda_b = true_lambda_b
    old_est_h_b = true_h_b
    old_est_v_eb_n = true_v_eb_n
    old_est_C_b_n = true_C_b_n
    old_est_b_a = np.zeros(3)
    old_est_b_g = np.zeros(3)


    # Initialize output arrays
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))

    # Generate initial output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[0, 4:7] = old_est_v_eb_n
    out_profile[0, 7:10] = ctm_to_euler(old_est_C_b_n)

    # # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
        old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n,
        true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
    )

    out_errors[0, 0] = old_time
    out_errors[0, 1:4] = delta_r_eb_n
    out_errors[0, 4:7] = delta_v_eb_n
    out_errors[0, 7:10] = delta_eul_nb_n

    # Initialize Kalman filter P matrix and IMU bias states
    P_matrix = initialize_lc_p_matrix(lc_kf_config)
    est_imu_bias = np.zeros(6)

    # # Initialize IMU quantization residuals
    # quant_residuals = np.zeros(6)

    # Determine number of DVL epochs
    num_dvl_epochs = int(np.ceil((in_dvl_profile[-1, 0] - old_time) /
                                 dvl_config['epoch_interval'])) + 1

    # Generate IMU bias and clock output records
    out_imu_bias_est = np.zeros((no_epochs, 7))
    out_imu_bias_est[0, 0] = old_time
    out_imu_bias_est[0, 1:7] = est_imu_bias

    # out_clock = np.zeros((num_gnss_epochs, 3))
    # out_clock[0, 0] = old_time
    # out_clock[0, 1:3] = est_clock

    # Generate KF uncertainty record
    out_kf_sd = np.zeros((no_epochs, 16))
    out_kf_sd[0, 0] = old_time
    for i in range(15):
        out_kf_sd[0, i + 1] = np.sqrt(P_matrix[i, i])

    # Initialize DVL model timing
    time_last_dvl = old_time
    dvl_epoch = 0

    # DNN data collection buffers (filled at every DVL epoch when collect_data=True)
    dvl_features_list = [] if collect_data else None
    dvl_labels_list   = [] if collect_data else None

    # Progress bar
    print('Processing: ', end='', flush=True)
    progress_mark = 0
    progress_epoch = 0

    # Main loop
    for epoch in range(1, no_epochs):
        # Update progress bar
        if (epoch - progress_epoch) > (no_epochs / 20):
            progress_mark += 1
            progress_epoch = epoch
            print('|', end='', flush=True)

        # Input data from motion profile
        time = in_imu_profile[epoch, 0]
        true_L_b = in_gt_profile[dvl_epoch, 1]
        true_lambda_b = in_gt_profile[dvl_epoch, 2]
        true_h_b = in_gt_profile[dvl_epoch, 3]
        true_v_eb_n = in_gt_profile[dvl_epoch, 4:7].copy()    # already NED
        true_eul_nb = in_gt_profile[dvl_epoch, 7:10].copy()
        true_C_b_n = euler_to_ctm(true_eul_nb).T

        # true_r_eb_e, true_v_eb_e, true_C_b_e = ned_to_ecef(
        #     true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
        # )

        # Time interval
        tor_i = time - old_time


        meas_f_ib_b = in_imu_profile[epoch, 1:4] - old_est_b_a
        meas_omega_ib_b = in_imu_profile[epoch, 4:7] - old_est_b_g

        # Prediction
        P_matrix = P_predict(tor_i, old_est_C_b_n, old_est_v_eb_n, old_est_L_b, old_est_lambda_b, old_est_h_b, P_matrix, meas_f_ib_b, meas_omega_ib_b, lc_kf_config)

        # Generate KF uncertainty output record
        out_kf_sd[epoch, 0] = time
        for i in range(15):
            out_kf_sd[epoch, i + 1] = np.sqrt(P_matrix[i, i])


        # Update estimated navigation solution
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n = nav_equations_ned(
            tor_i, old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n,
            meas_f_ib_b, meas_omega_ib_b
        )

        # Determine whether to update DVL update and run Kalman filter
        if (time - time_last_dvl) >= dvl_config['epoch_interval']:
            dvl_epoch += 1
            tor_s = time - time_last_dvl
            time_last_dvl = time

            # Input data from motion profile
            true_L_b = in_gt_profile[dvl_epoch, 1]
            true_lambda_b = in_gt_profile[dvl_epoch, 2]
            true_h_b = in_gt_profile[dvl_epoch, 3]
            true_v_eb_n = in_gt_profile[dvl_epoch, 4:7].copy()    # already NED
            true_eul_nb = in_gt_profile[dvl_epoch, 7:10].copy()
            true_C_b_n = euler_to_ctm(true_eul_nb).T


            dvl_v_eb_b = in_dvl_profile[dvl_epoch, 1:4]    # DVL body frame (X,Y,Z)

            # Capture pre-update state for DNN feature construction
            v_ekf_pre  = est_v_eb_n.copy()
            euler_pre  = ctm_to_euler(est_C_b_n)
            innovation = est_C_b_n.T @ v_ekf_pre - dvl_v_eb_b    # body_EKF - body_DVL

            if dnn_mode == 'joint':
                # Joint path: get DNN correction (if available), then run a single
                # combined DVL+DNN Kalman update. Cold-start (first W epochs) and
                # collect_data both fall back to DVL-only since no correction exists.
                correction = None
                if compensator is not None:
                    correction = compensator.update(innovation, dvl_v_eb_b, v_ekf_pre, euler_pre)
                if correction is None:
                    est_C_b_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b, est_imu_bias, P_matrix = lc_ekf_epoch(
                        dvl_v_eb_b, tor_s, est_C_b_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b, est_imu_bias, P_matrix,
                        meas_f_ib_b, meas_omega_ib_b, lc_kf_config
                    )
                else:
                    dnn_sd = lc_kf_config.get('dnn_vel_SD', lc_kf_config['vel_meas_SD'])
                    est_C_b_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b, est_imu_bias, P_matrix = lc_ekf_epoch_joint(
                        dvl_v_eb_b, correction, dnn_sd, tor_s, est_C_b_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b,
                        est_imu_bias, P_matrix, meas_f_ib_b, meas_omega_ib_b, lc_kf_config
                    )

                # Joint-mode label uses the pre-update residual (no DVL update applied yet)
                if collect_data:
                    feat  = np.concatenate([innovation, dvl_v_eb_b, v_ekf_pre, euler_pre])
                    label = true_v_eb_n - v_ekf_pre
                    dvl_features_list.append(feat)
                    dvl_labels_list.append(label)

            else:
                # Sequential path (default, current behavior): DVL update first, then DNN.
                est_C_b_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b, est_imu_bias, P_matrix = lc_ekf_epoch(
                    dvl_v_eb_b, tor_s, est_C_b_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b, est_imu_bias, P_matrix,
                    meas_f_ib_b, meas_omega_ib_b, lc_kf_config
                )

                # DNN training-data collection (one sample per DVL epoch)
                if collect_data:
                    feat  = np.concatenate([innovation, dvl_v_eb_b, v_ekf_pre, euler_pre])
                    label = true_v_eb_n - est_v_eb_n          # remaining error after EKF update
                    dvl_features_list.append(feat)
                    dvl_labels_list.append(label)

                # DNN velocity compensation (inference mode)
                if compensator is not None:
                    correction = compensator.update(innovation, dvl_v_eb_b, v_ekf_pre, euler_pre)
                    if correction is not None:
                        H_dnn = np.zeros((3, 15))
                        H_dnn[0:3, 3:6] = np.eye(3)
                        dnn_sd = lc_kf_config.get('dnn_vel_SD', lc_kf_config['vel_meas_SD'])
                        R_dnn = np.eye(3) * dnn_sd ** 2
                        P_matrix, K_dnn = update(P_matrix, H_dnn, R_dnn)
                        # Sign convention matches DVL: delta_z = v_nom - z_meas = -correction
                        x_dnn = K_dnn @ (-correction)
                        est_v_eb_n = est_v_eb_n - x_dnn[3:6]

            # Generate KF uncertainty output record
            out_kf_sd[epoch, 0] = time
            for i in range(15):
                out_kf_sd[epoch, i + 1] = np.sqrt(P_matrix[i, i])


        # Generate IMU bias output record (every IMU epoch)
        out_imu_bias_est[epoch, 0] = time
        out_imu_bias_est[epoch, 1:7] = est_imu_bias

        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = est_L_b
        out_profile[epoch, 2] = est_lambda_b
        out_profile[epoch, 3] = est_h_b
        out_profile[epoch, 4:7] = est_v_eb_n
        out_profile[epoch, 7:10] = ctm_to_euler(est_C_b_n.T)

        # Determine errors and generate output record
        delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
            est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n,
            true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
        )


        out_errors[epoch, 0] = time
        out_errors[epoch, 1:4] = delta_r_eb_n
        out_errors[epoch, 4:7] = delta_v_eb_n
        out_errors[epoch, 7:10] = delta_eul_nb_n

        # Reset old values
        old_time = time
        old_est_L_b = est_L_b
        old_est_lambda_b = est_lambda_b
        old_est_h_b = est_h_b
        old_est_v_eb_n = est_v_eb_n.copy()
        old_est_C_b_n = est_C_b_n.copy()
        old_est_b_a = est_imu_bias[0:3]
        old_est_b_g = est_imu_bias[3:6]

    # Complete progress bar
    print()

    # # Trim output arrays to actual GNSS epochs
    # out_imu_bias_est = out_imu_bias_est[:dvl_epoch + 1, :]
    # # out_clock = out_clock[:gnss_epoch + 1, :]
    # out_kf_sd = out_kf_sd[:no_epochs + 1, :]

    if collect_data:
        dvl_features = np.array(dvl_features_list)   # (N_dvl, 12)
        dvl_labels   = np.array(dvl_labels_list)     # (N_dvl,  3)
        return out_profile, out_errors, out_imu_bias_est, out_kf_sd, dvl_features, dvl_labels

    return out_profile, out_errors, out_imu_bias_est, out_kf_sd



