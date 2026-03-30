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
import navpy

from matplotlib import pyplot as plt

from lc_ekf_epoch import lc_ekf_epoch, update
from euler_to_ctm import euler_to_ctm
from initialize_ned_attitude import initialize_ned_attitude
from ctm_to_euler import ctm_to_euler
from calculate_errors_ned import calculate_errors_ned
from initialize_lc_p_matrix import initialize_lc_p_matrix
from kinematics_ned import kinematics_ned
from nav_equations_ned import nav_equations_ned
from body_to_ned import body_to_ned
from P_predict import P_predict
from skew_symmetric import skew_symmetric
from radii_of_curvature import radii_of_curvature


def lc_ins_dvl_sim(
        in_imu_profile: np.ndarray,
        in_dvl_profile: np.ndarray,
        in_gt_profile: np.ndarray,
        no_epochs: int,
        dvl_config: Dict[str, Any],
        lc_kf_config: Dict[str, float],
        collect_data: bool = False,
        compensator=None,
        update_P_after_dnn: bool = False,
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
    out_kf_sd : np.ndarray
        Output Kalman filter state uncertainties
    """

    # Constants
    deg_to_rad = 0.01745329252
    rad_to_deg = 1 / deg_to_rad
    micro_g_to_meters_per_second_squared = 9.80665e-6

    # Initialize true navigation solution
    old_time = in_gt_profile[0, 0]
    true_L_b = in_gt_profile[0, 1]
    true_lambda_b = in_gt_profile[0, 2]
    true_h_b = in_gt_profile[0, 3]
    true_eul_n_to_b = in_gt_profile[0, 7:10].copy()
    true_C_b_to_n = euler_to_ctm(true_eul_n_to_b).T
    true_v_eb_n = in_gt_profile[0, 4:7].copy()

    # Initialize Kalman filter P matrix and IMU bias states
    P_matrix = initialize_lc_p_matrix(lc_kf_config)
    est_imu_bias = np.zeros(6)

    # Generate KF uncertainty record
    out_kf_sd = np.zeros((no_epochs, 16))
    out_kf_sd[0, 0] = old_time
    for i in range(15):
        out_kf_sd[0, i + 1] = np.sqrt(P_matrix[i, i])

    P_diag = np.diag(P_matrix)
    std_devs = np.sqrt(P_diag)

    # Step 2: Generate initial errors (50% of std with random sign)
    np.random.seed(42)  # For reproducibility (change or remove for different errors each run)
    initial_errors = np.zeros(15)

    for i in range(15):
        sign = np.random.choice([-1, 1])  # Random positive or negative
        initial_errors[i] = sign * 0.5 * std_devs[i]  # 50% of standard deviation


    # Convert initial position errors from meters to radians for lat/lon
    R_N_init, R_E_init = radii_of_curvature(true_L_b)
    old_est_L_b = true_L_b - initial_errors[6] / R_N_init
    old_est_lambda_b = true_lambda_b - initial_errors[7] / ((R_E_init + true_h_b) * np.cos(true_L_b) + 1e-10)
    old_est_h_b = true_h_b - initial_errors[8]
    old_est_v_eb_n = true_v_eb_n + initial_errors[3:6]
    delta_psi_skew = skew_symmetric(initial_errors[0:3])
    old_est_C_b_to_n = (np.eye(3) + delta_psi_skew) @ true_C_b_to_n

    # CRITICAL: Re-orthonormalize to ensure it's a valid rotation matrix
    U, _, Vt = np.linalg.svd(old_est_C_b_to_n)
    old_est_C_b_to_n = U @ Vt

    # Bias errors (simple addition)
    old_est_b_a = initial_errors[9:12].copy()
    old_est_b_g = initial_errors[12:15].copy()
    est_imu_bias = initial_errors[9:15].copy()  # KF bias estimate must match initial perturbation


    # Initialize output arrays
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))


    # Generate initial output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[0, 4:7] = old_est_v_eb_n
    out_profile[0, 7:10] = ctm_to_euler(old_est_C_b_to_n.T)

    # # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
        old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_to_n,
        true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_to_n
    )

    out_errors[0, 0] = old_time
    out_errors[0, 1:4] = delta_r_eb_n
    out_errors[0, 4:7] = delta_v_eb_n
    out_errors[0, 7:10] = delta_eul_nb_n


    # Initialize IMU quantization residuals
    quant_residuals = np.zeros(6)

    # Determine number of DVL epochs
    num_dvl_epochs = int(np.ceil((in_dvl_profile[-1, 0] - old_time) /
                                 dvl_config['epoch_interval'])) + 1

    time_dvl = np.zeros(num_dvl_epochs)

    # Generate IMU bias and clock output records
    out_imu_bias_est = np.zeros((no_epochs, 7))
    out_imu_bias_est[0, 0] = old_time
    out_imu_bias_est[0, 1:7] = est_imu_bias

    # out_clock = np.zeros((num_gnss_epochs, 3))
    # out_clock[0, 0] = old_time
    # out_clock[0, 1:3] = est_clock


    # Initialize DVL model timing
    time_last_dvl = old_time
    dvl_epoch = 0

    # DNN data collection buffers (filled at every DVL epoch)
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
        true_L_b = in_gt_profile[epoch, 1]
        true_lambda_b = in_gt_profile[epoch, 2]
        true_h_b = in_gt_profile[epoch, 3]
        true_eul_n_to_b = in_gt_profile[epoch, 7:10].copy()
        true_C_b_to_n = euler_to_ctm(true_eul_n_to_b).T
        true_v_eb_n = in_gt_profile[epoch, 4:7].copy()

        # true_r_eb_e, true_v_eb_e, true_C_b_e = ned_to_ecef(
        #     true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_to_n
        # )

        # Time interval
        tor_i = time - old_time


        # meas_f_ib_b = in_imu_profile[epoch, 1:4]
        # meas_omega_ib_b = in_imu_profile[epoch, 4:7]
        meas_f_ib_b = in_imu_profile[epoch, 1:4] - old_est_b_a
        meas_omega_ib_b = in_imu_profile[epoch, 4:7] - old_est_b_g

        # Prediction
        P_matrix = P_predict(tor_i, old_est_C_b_to_n, old_est_v_eb_n, old_est_L_b, old_est_lambda_b, old_est_h_b, P_matrix, meas_f_ib_b, meas_omega_ib_b, lc_kf_config)

        # Generate KF uncertainty output record
        out_kf_sd[epoch, 0] = time
        for i in range(15):
            out_kf_sd[epoch, i + 1] = np.sqrt(P_matrix[i, i])

        P_diag = np.diag(P_matrix)
        std_devs = np.sqrt(P_diag)


        # Update estimated navigation solution
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_to_n = nav_equations_ned(
            tor_i, old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_to_n,
            meas_f_ib_b, meas_omega_ib_b
        )

        # Determine whether to update DVL update and run Kalman filter
        if (time - time_last_dvl) >= dvl_config['epoch_interval']:
            dvl_epoch += 1
            tor_s = time - time_last_dvl
            time_last_dvl = time
            time_dvl[dvl_epoch] = time_last_dvl

            # Input data from motion profile
            true_L_b = in_gt_profile[epoch, 1]
            true_lambda_b = in_gt_profile[epoch, 2]
            true_h_b = in_gt_profile[epoch, 3]
            true_eul_n_to_b = in_gt_profile[epoch, 7:10].copy()
            true_C_b_to_n = euler_to_ctm(true_eul_n_to_b).T
            true_v_eb_n = in_gt_profile[epoch, 4:7].copy()


            dvl_v_eb_b = in_dvl_profile[dvl_epoch, 1:4]

            # --- DNN: capture pre-update features ---
            C_nb_pre     = est_C_b_to_n.T
            innovation   = C_nb_pre @ est_v_eb_n - dvl_v_eb_b
            v_ekf_pre    = est_v_eb_n.copy()
            euler_pre    = ctm_to_euler(C_nb_pre)

            # Run Integration Kalman filter
            est_C_b_to_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b, est_imu_bias, P_matrix = lc_ekf_epoch(
                dvl_v_eb_b, tor_s, est_C_b_to_n, est_v_eb_n, est_L_b, est_lambda_b, est_h_b, est_imu_bias, P_matrix,
                meas_f_ib_b, meas_omega_ib_b, lc_kf_config
            )

            # --- DNN: apply velocity compensation (inference mode) ---
            if compensator is not None:
                correction = compensator.update(innovation, dvl_v_eb_b, v_ekf_pre, euler_pre)
                if correction is not None:
                    est_v_eb_n = est_v_eb_n + correction

                    # --- Optionally update P after DNN correction ---
                    # Treats the DNN correction as a virtual NED-velocity measurement.
                    # H selects velocity states (indices 3:6) directly in NED frame.
                    # Toggle via update_P_after_dnn=True / False (default False = no change).
                    if update_P_after_dnn:
                        H_dnn = np.zeros((3, 15))
                        H_dnn[0:3, 3:6] = np.eye(3)
                        dnn_sd = lc_kf_config.get('dnn_vel_SD', lc_kf_config['vel_meas_SD'])
                        R_dnn = np.eye(3) * dnn_sd ** 2
                        P_matrix, _ = update(P_matrix, H_dnn, R_dnn)

            # --- DNN: store training sample (collect_data mode) ---
            if collect_data:
                feat  = np.concatenate([innovation, dvl_v_eb_b, v_ekf_pre, euler_pre])
                label = true_v_eb_n - est_v_eb_n          # remaining error after EKF update
                dvl_features_list.append(feat)
                dvl_labels_list.append(label)

            # Generate IMU bias and clock output records
            # out_imu_bias_est[epoch, 0] = time
            # out_imu_bias_est[epoch, 1:7] = est_imu_bias
            # out_clock[gnss_epoch, 0] = time
            # out_clock[gnss_epoch, 1:3] = est_clock

            # Generate KF uncertainty output record
            out_kf_sd[epoch, 0] = time
            for i in range(15):
                out_kf_sd[epoch, i + 1] = np.sqrt(P_matrix[i, i])

            P_diag = np.diag(P_matrix)
            std_devs = np.sqrt(P_diag)

        if(time>2000):
            print("breakpoint")

        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = est_L_b
        out_profile[epoch, 2] = est_lambda_b
        out_profile[epoch, 3] = est_h_b
        out_profile[epoch, 4:7] = est_v_eb_n
        out_profile[epoch, 7:10] = ctm_to_euler(est_C_b_to_n.T)

        out_imu_bias_est[epoch, 0] = time
        out_imu_bias_est[epoch, 1:7] = est_imu_bias

        # Determine errors and generate output record
        delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = calculate_errors_ned(
            est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_to_n,
            true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_to_n
        )


        out_errors[epoch, 0] = time
        out_errors[epoch, 1:4] = delta_r_eb_n
        out_errors[epoch, 4:7] = delta_v_eb_n
        out_errors[epoch, 7:10] = delta_eul_nb_n

        # Reset old values
        old_time = time
        # old_true_r_eb_e = true_r_eb_e.copy()
        # old_true_v_eb_e = true_v_eb_e.copy()
        # old_true_C_b_e = true_C_b_e.copy()
        old_est_L_b = est_L_b
        old_est_lambda_b = est_lambda_b
        old_est_h_b = est_h_b
        old_est_v_eb_n = est_v_eb_n.copy()
        old_est_C_b_to_n = est_C_b_to_n.copy()
        old_est_b_a = est_imu_bias[0:3]
        old_est_b_g = est_imu_bias[3:6]

    # Complete progress bar
    print()

    # # Trim output arrays to actual GNSS epochs
    # out_imu_bias_est = out_imu_bias_est[:dvl_epoch + 1, :]
    # # out_clock = out_clock[:gnss_epoch + 1, :]
    # out_kf_sd = out_kf_sd[:no_epochs + 1, :]

    # fig_3d = plot_3d_trajectory(in_gt_profile, out_profile)

    if collect_data:
        dvl_features = np.array(dvl_features_list)   # (N_dvl, 12)
        dvl_labels   = np.array(dvl_labels_list)     # (N_dvl,  3)
        return out_profile, out_errors, out_imu_bias_est, out_kf_sd, dvl_features, dvl_labels

    return out_profile, out_errors, out_imu_bias_est, out_kf_sd


def plot_3d_trajectory(in_gt_profile, out_profile):
    """
    Plot 3D trajectory comparing ground truth and estimated navigation solutions.

    Parameters
    ----------
    in_gt_profile : np.ndarray
        Ground truth profile array (no_epochs x 10)
    out_profile : np.ndarray
        Estimated navigation solution profile array (no_epochs x 10)
    """

    from mpl_toolkits.mplot3d import Axes3D

    # Extract ground truth data
    gt_lat = np.rad2deg(in_gt_profile[:, 1])  # latitude (convert rad to deg for navpy)
    gt_lon = np.rad2deg(in_gt_profile[:, 2])  # longitude (convert rad to deg for navpy)
    gt_h = in_gt_profile[:, 3]  # height (m)

    # Extract estimated data
    est_lat = np.rad2deg(out_profile[:, 1])  # latitude (convert rad to deg for navpy)
    est_lon = np.rad2deg(out_profile[:, 2])  # longitude (convert rad to deg for navpy)
    est_h = out_profile[:, 3]  # height (m)

    # Debug: Print some values to verify data
    print(f"\nGT - First position: lat={gt_lat[0]:.6f}°, lon={gt_lon[0]:.6f}°, h={gt_h[0]:.3f}m")
    print(f"EST - First position: lat={est_lat[0]:.6f}°, lon={est_lon[0]:.6f}°, h={est_h[0]:.3f}m")
    print(f"GT - Last position: lat={gt_lat[-1]:.6f}°, lon={gt_lon[-1]:.6f}°, h={gt_h[-1]:.3f}m")
    print(f"EST - Last position: lat={est_lat[-1]:.6f}°, lon={est_lon[-1]:.6f}°, h={est_h[-1]:.3f}m")

    # Use first GT position as reference point
    lat_ref = gt_lat[0]
    lon_ref = gt_lon[0]
    h_ref = gt_h[0]

    # Convert GT trajectory to NED coordinates using navpy
    gt_ned = np.array([navpy.lla2ned(gt_lat[i], gt_lon[i], gt_h[i],
                                     lat_ref, lon_ref, h_ref)
                       for i in range(len(gt_lat))])
    gt_north = gt_ned[:, 0]
    gt_east = gt_ned[:, 1]
    gt_down = gt_ned[:, 2]

    # Convert estimated trajectory to NED coordinates using navpy
    est_ned = np.array([navpy.lla2ned(est_lat[i], est_lon[i], est_h[i],
                                      lat_ref, lon_ref, h_ref)
                        for i in range(len(est_lat))])
    est_north = est_ned[:, 0]
    est_east = est_ned[:, 1]
    est_down = est_ned[:, 2]

    # Debug: Print converted coordinates
    print(f"\nGT North range: [{gt_north.min():.3f}, {gt_north.max():.3f}] m")
    print(f"EST North range: [{est_north.min():.3f}, {est_north.max():.3f}] m")
    print(f"GT East range: [{gt_east.min():.3f}, {gt_east.max():.3f}] m")
    print(f"EST East range: [{est_east.min():.3f}, {est_east.max():.3f}] m")
    print(f"GT Down range: [{gt_down.min():.3f}, {gt_down.max():.3f}] m")
    print(f"EST Down range: [{est_down.min():.3f}, {est_down.max():.3f}] m")

    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth trajectory
    ax.plot(gt_north, gt_east, gt_down,
            'b-', linewidth=2.5, label='Ground Truth', alpha=0.8)

    # Plot estimated trajectory
    ax.plot(est_north, est_east, est_down,
            'r--', linewidth=2, label='Estimated', alpha=0.8)

    # Mark start and end points
    ax.scatter(gt_north[0], gt_east[0], gt_down[0],
               c='green', s=150, marker='o', label='Start', zorder=5,
               edgecolors='black', linewidths=2)
    ax.scatter(gt_north[-1], gt_east[-1], gt_down[-1],
               c='red', s=150, marker='s', label='End (GT)', zorder=5,
               edgecolors='black', linewidths=2)
    ax.scatter(est_north[-1], est_east[-1], est_down[-1],
               c='orange', s=150, marker='^', label='End (EST)', zorder=5,
               edgecolors='black', linewidths=2)

    # Labels and formatting
    ax.set_xlabel('North (m)', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('East (m)', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_zlabel('Down (m)', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_title('3D Trajectory: Ground Truth vs Estimated',
                 fontsize=16, fontweight='bold', pad=20)

    # Invert z-axis to show depth correctly
    ax.invert_zaxis()

    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Set viewing angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    return fig