"""
Loosely Coupled INS/GNSS Integration using Extended Kalman Filter

Simulates inertial navigation using ECEF navigation equations and kinematic model,
GNSS using a least-squares positioning algorithm, and loosely-coupled INS/GNSS integration.

Converted from MATLAB code by Paul Groves
Original: "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems," Second Edition.
"""

import numpy as np
from typing import Tuple, Dict, Any
import sys
from lc_ekf_epoch import lc_ekf_epoch
from euler_to_ctm import euler_to_ctm


def loosely_coupled_ins_dvl(
        in_profile: np.ndarray,
        no_epochs: int,
        initialization_errors: Dict[str, np.ndarray],
        imu_errors: Dict[str, Any],
        gnss_config: Dict[str, Any],
        lc_kf_config: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    gnss_config : dict
        - 'epoch_interval': Interval between GNSS epochs (s)
        - 'init_est_r_ea_e': Initial estimated position (m; ECEF)
        - 'no_sat': Number of satellites
        - 'r_os': Orbital radius of satellites (m)
        - 'inclination': Inclination angle (deg)
        - 'const_delta_lambda': Longitude offset (deg)
        - 'const_delta_t': Timing offset (s)
        - 'mask_angle': Mask angle (deg)
        - 'SIS_err_SD': Signal in space error SD (m)
        - 'zenith_iono_err_SD': Zenith ionosphere error SD (m)
        - 'zenith_trop_err_SD': Zenith troposphere error SD (m)
        - 'code_track_err_SD': Code tracking error SD (m)
        - 'rate_track_err_SD': Range rate tracking error SD (m/s)
        - 'rx_clock_offset': Receiver clock offset at time=0 (m)
        - 'rx_clock_drift': Receiver clock drift at time=0 (m/s)
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

    # Initialize true navigation solution
    old_time = in_profile[0, 0]
    true_L_b = in_profile[0, 1]
    true_lambda_b = in_profile[0, 2]
    true_h_b = in_profile[0, 3]
    true_v_eb_n = in_profile[0, 4:7].copy()
    true_eul_nb = in_profile[0, 7:10].copy()
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

    old_est_r_eb_e = gnss_r_eb_e.copy()
    old_est_v_eb_e = gnss_v_eb_e.copy()

    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n = pv_ecef_to_ned(
        old_est_r_eb_e, old_est_v_eb_e
    )
    est_L_b = old_est_L_b

    # Initialize estimated attitude solution
    old_est_C_b_n = initialize_ned_attitude(true_C_b_n, initialization_errors)
    _, _, old_est_C_b_e = ned_to_ecef(
        old_est_L_b, old_est_lambda_b, old_est_h_b,
        old_est_v_eb_n, old_est_C_b_n
    )

    # Initialize output arrays
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))

    # Generate initial output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[0, 4:7] = old_est_v_eb_n
    out_profile[0, 7:10] = ctm_to_euler(old_est_C_b_n.T)

    # Determine errors and generate output record
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

    # Initialize IMU quantization residuals
    quant_residuals = np.zeros(6)

    # Determine number of GNSS epochs
    num_gnss_epochs = int(np.ceil((in_profile[-1, 0] - old_time) /
                                  gnss_config['epoch_interval'])) + 1

    # Generate IMU bias and clock output records
    out_imu_bias_est = np.zeros((num_gnss_epochs, 7))
    out_imu_bias_est[0, 0] = old_time
    out_imu_bias_est[0, 1:7] = est_imu_bias

    out_clock = np.zeros((num_gnss_epochs, 3))
    out_clock[0, 0] = old_time
    out_clock[0, 1:3] = est_clock

    # Generate KF uncertainty record
    out_kf_sd = np.zeros((num_gnss_epochs, 16))
    out_kf_sd[0, 0] = old_time
    for i in range(15):
        out_kf_sd[0, i + 1] = np.sqrt(P_matrix[i, i])

    # Initialize GNSS model timing
    time_last_gnss = old_time
    gnss_epoch = 0

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
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = in_profile[epoch, 4:7].copy()
        true_eul_nb = in_profile[epoch, 7:10].copy()
        true_C_b_n = euler_to_ctm(true_eul_nb).T

        true_r_eb_e, true_v_eb_e, true_C_b_e = ned_to_ecef(
            true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
        )

        # Time interval
        tor_i = time - old_time

        # Calculate specific force and angular rate
        true_f_ib_b, true_omega_ib_b = kinematics_ecef(
            tor_i, true_C_b_e, old_true_C_b_e,
            true_v_eb_e, old_true_v_eb_e, old_true_r_eb_e
        )

        # Simulate IMU errors
        meas_f_ib_b, meas_omega_ib_b, quant_residuals = imu_model(
            tor_i, true_f_ib_b, true_omega_ib_b, imu_errors, quant_residuals
        )

        # Correct IMU errors
        meas_f_ib_b = meas_f_ib_b - est_imu_bias[0:3]
        meas_omega_ib_b = meas_omega_ib_b - est_imu_bias[3:6]

        # Update estimated navigation solution
        est_r_eb_e, est_v_eb_e, est_C_b_e = nav_equations_ecef(
            tor_i, old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e,
            meas_f_ib_b, meas_omega_ib_b
        )

        # Determine whether to update GNSS simulation and run Kalman filter
        if (time - time_last_gnss) >= gnss_config['epoch_interval']:
            gnss_epoch += 1
            tor_s = time - time_last_gnss
            time_last_gnss = time

            # Determine satellite positions and velocities
            sat_r_es_e, sat_v_es_e = satellite_positions_and_velocities(
                time, gnss_config
            )

            # Generate GNSS measurements
            gnss_measurements, no_gnss_meas = generate_gnss_measurements(
                time, sat_r_es_e, sat_v_es_e, true_r_eb_e,
                true_L_b, true_lambda_b, true_v_eb_e,
                gnss_biases, gnss_config
            )

            # Determine GNSS position solution
            gnss_r_eb_e, gnss_v_eb_e, est_clock = gnss_ls_position_velocity(
                gnss_measurements, no_gnss_meas, gnss_r_eb_e, gnss_v_eb_e
            )

            # Run Integration Kalman filter
            est_C_b_e, est_v_eb_e, est_r_eb_e, est_imu_bias, P_matrix = lc_kf_epoch(
                gnss_r_eb_e, gnss_v_eb_e, tor_s, est_C_b_e, est_v_eb_e,
                est_r_eb_e, est_imu_bias, P_matrix, meas_f_ib_b,
                est_L_b, lc_kf_config
            )

            # Generate IMU bias and clock output records
            out_imu_bias_est[gnss_epoch, 0] = time
            out_imu_bias_est[gnss_epoch, 1:7] = est_imu_bias
            out_clock[gnss_epoch, 0] = time
            out_clock[gnss_epoch, 1:3] = est_clock

            # Generate KF uncertainty output record
            out_kf_sd[gnss_epoch, 0] = time
            for i in range(15):
                out_kf_sd[gnss_epoch, i + 1] = np.sqrt(P_matrix[i, i])

        # Convert navigation solution to NED
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n = ecef_to_ned(
            est_r_eb_e, est_v_eb_e, est_C_b_e
        )

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
        old_true_r_eb_e = true_r_eb_e.copy()
        old_true_v_eb_e = true_v_eb_e.copy()
        old_true_C_b_e = true_C_b_e.copy()
        old_est_r_eb_e = est_r_eb_e.copy()
        old_est_v_eb_e = est_v_eb_e.copy()
        old_est_C_b_e = est_C_b_e.copy()

    # Complete progress bar
    print()

    # Trim output arrays to actual GNSS epochs
    out_imu_bias_est = out_imu_bias_est[:gnss_epoch + 1, :]
    out_clock = out_clock[:gnss_epoch + 1, :]
    out_kf_sd = out_kf_sd[:gnss_epoch + 1, :]

    return out_profile, out_errors, out_imu_bias_est, out_clock, out_kf_sd


# Placeholder functions (you'll need to implement these based on your other MATLAB files)

def ned_to_ecef(L_b, lambda_b, h_b, v_eb_n, C_b_n):
    """Convert NED coordinates to ECEF."""
    raise NotImplementedError("This function needs to be implemented")


def satellite_positions_and_velocities(time, gnss_config):
    """Calculate satellite positions and velocities."""
    raise NotImplementedError("This function needs to be implemented")


def initialize_gnss_biases(sat_r_es_e, r_eb_e, L_b, lambda_b, gnss_config):
    """Initialize GNSS biases."""
    raise NotImplementedError("This function needs to be implemented")


def generate_gnss_measurements(time, sat_r_es_e, sat_v_es_e, r_eb_e,
                               L_b, lambda_b, v_eb_e, gnss_biases, gnss_config):
    """Generate GNSS measurements."""
    raise NotImplementedError("This function needs to be implemented")


def gnss_ls_position_velocity(measurements, no_meas, r_eb_e, v_eb_e):
    """GNSS least squares position and velocity solution."""
    raise NotImplementedError("This function needs to be implemented")


def pv_ecef_to_ned(r_eb_e, v_eb_e):
    """Convert ECEF position/velocity to NED."""
    raise NotImplementedError("This function needs to be implemented")


def initialize_ned_attitude(C_b_n, initialization_errors):
    """Initialize NED attitude with errors."""
    raise NotImplementedError("This function needs to be implemented")


def ctm_to_euler(C):
    """Convert coordinate transformation matrix to Euler angles."""
    raise NotImplementedError("This function needs to be implemented")


def calculate_errors_ned(est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n,
                         true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n):
    """Calculate navigation errors in NED frame."""
    raise NotImplementedError("This function needs to be implemented")


def initialize_lc_p_matrix(lc_kf_config):
    """Initialize loosely coupled Kalman filter P matrix."""
    raise NotImplementedError("This function needs to be implemented")


def kinematics_ecef(tor_i, C_b_e, old_C_b_e, v_eb_e, old_v_eb_e, old_r_eb_e):
    """Calculate specific force and angular rate from kinematics."""
    raise NotImplementedError("This function needs to be implemented")


def imu_model(tor_i, f_ib_b, omega_ib_b, imu_errors, quant_residuals):
    """Simulate IMU errors."""
    raise NotImplementedError("This function needs to be implemented")


def nav_equations_ecef(tor_i, r_eb_e, v_eb_e, C_b_e, f_ib_b, omega_ib_b):
    """ECEF navigation equations."""
    raise NotImplementedError("This function needs to be implemented")


def ecef_to_ned(r_eb_e, v_eb_e, C_b_e):
    """Convert ECEF coordinates to NED."""
    raise NotImplementedError("This function needs to be implemented")