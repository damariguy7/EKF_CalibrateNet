"""

Author: Guy Damari
Date: December 15, 2025
Description:
Loosely Coupled Kalman Filter Epoch Update

Implements one cycle of the loosely coupled INS/GNSS Kalman filter
plus closed-loop correction of all inertial states.

"""

import numpy as np
from typing import Dict
from gravity_ned import gravity_ned
from skew_symmetric import skew_symmetric
from radii_of_curvature import radii_of_curvature
from ctm_to_euler import ctm_to_euler


def P_predict(
        tor_s: float,
        est_C_b_n_old: np.ndarray,
        est_v_eb_n_old: np.ndarray,
        est_L_b_old: float,
        est_lambda_b_old: float,
        est_h_b_old: float,
        P_matrix_old: np.ndarray,
        meas_f_ib_b: np.ndarray,
        meas_omega_ib_b: np.ndarray,
        lc_kf_config: Dict[str, float]
):
    """
    Implements one cycle of the loosely coupled INS/DVL EKF
    plus closed-loop correction of all inertial states.

    Parameters
    ----------

    tor_s : float
        Propagation interval (s)
    est_C_b_e_old : np.ndarray
        Prior estimated body to ECEF coordinate transformation matrix, shape (3, 3)
    est_v_eb_e_old : np.ndarray
        Prior estimated ECEF user velocity (m/s), shape (3,)
    est_r_eb_e_old : np.ndarray
        Prior estimated ECEF user position (m), shape (3,)
    est_imu_bias_old : np.ndarray
        Prior estimated IMU biases (body axes), shape (6,)
        Rows 0-2: accelerometer biases (m/s^2)
        Rows 3-5: gyro biases (rad/s)
    P_matrix_old : np.ndarray
        Previous Kalman filter error covariance matrix, shape (15, 15)
    est_L_b_old : float
        Previous latitude solution (rad)
    lc_kf_config : dict
        Configuration dictionary containing:
        - 'gyro_noise_PSD': Gyro noise PSD (rad^2/s)
        - 'accel_noise_PSD': Accelerometer noise PSD (m^2 s^-3)
        - 'accel_bias_PSD': Accelerometer bias random walk PSD (m^2 s^-5)
        - 'gyro_bias_PSD': Gyro bias random walk PSD (rad^2 s^-3)
        - 'pos_meas_SD': Position measurement noise SD per axis (m)
        - 'vel_meas_SD': Velocity measurement noise SD per axis (m/s)

    Returns
    -------
    P_matrix_new : np.ndarray
        Updated Kalman filter error covariance matrix, shape (15, 15)
    """

    # Constants
    c = 299792458  # Speed of light in m/s
    omega_ie = 7.292115e-5  # Earth rotation rate in rad/s
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity

    # # Euler angles from body to NED, in the order roll, pitch, yaw (rad)
    # euler_b_n = ctm_to_euler(est_C_b_n_old)
    #
    # roll = float(euler_b_n[0])

    est_v_eb_n_N_old = est_v_eb_n_old[0]
    est_v_eb_n_E_old = est_v_eb_n_old[1]
    est_v_eb_n_D_old = est_v_eb_n_old[2]

    # Calculate radii of curvature
    R_N, R_E = radii_of_curvature(est_L_b_old)


    # Skew symmetric matrix of Earth rate
    Omega_ie = skew_symmetric(np.array([0, 0, omega_ie]))
    Omega_in_n = skew_symmetric(np.array([
        (est_v_eb_n_E_old/(np.cos(est_L_b_old)*(R_E + est_h_b_old)) + omega_ie) * np.cos(est_L_b_old),
        -est_v_eb_n_N_old/(R_N + est_h_b_old),
        -(est_v_eb_n_E_old / (np.cos(est_L_b_old) * (R_E + est_h_b_old)) + omega_ie) * np.sin(est_L_b_old)
    ]))



    # ========================================================================
    # SYSTEM PROPAGATION PHASE
    # ========================================================================

    # 1. Determine transition matrix using (14.50) (first-order approximation)
    Phi_matrix = np.eye(15)


    # Attitude error propagation

    # I_3 + F_11 * tor_s
    Phi_matrix[0:3, 0:3] = Phi_matrix[0:3, 0:3] - Omega_in_n * tor_s

    # F_12 * tor_s
    Phi_matrix[0:3, 3:6] = np.array([
        [0, -1/(R_E + est_h_b_old), 0],
        [1/(R_N + est_h_b_old), 0, 0],
        [0, np.tan(est_L_b_old)/(R_E + est_h_b_old), 0]
    ]) * tor_s

    # F_13 * tor_s
    Phi_matrix[0:3, 6:9] = np.array([
        [omega_ie * np.sin(est_L_b_old), 0, est_v_eb_n_E_old/(R_E + est_h_b_old)**2],
        [0, 0, -est_v_eb_n_N_old/(R_N + est_h_b_old)**2],
        [omega_ie * np.cos(est_L_b_old) + est_v_eb_n_E_old/((R_E + est_h_b_old) * np.cos(est_L_b_old)**2), 0 , -est_v_eb_n_E_old*np.tan(est_L_b_old)/((R_E + est_h_b_old)**2)]
    ]) * tor_s

    Phi_matrix[0:3, 12:15] = est_C_b_n_old * tor_s



    # Velocity error propagation
    # F_21  * tor_s
    Phi_matrix[3:6, 0:3] = - skew_symmetric(est_C_b_n_old@meas_f_ib_b) * tor_s

    # I_3 + F_22 * tor_s
    Phi_matrix[3:6, 3:6] = Phi_matrix[3:6, 3:6] + np.array([
        [est_v_eb_n_D_old/(R_N + est_h_b_old), -(2*est_v_eb_n_E_old*np.tan(est_L_b_old))/(R_E + est_h_b_old) - 2*omega_ie*np.sin(est_L_b_old), est_v_eb_n_N_old/(R_N + est_h_b_old)],
        [est_v_eb_n_E_old*np.tan(est_L_b_old)/(R_E + est_h_b_old) + 2*omega_ie*np.sin(est_L_b_old), (est_v_eb_n_N_old*np.tan(est_L_b_old) + est_v_eb_n_D_old)/(R_E + est_h_b_old), est_v_eb_n_E_old/(R_E + est_h_b_old) + 2*omega_ie*np.cos(est_L_b_old)],
        [-2*est_v_eb_n_N_old/(R_N + est_h_b_old), -2*est_v_eb_n_E_old/(R_E + est_h_b_old) - 2*omega_ie*np.cos(est_L_b_old), 0]
    ]) * tor_s

    # Calculate surface gravity using the Somigliana model, (2.134)
    sinsqL = np.sin(est_L_b_old) ** 2
    g_0 = 9.7803253359 * (1 + 0.001931853 * sinsqL) / np.sqrt(1 - e ** 2 * sinsqL)

    # Calculate geocentric radius from (2.137)
    geocentric_radius = (R_0 / np.sqrt(1 - (e * np.sin(est_L_b_old)) ** 2) *
                         np.sqrt(np.cos(est_L_b_old) ** 2 +
                                 (1 - e ** 2) ** 2 * np.sin(est_L_b_old) ** 2))

    # F_23 * tor_s
    Phi_matrix[3:6, 6:9] = np.array([
        [-(est_v_eb_n_E_old**2)/((np.cos(est_L_b_old)**2)*(R_E + est_h_b_old)) - 2*est_v_eb_n_E_old*omega_ie*np.cos(est_L_b_old), 0, (est_v_eb_n_E_old**2)*np.tan(est_L_b_old)/((R_E + est_h_b_old)**2) - est_v_eb_n_N_old*est_v_eb_n_D_old/((R_N + est_h_b_old)**2)],
        [est_v_eb_n_N_old*est_v_eb_n_E_old/((np.cos(est_L_b_old)**2)*(R_E + est_h_b_old)) + 2*est_v_eb_n_N_old*omega_ie*np.cos(est_L_b_old) - 2*est_v_eb_n_D_old*omega_ie*np.sin(est_L_b_old), 0, -(est_v_eb_n_N_old*est_v_eb_n_E_old*np.tan(est_L_b_old) + est_v_eb_n_E_old*est_v_eb_n_D_old)/((R_E + est_h_b_old)**2)],
        [2*est_v_eb_n_E_old*omega_ie*np.sin(est_L_b_old), 0, (est_v_eb_n_E_old**2)/((R_E + est_h_b_old)**2) + (est_v_eb_n_N_old**2)/((R_N + est_h_b_old)**2) - 2*g_0/geocentric_radius]
    ]) * tor_s

    Phi_matrix[3:6, 9:12] = est_C_b_n_old * tor_s

    # Position error propagation

    # F_32 * tor_s
    Phi_matrix[6:9, 3:6] = np.array([
        [1 / (R_N + est_h_b_old), 0, 0],
        [0, 1 / ((R_E + est_h_b_old) * np.cos(est_L_b_old)), 0],
        [0, 0, -1]
    ]) * tor_s

    # I_3 + F_33 * tor_s
    Phi_matrix[6:9, 6:9] = Phi_matrix[6:9, 6:9] + np.array([
        [0, 0, - est_v_eb_n_N_old / ((R_N + est_h_b_old) ** 2)],
        [est_v_eb_n_E_old * np.sin(est_L_b_old) / ((R_E + est_h_b_old) * (np.cos(est_L_b_old) ** 2)), 0, -est_v_eb_n_E_old / (((R_E + est_h_b_old) ** 2) * np.cos(est_L_b_old))],
        [0, 0, 0]
    ]) * tor_s

    G_k = G_k_update(est_C_b_n_old)

    Q_prime_matrix = np.zeros((12, 12))
    Q_prime_matrix[0:3, 0:3] = np.eye(3) * lc_kf_config['gyro_noise_PSD']
    Q_prime_matrix[3:6, 3:6] = np.eye(3) * lc_kf_config['accel_noise_PSD']
    Q_prime_matrix[6:9, 6:9] = np.eye(3) * lc_kf_config['accel_bias_PSD']
    Q_prime_matrix[9:12, 9:12] = np.eye(3) * lc_kf_config['gyro_bias_PSD']

    Q_k = G_k @ Q_prime_matrix @ G_k.T * tor_s

    P_matrix_new = predict(Phi_matrix, P_matrix_old, Q_k).copy()

    return P_matrix_new




def G_k_update(rotation_matrix):
    """
    Shaping matrix G for random-walk driven blocks (maps 12-dim w to 15-dim state).

    Args:
        rotation_matrix (np.ndarray): C_bn (3x3).

    Returns:
        np.ndarray: G (15x12).
    """
    zeros_3x3 = np.zeros((3, 3))
    identity_3x3 = np.eye(3)

    G_top = np.concatenate([rotation_matrix, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    G_mid1 = np.concatenate([zeros_3x3, rotation_matrix, zeros_3x3, zeros_3x3], axis=1)
    G_mid2 = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    G_bot1 = np.concatenate([zeros_3x3, zeros_3x3, identity_3x3, zeros_3x3], axis=1)
    G_bot2 = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, identity_3x3], axis=1)
    # G_top = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    # G_mid1 = np.concatenate([rotation_matrix, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    # G_mid2 = np.concatenate([zeros_3x3, rotation_matrix, zeros_3x3, zeros_3x3], axis=1)
    # G_bot1 = np.concatenate([zeros_3x3, zeros_3x3, identity_3x3, zeros_3x3], axis=1)
    # G_bot2 = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, identity_3x3], axis=1)

    # return np.concatenate([G_mid2, G_mid1, G_top, G_bot1, G_bot2], axis=0)
    return np.concatenate([G_top, G_mid1, G_mid2, G_bot1, G_bot2], axis=0)


def predict(phi, P, Q):
    """
    Covariance prediction step.

    Args:
        phi (np.ndarray): Discrete state transition (15x15).
        P (np.ndarray): Prior covariance (15x15).
        Q (np.ndarray): Discrete process covariance (15x15).

    Returns:
        np.ndarray: Predicted covariance (15x15).
    """
    P_pred = phi @ P @ phi.T + Q
    return (P_pred + P_pred.T) / 2