"""

Author: Guy Damari
Date: December 15, 2025
Description:
Loosely Coupled Kalman Filter Epoch Update

Implements one cycle of the loosely coupled INS/GNSS Kalman filter
plus closed-loop correction of all inertial states.

"""
from math import atan2, asin

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

    est_pos_old = np.array([est_L_b_old, est_lambda_b_old, est_h_b_old])

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

    # F_32 * tor_s  (position state in radians: δL̇=vN/(RN+h), δλ̇=vE/((RE+h)*cosL), δḣ=-vD)
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

    # F = F_system_matrix(est_pos_old, est_v_eb_n_old, est_C_b_n_old,meas_f_ib_b)
    # Phi_matrix = np.eye(15) + F * tor_s

    G_k = G_k_update(est_C_b_n_old)

    Q_prime_matrix = np.zeros((12, 12))
    Q_prime_matrix[0:3, 0:3] = np.eye(3) * lc_kf_config['accel_noise_PSD']
    Q_prime_matrix[3:6, 3:6] = np.eye(3) * lc_kf_config['gyro_noise_PSD']
    Q_prime_matrix[6:9, 6:9] = np.eye(3) * lc_kf_config['accel_bias_PSD']
    Q_prime_matrix[9:12, 9:12] = np.eye(3) * lc_kf_config['gyro_bias_PSD']
    # Q_prime_matrix = np.zeros((12, 12))
    # Q_prime_matrix[0:3, 0:3] = np.eye(3) * lc_kf_config['gyro_noise_PSD']
    # Q_prime_matrix[3:6, 3:6] = np.eye(3) * lc_kf_config['accel_noise_PSD']
    # Q_prime_matrix[6:9, 6:9] = np.eye(3) * lc_kf_config['accel_bias_PSD']
    # Q_prime_matrix[9:12, 9:12] = np.eye(3) * lc_kf_config['gyro_bias_PSD']

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

    G_att = np.concatenate([zeros_3x3,rotation_matrix, zeros_3x3, zeros_3x3], axis=1)
    G_vel = np.concatenate([rotation_matrix, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    G_pos = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    G_ba = np.concatenate([zeros_3x3, zeros_3x3, identity_3x3, zeros_3x3], axis=1)
    G_bg = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, identity_3x3], axis=1)
    # G_top = np.concatenate([rotation_matrix, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    # G_mid1 = np.concatenate([zeros_3x3, rotation_matrix, zeros_3x3, zeros_3x3], axis=1)
    # G_mid2 = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    # G_bot1 = np.concatenate([zeros_3x3, zeros_3x3, identity_3x3, zeros_3x3], axis=1)
    # G_bot2 = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, identity_3x3], axis=1)
    # G_pos = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    # G_vel = np.concatenate([zeros_3x3, rotation_matrix, zeros_3x3, zeros_3x3], axis=1)
    # G_att = np.concatenate([rotation_matrix, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    # G_ba = np.concatenate([zeros_3x3, zeros_3x3, identity_3x3, zeros_3x3], axis=1)
    # G_bg = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, identity_3x3], axis=1)

    # return np.concatenate([G_mid2, G_mid1, G_top, G_bot1, G_bot2], axis=0)
    # return np.concatenate([G_pos, G_vel, G_att, G_ba, G_bg], axis=0)
    return np.concatenate([G_att, G_vel, G_pos, G_ba, G_bg], axis=0)

def F_system_matrix(location, velocity, rotation_matrix, acceleration):
    """
    Continuous-time error-state system matrix F (15x15).

    Args:
        location (np.ndarray): [lat, lon, alt].
        velocity (np.ndarray): NED velocity [m/s], size (3,).
        rotation_matrix (np.ndarray): C_bn (3x3).
        acceleration (np.ndarray): Specific force [m/s^2], size (3,).

    Returns:
        np.ndarray: F (15x15).
    """
    w_ie = 7.292115e-5
    e = 0.0818191908426
    latitude = location[0]
    altitude = location[2]
    v_north, v_east, v_down = velocity

    R_eS_e = np.sqrt((R_E(latitude) * np.cos(latitude)) ** 2 +
                     ((1 - e * 2) * R_E(latitude) * np.sin(latitude)) * 2)
    zeros_3x3 = np.zeros((3, 3))

    F_EE = -vec2skew(w_ie_n(location) + w_en_n(location, velocity))
    F_Ev = np.array([[0, -1 / (R_E(latitude) + altitude), 0],
                     [1 / (R_N(latitude) + altitude), 0, 0],
                     [0, np.tan(latitude) / (R_E(latitude) + altitude), 0]])
    F_Ep = np.array([
        [w_ie * np.sin(latitude), 0, v_east / (R_E(latitude) + altitude) ** 2],
        [0, 0, -v_north / (R_N(latitude) + altitude) ** 2],
        [w_ie * np.cos(latitude) + v_east / ((R_E(latitude) + altitude) * np.cos(latitude) ** 2),
         0,
         (-v_east * np.tan(latitude)) / (R_E(latitude) + altitude) ** 2]
    ])
    F_vE = -vec2skew(rotation_matrix @ acceleration)
    F_vv = np.array([
        [v_down / (R_N(latitude) + altitude),
         -2 * v_east * np.tan(latitude) / (R_E(latitude) + altitude) - 2 * w_ie * np.sin(latitude),
         v_north / (R_N(latitude) + altitude)],
        [2 * w_ie * np.sin(latitude) + v_east * np.tan(latitude) / (R_E(latitude) + altitude),
         (v_north * np.tan(latitude) + v_down) / (R_E(latitude) + altitude),
         v_east / (R_E(latitude) + altitude) + 2 * w_ie * np.cos(latitude)],
        [-2 * v_north / (R_N(latitude) + altitude),
         -2 * v_east / (R_E(latitude) + altitude) - 2 * w_ie * np.cos(latitude),
         0
         ]
    ])
    sec_lat = 1 / np.cos(latitude)
    F_vp = np.array([
        [-v_east * 2 * sec_lat * 2 / (R_E(latitude) + altitude) - 2 * v_east * w_ie * np.cos(latitude),
         0,
         v_east * 2 * np.tan(latitude) / (R_E(latitude) + altitude) * 2 - v_north * v_down / (
                 R_N(latitude) + altitude) ** 2],
        [v_north * v_east * sec_lat ** 2 / (R_E(latitude) + altitude) +
         2 * v_north * w_ie * np.cos(latitude) - 2 * v_down * w_ie * np.sin(latitude),
         0,
         -(v_north * v_east * np.tan(latitude) + v_east * v_down) / (R_E(latitude) + altitude) ** 2],
        [2 * v_east * w_ie * np.sin(latitude),
         0,
         v_east * 2 / (R_E(latitude) + altitude) * 2 +
         v_north * 2 / (R_N(latitude) + altitude) * 2 -
         2 * standard_gravity(location) / R_eS_e]
    ])
    F_pv = np.array([
        [1 / (R_N(latitude) + altitude), 0, 0],
        [0, 1 / ((R_E(latitude) + altitude) * np.cos(latitude)), 0],
        [0, 0, -1]
    ])
    F_pp = np.array([
        [0, 0, -v_north / (R_N(latitude) + altitude) ** 2],
        [v_east * np.sin(latitude) / ((R_E(latitude) + altitude) * np.cos(latitude) ** 2),
         0,
         v_east / ((R_E(latitude) + altitude) ** 2 * np.cos(latitude))],
        [0, 0, 0]
    ])

    F_top = np.concatenate([F_pp, F_pv, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)
    F_mid1 = np.concatenate([F_vp, F_vv, F_vE, rotation_matrix, zeros_3x3], axis=1)
    F_mid2 = np.concatenate([F_Ep, F_Ev, F_EE, zeros_3x3, rotation_matrix], axis=1)
    F_bot = np.concatenate([zeros_3x3, zeros_3x3, zeros_3x3, zeros_3x3, zeros_3x3], axis=1)

    F = np.concatenate([F_top, F_mid1, F_mid2, F_bot, F_bot], axis=0)
    return F


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


# ----------------------------------------------------------
# Rotation and orientation utilities
# ----------------------------------------------------------
def Rbn(roll, pitch, yaw):
    """
    Compute body-to-NED rotation matrix from Euler angles.

    Args:
        roll (float): Roll angle [rad].
        pitch (float): Pitch angle [rad].
        yaw (float): Yaw angle [rad].

    Returns:
        np.ndarray: 3x3 body-to-NED rotation matrix.
    """
    tmp1 = np.array([[np.cos(yaw), np.sin(yaw), 0],
                     [-np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]])
    tmp2 = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                     [0, 1, 0],
                     [np.sin(pitch), 0, np.cos(pitch)]])
    tmp3 = np.array([[1, 0, 0],
                     [0, np.cos(roll), np.sin(roll)],
                     [0, -np.sin(roll), np.cos(roll)]])
    return (tmp3 @ tmp2 @ tmp1).T

def vec2skew(vector):
    """
    Convert a vector into a skew-symmetric matrix.

    Args:
        vector (np.ndarray): 3-element vector.

    Returns:
        np.ndarray: 3x3 skew-symmetric matrix.
    """
    a1, a2, a3 = vector.flatten()
    return np.array([[0, -a3, a2],
                     [a3, 0, -a1],
                     [-a2, a1, 0]])

def rotation_to_euler(R):
    """
    Convert a rotation matrix to Euler angles.

    Args:
        R (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: [roll, pitch, yaw] in radians.
    """
    roll = atan2(R[2, 1], R[2, 2])
    pitch = -asin(R[2, 0])
    yaw = atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])

# ----------------------------------------------------------
# Earth & gravity models (WGS-84)
# ----------------------------------------------------------
def g_ned(lla):
    """
    Normal gravity on WGS-84 ellipsoid.

    Args:
        lla (np.ndarray): [lat, lon, alt].

    Returns:
        float: Gravity magnitude [m/s^2].
    """
    latitude_rad = lla[0]
    gamma_e = 9.7803267715
    k = 0.001931851353
    e_sq = 0.00669438002290
    sin_phi = np.sin(latitude_rad)
    # return gamma_e * (1 + k * sin_phi * 2) / np.sqrt(1 - e_sq * sin_phi * 2)
    return 9.8106

def standard_gravity(location):
    """
    Standard gravity model (Groves p.47).

    Args:
        location (np.ndarray): [lat, lon, alt].

    Returns:
        float: Gravity [m/s^2].
    """
    latitude = location[0]
    eccentricity = 0.0818191908426
    # return 9.7803253359 * (
    #         1 + (0.001931853 * np.power(sin(latitude), 2)) /
    #         np.sqrt(1 - np.power(eccentricity * sin(latitude), 2))
    # )

    return 9.8106

def R_N(latitude):
    """
    Meridian radius of curvature.

    Args:
        latitude (float): Latitude [rad].

    Returns:
        float: Radius [m].
    """
    R = 6378137
    e = 0.0818191908426
    return (R * (1 - e * 2)) / (1 - (e * 2) * np.sin(latitude) * 2) * (3 / 2)

def R_E(latitude):
    """
    Prime vertical radius of curvature.

    Args:
        latitude (float): Latitude [rad].

    Returns:
        float: Radius [m].
    """
    R = 6378137
    e = 0.0818191908426
    return R / np.sqrt(1 - (e * 2) * np.sin(latitude) * 2)

def w_en_n(lla, v_ned):
    """
    Transport rate (motion-induced rotation in NED).

    Args:
        lla (np.ndarray): [lat, lon, alt].
        v_ned (np.ndarray): Velocity in NED [m/s].

    Returns:
        np.ndarray: [rad/s] 3-vector.
    """
    vN, vE = v_ned[0], v_ned[1]
    latitude, altitude = lla[0], lla[2]
    return np.array([
        vE / (R_E(latitude) + altitude),
        -vN * np.tan(latitude) / (R_N(latitude) + altitude),
        -vE / (R_E(latitude) + altitude)
    ])

def w_ie_n(lla):
    """
    Earth rotation rate in NED.

    Args:
        lla (np.ndarray): [lat, lon, alt].

    Returns:
        np.ndarray: [rad/s] 3-vector.
    """
    Sigma = 7.292115e-5
    latitude = lla[0]
    return np.array([Sigma * np.cos(latitude), 0, -Sigma * np.sin(latitude)])

def w_nb_b(w_ib_b, lla, v_ned, C_bn):
    """
    Body angular rate wrt navigation frame (expressed in body frame).

    Args:
        w_ib_b (np.ndarray): Gyro measurement [rad/s].
        lla (np.ndarray): [lat, lon, alt].
        v_ned (np.ndarray): Velocity in NED [m/s].
        C_bn (np.ndarray): Body-to-NED rotation matrix.

    Returns:
        np.ndarray: [rad/s] 3-vector.
    """
    C_nb = C_bn.T
    return w_ib_b.flatten() - C_nb @ (w_ie_n(lla) + w_en_n(lla, v_ned))