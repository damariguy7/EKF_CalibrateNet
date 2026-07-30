"""

Author: Guy Damari
Date: December 15, 2025
Description:
Loosely Coupled Kalman Filter Epoch Update

Implements one cycle of the loosely coupled INS/DVL Kalman filter
plus closed-loop correction of all inertial states.

"""

import numpy as np
from typing import Dict
from gravity_ned import gravity_ned
from skew_symmetric import skew_symmetric
from radii_of_curvature import radii_of_curvature
from ctm_to_euler import ctm_to_euler


def lc_ekf_epoch(
        dvl_v_eb_b: np.ndarray,
        tor_s: float,
        est_C_b_to_n_old: np.ndarray,
        est_v_eb_n_old: np.ndarray,
        est_L_b_old: float,
        est_lambda_b_old: float,
        est_h_b_old: float,
        est_imu_bias_old: np.ndarray,
        P_matrix_old: np.ndarray,
        meas_f_ib_b: np.ndarray,
        meas_omega_ib_b: np.ndarray,
        lc_kf_config: Dict[str, float],
        dnn_corr: np.ndarray = None,
        dnn_corr_scale: float = 1.0
):
    """
    Implements one cycle of the loosely coupled INS/DVL EKF
    plus closed-loop correction of all inertial states.

    Parameters
    ----------

    dvl_v_eb_e : np.ndarray
        DVL estimated NED velocity (m/s), shape (3,)
    tor_s : float
        Propagation interval (s)
    est_C_b_to_n_old : np.ndarray
        Prior estimated body to NED coordinate transformation matrix, shape (3, 3)
    est_v_eb_e_old : np.ndarray
        Prior estimated NED user velocity (m/s), shape (3,)
    est_r_eb_e_old : np.ndarray
        Prior estimated NED user position (m), shape (3,)
    est_imu_bias_old : np.ndarray
        Prior estimated IMU biases (body axes), shape (6,)
        Rows 0-2: accelerometer biases (m/s^2)
        Rows 3-5: gyro biases (rad/s)
    P_matrix_old : np.ndarray
        Previous Kalman filter error covariance matrix, shape (15, 15)
    meas_f_ib_b : np.ndarray
        Measured specific force, shape (3,)
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
    est_C_b_e_new : np.ndarray
        Updated estimated body to NED coordinate transformation matrix, shape (3, 3)
    est_v_eb_e_new : np.ndarray
        Updated estimated NED user velocity (m/s), shape (3,)
    est_r_eb_e_new : np.ndarray
        Updated estimated NED user position (m), shape (3,)
    est_imu_bias_new : np.ndarray
        Updated estimated IMU biases, shape (6,)
        Rows 0-2: estimated accelerometer biases (m/s^2)
        Rows 3-5: estimated gyro biases (rad/s)
    P_matrix_new : np.ndarray
        Updated Kalman filter error covariance matrix, shape (15, 15)
    """

    # # Constants
    # c = 299792458  # Speed of light in m/s
    # omega_ie = 7.292115e-5  # Earth rotation rate in rad/s
    # R_0 = 6378137  # WGS84 Equatorial radius in meters
    # e = 0.0818191908425  # WGS84 eccentricity
    #
    # # # Euler angles from body to NED, in the order roll, pitch, yaw (rad)
    # # euler_b_n = ctm_to_euler(est_C_b_to_n_old)
    # #
    # # roll = float(euler_b_n[0])
    #
    # est_v_eb_n_N_old = est_v_eb_n_old[0]
    # est_v_eb_n_E_old = est_v_eb_n_old[1]
    # est_v_eb_n_D_old = est_v_eb_n_old[2]
    #
    # Calculate radii of curvature
    R_N, R_E = radii_of_curvature(est_L_b_old)
    #
    #
    # # Skew symmetric matrix of Earth rate
    # Omega_ie = skew_symmetric(np.array([0, 0, omega_ie]))
    # Omega_in_n = skew_symmetric(np.array([
    #     (est_v_eb_n_E_old/(np.cos(est_L_b_old)*(R_E + est_h_b_old)) + omega_ie) * np.cos(est_L_b_old),
    #     -est_v_eb_n_N_old/(R_N + est_h_b_old),
    #     -(est_v_eb_n_E_old / (np.cos(est_L_b_old) * (R_E + est_h_b_old)) + omega_ie) * np.sin(est_L_b_old)
    # ]))
    #
    #
    #
    # # ========================================================================
    # # SYSTEM PROPAGATION PHASE
    # # ========================================================================
    #
    # # 1. Determine transition matrix using (14.50) (first-order approximation)
    # Phi_matrix = np.eye(15)
    #
    #
    # # Attitude error propagation
    #
    # # I_3 + F_11 * tor_s
    # Phi_matrix[0:3, 0:3] = Phi_matrix[0:3, 0:3] - Omega_in_n * tor_s
    #
    # # F_12 * tor_s
    # Phi_matrix[0:3, 3:6] = np.array([
    #     [0, -1/(R_E + est_h_b_old), 0],
    #     [1/(R_N + est_h_b_old), 0, 0],
    #     [0, np.tan(est_L_b_old)/(R_E + est_h_b_old), 0]
    # ]) * tor_s
    #
    # # F_13 * tor_s
    # Phi_matrix[0:3, 6:9] = np.array([
    #     [omega_ie * np.sin(est_L_b_old), 0, est_v_eb_n_E_old/(R_E + est_h_b_old)**2],
    #     [0, 0, -est_v_eb_n_N_old/(R_N + est_h_b_old)**2],
    #     [omega_ie * np.cos(est_L_b_old) + est_v_eb_n_E_old/((R_E + est_h_b_old) * np.cos(est_L_b_old)**2), 0 , -est_v_eb_n_E_old*np.tan(est_L_b_old)/((R_E + est_h_b_old)**2)]
    # ]) * tor_s
    #
    # Phi_matrix[0:3, 12:15] = est_C_b_to_n_old * tor_s
    #
    #
    #
    # # Velocity error propagation
    # # F_21  * tor_s
    # Phi_matrix[3:6, 0:3] = - skew_symmetric(est_C_b_to_n_old@meas_f_ib_b) * tor_s
    #
    # # I_3 + F_22 * tor_s
    # Phi_matrix[3:6, 3:6] = Phi_matrix[3:6, 3:6] + np.array([
    #     [est_v_eb_n_D_old/(R_N + est_h_b_old), -(2*est_v_eb_n_E_old*np.tan(est_L_b_old))/(R_E + est_h_b_old) - 2*omega_ie*np.sin(est_L_b_old), est_v_eb_n_N_old/(R_N + est_h_b_old)],
    #     [est_v_eb_n_E_old*np.tan(est_L_b_old)/(R_E + est_h_b_old) + 2*omega_ie*np.sin(est_L_b_old), (est_v_eb_n_N_old*np.tan(est_L_b_old) + est_v_eb_n_D_old)/(R_E + est_h_b_old), est_v_eb_n_E_old/(R_E + est_h_b_old) + 2*omega_ie*np.cos(est_L_b_old)],
    #     [-2*est_v_eb_n_N_old/(R_N + est_h_b_old), -est_v_eb_n_E_old/(R_E + est_h_b_old) - 2*omega_ie*np.cos(est_L_b_old), 0]
    # ]) * tor_s
    #
    # # Calculate surface gravity using the Somigliana model, (2.134)
    # sinsqL = np.sin(est_L_b_old) ** 2
    # g_0 = 9.7803253359 * (1 + 0.001931853 * sinsqL) / np.sqrt(1 - e ** 2 * sinsqL)
    #
    # # Calculate geocentric radius from (2.137)
    # geocentric_radius = (R_0 / np.sqrt(1 - (e * np.sin(est_L_b_old)) ** 2) *
    #                      np.sqrt(np.cos(est_L_b_old) ** 2 +
    #                              (1 - e ** 2) ** 2 * np.sin(est_L_b_old) ** 2))
    #
    # # F_23 * tor_s
    # Phi_matrix[3:6, 6:9] = np.array([
    #     [-(est_v_eb_n_E_old**2)/((np.cos(est_L_b_old)**2)*(R_E + est_h_b_old)) - 2*est_v_eb_n_E_old*omega_ie*np.cos(est_L_b_old), 0, (est_v_eb_n_E_old**2)*np.tan(est_L_b_old)/(R_E + est_h_b_old) - est_v_eb_n_N_old*est_v_eb_n_D_old/(R_N + est_h_b_old)**2],
    #     [est_v_eb_n_N_old*est_v_eb_n_E_old/((np.cos(est_L_b_old)**2)*(R_E + est_h_b_old)) + 2*est_v_eb_n_N_old*omega_ie*np.cos(est_L_b_old) - 2*est_v_eb_n_D_old*omega_ie*np.sin(est_L_b_old), 0, -(est_v_eb_n_N_old*est_v_eb_n_E_old*np.tan(est_L_b_old) + est_v_eb_n_E_old*est_v_eb_n_D_old)/((R_E + est_h_b_old)**2)],
    #     [2*est_v_eb_n_D_old*omega_ie*np.sin(est_L_b_old), 0, (est_v_eb_n_E_old**2)/(R_E + est_h_b_old)**2 + (est_v_eb_n_N_old**2)/(R_N + est_h_b_old)**2 - 2*g_0/geocentric_radius]
    # ]) * tor_s
    #
    # Phi_matrix[3:6, 9:12] = est_C_b_to_n_old * tor_s
    #
    #
    #
    # # Position error propagation
    #
    # # F_32 * tor_s
    # Phi_matrix[6:9, 3:6] = np.array([
    #     [1 / (R_N + est_h_b_old), 0, 0],
    #     [0, 1 / ((R_E + est_h_b_old) * np.cos(est_L_b_old)), 0],
    #     [0, 0, -1]
    # ]) * tor_s
    #
    # # I_3 + F_33 * tor_s
    # Phi_matrix[6:9, 6:9] = Phi_matrix[6:9, 6:9] + np.array([
    #     [0, 0, - est_v_eb_n_N_old / (R_N + est_h_b_old) ** 2],
    #     [est_v_eb_n_E_old * np.sin(est_L_b_old) / (R_E + est_h_b_old) * (np.cos(est_L_b_old) ** 2), 0, -est_v_eb_n_E_old / ((R_E + est_h_b_old) ** 2) * np.cos(est_L_b_old)],
    #     [0, 0, 0]
    # ]) * tor_s
    #
    #
    # # 2. Determine approximate system noise covariance matrix using (14.82)
    # Q_prime_matrix = np.zeros((15, 15))
    # Q_prime_matrix[0:3, 0:3] = np.eye(3) * lc_kf_config['gyro_noise_PSD'] * tor_s
    # Q_prime_matrix[3:6, 3:6] = np.eye(3) * lc_kf_config['accel_noise_PSD'] * tor_s
    # Q_prime_matrix[9:12, 9:12] = np.eye(3) * lc_kf_config['accel_bias_PSD'] * tor_s
    # Q_prime_matrix[12:15, 12:15] = np.eye(3) * lc_kf_config['gyro_bias_PSD'] * tor_s
    #
    # 3. Propagate state estimates using (3.14)
    # Note: all states are zero due to closed-loop correction
    x_est_propagated = np.zeros(15)

    #
    # # 4. Propagate state estimation error covariance matrix using (3.46)
    # P_matrix_propagated = (Phi_matrix @ (P_matrix_old + 0.5 * Q_prime_matrix) @
    #                        Phi_matrix.T + 0.5 * Q_prime_matrix)

    # ========================================================================
    # MEASUREMENT UPDATE PHASE
    # ========================================================================

    # # 5. Set-up measurement matrix using (14.115)
    # H_matrix = np.zeros((6, 15))
    # H_matrix[0:3, 6:9] = -np.eye(3)
    # H_matrix[3:6, 3:6] = -np.eye(3)

    # 5. Set-up measurement matrix using (14.115)
    H_matrix = np.zeros((3, 15))
    # H_matrix[0:3, 0:3] = est_C_b_to_n_old
    # H_matrix[0:3, 3:6] = -est_C_b_to_n_old @ skew_symmetric(est_v_eb_n_old)
    H_matrix[0:3, 0:3] = -est_C_b_to_n_old.T @ skew_symmetric(est_v_eb_n_old)
    H_matrix[0:3, 3:6] = est_C_b_to_n_old.T



    # # # 6. Set-up measurement noise covariance matrix
    # # # Assuming all components of GNSS position and velocity are independent
    # # # and have equal variance
    # # R_matrix = np.zeros((6, 6))
    # # # R_matrix[0:3, 0:3] = np.eye(3) * lc_kf_config['pos_meas_SD'] ** 2
    # # R_matrix[3:6, 3:6] = np.eye(3) * lc_kf_config['vel_meas_SD'] ** 2
    #

    # 6. Set-up measurement noise covariance matrix
    R_matrix = np.zeros((3, 3))
    # R_matrix[0:3, 0:3] = np.zeros((3, 3))
    R_matrix[0:3, 0:3] = np.eye(3) * lc_kf_config['vel_meas_SD'] ** 2

    P_matrix_new, K_matrix = update(P_matrix_old, H_matrix, R_matrix)


    # # 7. Calculate Kalman gain using (3.21)
    # K_matrix = P_matrix_propagated @ H_matrix.T @ np.linalg.inv(
    #     H_matrix @ P_matrix_propagated @ H_matrix.T + R_matrix
    # )

    # 8. Formulate measurement innovations using (14.102)
    # Note: zero lever arm is assumed here
    delta_z = np.zeros(3)

    delta_z[0:3] = est_C_b_to_n_old.T @ est_v_eb_n_old - dvl_v_eb_b

    # 'fuse' mode: fold the DNN velocity correction (NED) into the DVL innovation
    # (rotated to body). The single Kalman update below then applies both the DVL
    # and DNN corrections through the DVL gain, and P shrinks by the DVL amount only
    # (the Joseph update does not depend on delta_z). dnn_corr=None → plain DVL.
    if dnn_corr is not None:
        delta_z[0:3] = delta_z[0:3] - dnn_corr_scale * (est_C_b_to_n_old.T @ dnn_corr)

    # 9. Update state estimates using (3.24)
    x_est_new = x_est_propagated + K_matrix @ delta_z

    # # Apply corrections (attitude via small-angle)
    # R_body2ned = (np.eye(3) - self.vec2skew(dx_tmp[6:9])) @ R_body2ned
    # U, s, VT = svd(R_body2ned)
    # R_body2ned = U @ VT
    #
    # LatLongAlt = LatLongAlt - dx_tmp[0:3].copy()
    # Velocity = Velocity - dx_tmp[3:6]
    # self.ba = self.ba + dx_tmp[9:12]
    # self.bg = self.bg + dx_tmp[12:15]

    # # # # 10. Update state estimation error covariance matrix using (3.25)
    # # P_matrix_new = (np.eye(15) - K_matrix @ H_matrix) @ P_matrix_propagated
    #
    # # Replace with Joseph form for numerical stability:
    # I_KH = np.eye(15) - K_matrix @ H_matrix
    # P_matrix_new = I_KH @ P_matrix_propagated @ I_KH.T + K_matrix @ R_matrix @ K_matrix.T

    # ========================================================================
    # CLOSED-LOOP CORRECTION
    # ========================================================================

    # Correct attitude, velocity, and position using (14.7-9)
    est_C_b_n_new = (np.eye(3) - skew_symmetric(x_est_new[0:3])) @ est_C_b_to_n_old
    U, _, Vt = np.linalg.svd(est_C_b_n_new)
    est_C_b_n_new = U @ Vt
    est_v_eb_n_new = est_v_eb_n_old - x_est_new[3:6]


    # update only est_L_b, est_lambda_b, est_h_b, with nav equations, because they are not observable

    h_b_est_new = est_h_b_old - x_est_new[8]

    # Update latitude using (5.56)
    # x_est_new[6] is position error in meters (North); divide by R_N to convert to radians
    L_b_est_new = est_L_b_old - x_est_new[6] / R_N

    # Update longitude using (5.56)
    # x_est_new[7] is position error in meters (East); divide by (R_E+h)*cos(L) to convert to radians
    lambda_b_est_new = est_lambda_b_old - x_est_new[7] / ((R_E + est_h_b_old) * np.cos(est_L_b_old))


    est_L_b_new = L_b_est_new
    est_lambda_b_new = lambda_b_est_new
    est_h_b_new = h_b_est_new
    # est_r_eb_e_new = est_r_eb_e_old - x_est_new[6:9]

    # Update IMU bias estimates
    est_imu_bias_new = est_imu_bias_old + x_est_new[9:15]

    return est_C_b_n_new, est_v_eb_n_new, est_L_b_new, est_lambda_b_new, est_h_b_new, est_imu_bias_new, P_matrix_new


def lc_ekf_epoch_joint(
        dvl_v_eb_b: np.ndarray,
        dnn_correction: np.ndarray,
        dnn_vel_SD: float,
        tor_s: float,
        est_C_b_to_n_old: np.ndarray,
        est_v_eb_n_old: np.ndarray,
        est_L_b_old: float,
        est_lambda_b_old: float,
        est_h_b_old: float,
        est_imu_bias_old: np.ndarray,
        P_matrix_old: np.ndarray,
        meas_f_ib_b: np.ndarray,
        meas_omega_ib_b: np.ndarray,
        lc_kf_config: Dict[str, float],
        dnn_att_correction: np.ndarray = None,
        dnn_att_SD: float = None
):
    """
    Joint DVL + DNN measurement update — single Joseph-form Kalman update that
    fuses the body-frame DVL velocity with a DNN-provided NED-velocity error
    estimate at the same linearization point (the predicted state).

    The stacked measurement (6 rows, or 9 with the optional attitude block):
        H_combined = [ H_dvl ; H_dnn_ned_vel (; H_dnn_att) ]
        R_combined = blkdiag(R_dvl, dnn_vel_SD^2 I_3 (, dnn_att_SD^2 I_3))
        delta_z    = [ C_b_to_n^T @ v_eb_n - dvl_v_eb_b ; -dnn_correction (; -dnn_att_correction) ]

    Sign convention for the DNN rows: the corrections are the DNN's estimate of
    (true - predicted), so delta_z = nominal - pseudo-measurement = -correction at
    the predicted linearization point. The attitude block (Option B) directly
    observes the attitude-error state (H_att = I_3 on states 0:3); its label is
    delta_eul = -ctm_to_euler(est_C @ true_C^T), matching the closed-loop
    correction est_C_new = (I - skew(x[0:3])) est_C.
    """

    R_N, R_E = radii_of_curvature(est_L_b_old)

    x_est_propagated = np.zeros(15)

    # DVL block
    H_dvl = np.zeros((3, 15))
    H_dvl[0:3, 0:3] = -est_C_b_to_n_old.T @ skew_symmetric(est_v_eb_n_old)
    H_dvl[0:3, 3:6] = est_C_b_to_n_old.T

    # DNN NED-velocity block (identity on velocity states)
    H_dnn = np.zeros((3, 15))
    H_dnn[0:3, 3:6] = np.eye(3)

    use_att = dnn_att_correction is not None
    if use_att:
        # DNN attitude block (identity on attitude-error states 0:3)
        H_att = np.zeros((3, 15))
        H_att[0:3, 0:3] = np.eye(3)
        H_combined = np.vstack([H_dvl, H_dnn, H_att])
        R_combined = np.zeros((9, 9))
        R_combined[0:3, 0:3] = np.eye(3) * lc_kf_config['vel_meas_SD'] ** 2
        R_combined[3:6, 3:6] = np.eye(3) * dnn_vel_SD ** 2
        R_combined[6:9, 6:9] = np.eye(3) * dnn_att_SD ** 2
        delta_z = np.zeros(9)
        delta_z[6:9] = -dnn_att_correction
    else:
        H_combined = np.vstack([H_dvl, H_dnn])
        R_combined = np.zeros((6, 6))
        R_combined[0:3, 0:3] = np.eye(3) * lc_kf_config['vel_meas_SD'] ** 2
        R_combined[3:6, 3:6] = np.eye(3) * dnn_vel_SD ** 2
        delta_z = np.zeros(6)

    P_matrix_new, K_matrix = update(P_matrix_old, H_combined, R_combined)

    delta_z[0:3] = est_C_b_to_n_old.T @ est_v_eb_n_old - dvl_v_eb_b
    delta_z[3:6] = -dnn_correction

    x_est_new = x_est_propagated + K_matrix @ delta_z

    # Closed-loop correction (identical to lc_ekf_epoch)
    est_C_b_n_new = (np.eye(3) - skew_symmetric(x_est_new[0:3])) @ est_C_b_to_n_old
    U, _, Vt = np.linalg.svd(est_C_b_n_new)
    est_C_b_n_new = U @ Vt
    est_v_eb_n_new = est_v_eb_n_old - x_est_new[3:6]

    est_h_b_new      = est_h_b_old      - x_est_new[8]
    est_L_b_new      = est_L_b_old      - x_est_new[6] / R_N
    est_lambda_b_new = est_lambda_b_old - x_est_new[7] / ((R_E + est_h_b_old) * np.cos(est_L_b_old))

    est_imu_bias_new = est_imu_bias_old + x_est_new[9:15]

    return est_C_b_n_new, est_v_eb_n_new, est_L_b_new, est_lambda_b_new, est_h_b_new, est_imu_bias_new, P_matrix_new


def update(P_pred, H, R):
    """
    Measurement update step (Joseph form).

    Args:
        P_pred (np.ndarray): Predicted covariance (15x15).
        H (np.ndarray): Measurement matrix (m x 15).
        dz (np.ndarray): Innovation (m,).
        R (np.ndarray): Measurement noise (m x m).

    Returns:
        tuple:
            dx_update (np.ndarray): State correction (15,).
            P_update (np.ndarray): Updated covariance (15x15).
            K (np.ndarray): Kalman gain (15 x m).
    """
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    # dx_update = K @ dz
    temp = np.eye(15) - K @ H
    P_update = temp @ P_pred @ temp.T + K @ R @ K.T
    P_update = (P_update + P_update.T) / 2
    return P_update, K