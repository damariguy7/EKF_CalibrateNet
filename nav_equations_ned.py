"""
Author: Guy Damari
Date: December 15, 2025
Description:

Nav_equations_NED - Runs precision local-navigation-frame inertial
navigation equations (Note: only the attitude update and specific force
frame transformation phases are precise.)

Inputs:
  tor_i         time interval between epochs (s)
  old_L_b       previous latitude (rad)
  old_lambda_b  previous longitude (rad)
  old_h_b       previous height (m)
  old_C_b_n     previous body-to-NED coordinate transformation matrix
  old_v_eb_n    previous velocity of body frame w.r.t. ECEF frame, resolved
                along north, east, and down (m/s)
  f_ib_b        specific force of body frame w.r.t. ECEF frame, resolved
                along body-frame axes, averaged over time interval (m/s^2)
  omega_ib_b    angular rate of body frame w.r.t. ECEF frame, resolved
                about body-frame axes, averaged over time interval (rad/s)

Outputs:
  L_b           latitude (rad)
  lambda_b      longitude (rad)
  h_b           height (m)
  v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
                north, east, and down (m/s)
  a_eb_n        acceleration of body frame w.r.t. ECEF frame, resolved along
                north, east, and down (m/s^2)
  C_b_n         body-to-NED coordinate transformation matrix
"""

import numpy as np
from typing import Tuple
from radii_of_curvature import radii_of_curvature
from skew_symmetric import skew_symmetric


def nav_equations_ned(tor_i: float, old_L_b: float, old_lambda_b: float,
                      old_h_b: float, old_v_eb_n: np.ndarray,
                      old_C_b_n: np.ndarray, f_ib_b: np.ndarray,
                      omega_ib_b: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Run precision local-navigation-frame inertial navigation equations.

    Parameters
    ----------
    tor_i : float
        Time interval between epochs (s)
    old_L_b : float
        Previous latitude (rad)
    old_lambda_b : float
        Previous longitude (rad)
    old_h_b : float
        Previous height (m)
    old_v_eb_n : np.ndarray
        Previous velocity of body frame w.r.t. ECEF frame, resolved along NED (m/s), shape (3,)
    old_C_b_n : np.ndarray
        Previous body-to-NED coordinate transformation matrix, shape (3, 3)
    f_ib_b : np.ndarray
        Specific force of body frame w.r.t. ECEF frame, resolved along body-frame axes,
        averaged over time interval (m/s^2), shape (3,)
    omega_ib_b : np.ndarray
        Angular rate of body frame w.r.t. ECEF frame, resolved about body-frame axes,
        averaged over time interval (rad/s), shape (3,)

    Returns
    -------
    L_b : float
        Latitude (rad)
    lambda_b : float
        Longitude (rad)
    h_b : float
        Height (m)
    v_eb_n : np.ndarray
        Velocity of body frame w.r.t. ECEF frame, resolved along NED (m/s), shape (3,)
    a_eb_n : np.ndarray
        Acceleration of body frame w.r.t. ECEF frame, resolved along NED (m/s^2), shape (3,)
    C_b_n : np.ndarray
        Body-to-NED coordinate transformation matrix, shape (3, 3)
    """

    # Parameters
    omega_ie = 7.292115e-5  # Earth rotation rate (rad/s)

    # Begins

    # PRELIMINARIES
    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tor_i
    mag_alpha = np.sqrt(alpha_ib_b.T @ alpha_ib_b)
    Alpha_ib_b = skew_symmetric(alpha_ib_b)

    # From (2.123), determine the angular rate of the ECEF frame
    # w.r.t the ECI frame, resolved about NED
    omega_ie_n = omega_ie * np.array([np.cos(old_L_b), 0, -np.sin(old_L_b)])

    # From (5.44), determine the angular rate of the NED frame
    # w.r.t the ECEF frame, resolved about NED
    old_R_N, old_R_E = radii_of_curvature(old_L_b)
    old_omega_en_n = np.array([
        old_v_eb_n[1] / (old_R_E + old_h_b),
        -old_v_eb_n[0] / (old_R_N + old_h_b),
        -old_v_eb_n[1] * np.tan(old_L_b) / (old_R_E + old_h_b)
    ])

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate the average body-to-ECEF-frame coordinate transformation
    # matrix over the update interval using (5.84) and (5.86)
    if mag_alpha > 1e-8:
        ave_C_b_n = old_C_b_n @ (np.eye(3) + (1 - np.cos(mag_alpha)) / mag_alpha ** 2 *
                                 Alpha_ib_b + (1 - np.sin(mag_alpha) / mag_alpha) / mag_alpha ** 2 *
                                 Alpha_ib_b @ Alpha_ib_b) - \
                    0.5 * skew_symmetric(old_omega_en_n + omega_ie_n) @ old_C_b_n
    else:
        ave_C_b_n = old_C_b_n - \
                    0.5 * skew_symmetric(old_omega_en_n + omega_ie_n) @ old_C_b_n

    # Transform specific force to ECEF-frame resolving axes using (5.86)
    f_ib_n = ave_C_b_n @ f_ib_b

    # UPDATE VELOCITY
    # From (5.54),
    v_eb_n = old_v_eb_n + tor_i * (f_ib_n + np.array([0, 0, 9.81]) -
                                   skew_symmetric(2 * omega_ie_n) @ old_v_eb_n)

    a_eb_n = f_ib_n + np.array([0, 0, 9.81])

    # UPDATE CURVILINEAR POSITION
    # Update height using (5.56)
    h_b = old_h_b - 0.5 * tor_i * (old_v_eb_n[2] + v_eb_n[2])

    # Update latitude using (5.56)
    L_b = old_L_b + 0.5 * tor_i * (old_v_eb_n[0] / (old_R_N + old_h_b) +
                                   v_eb_n[0] / (old_R_N + h_b))

    # Calculate meridian and transverse radii of curvature
    R_N, R_E = radii_of_curvature(L_b)

    # Update longitude using (5.56)
    lambda_b = old_lambda_b + 0.5 * tor_i * (old_v_eb_n[1] / ((old_R_E +
                                                               old_h_b) * np.cos(old_L_b)) + v_eb_n[1] / (
                                                         (R_E + h_b) * np.cos(L_b)))

    # ATTITUDE UPDATE
    # From (5.44), determine the angular rate of the NED frame
    # w.r.t the ECEF frame, resolved about NED
    omega_en_n = np.array([
        v_eb_n[1] / (R_E + h_b),
        -v_eb_n[0] / (R_N + h_b),
        -v_eb_n[1] * np.tan(L_b) / (R_E + h_b)
    ])

    # Obtain coordinate transformation matrix from the new attitude w.r.t. an
    # inertial frame to the old using Rodrigues' formula, (5.73)
    if mag_alpha > 1e-8:
        C_new_old = np.eye(3) + np.sin(mag_alpha) / mag_alpha * Alpha_ib_b + \
                    (1 - np.cos(mag_alpha)) / mag_alpha ** 2 * Alpha_ib_b @ Alpha_ib_b
    else:
        C_new_old = np.eye(3) + Alpha_ib_b

    # Update attitude using (5.77)
    C_b_n = (np.eye(3) - skew_symmetric(omega_ie_n + 0.5 * omega_en_n + 0.5 *
                                        old_omega_en_n) * tor_i) @ old_C_b_n @ C_new_old

    # Ends

    return L_b, lambda_b, h_b, v_eb_n, C_b_n