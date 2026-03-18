"""
Calculate_errors_NED - Calculates the position, velocity, and attitude
errors of a NED navigation solution.

Inputs:
  est_L_b       latitude solution (rad)
  est_lambda_b  longitude solution (rad)
  est_h_b       height solution (m)
  est_v_eb_n    velocity solution of body frame w.r.t. ECEF frame,
                resolved along north, east, and down (m/s)
  est_C_b_to_n     body-to-NED coordinate transformation matrix solution
  true_L_b      true latitude (rad)
  true_lambda_b true longitude (rad)
  true_h_b      true height (m)
  true_v_eb_n   true velocity of body frame w.r.t. ECEF frame, resolved
                along north, east, and down (m/s)
  true_C_b_to_n    true body-to-NED coordinate transformation matrix

Outputs:
  delta_r_eb_n     position error resolved along NED (m)
  delta_v_eb_n     velocity error resolved along NED (m/s)
  delta_eul_nb_n   attitude error as NED Euler angles (rad)
                   These are expressed about north, east, and down

"""

import numpy as np
from typing import Tuple
from radii_of_curvature import radii_of_curvature
from ctm_to_euler import ctm_to_euler


def calculate_errors_ned(
        est_L_b: float,
        est_lambda_b: float,
        est_h_b: float,
        est_v_eb_n: np.ndarray,
        est_C_b_to_n: np.ndarray,
        true_L_b: float,
        true_lambda_b: float,
        true_h_b: float,
        true_v_eb_n: np.ndarray,
        true_C_b_to_n: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the position, velocity, and attitude errors of a NED navigation solution.

    Parameters
    ----------
    est_L_b : float
        Latitude solution (rad)
    est_lambda_b : float
        Longitude solution (rad)
    est_h_b : float
        Height solution (m)
    est_v_eb_n : np.ndarray
        Velocity solution of body frame w.r.t. ECEF frame,
        resolved along north, east, and down (m/s), shape (3,)
    est_C_b_to_n : np.ndarray
        Body-to-NED coordinate transformation matrix solution, shape (3, 3)
    true_L_b : float
        True latitude (rad)
    true_lambda_b : float
        True longitude (rad)
    true_h_b : float
        True height (m)
    true_v_eb_n : np.ndarray
        True velocity of body frame w.r.t. ECEF frame, resolved
        along north, east, and down (m/s), shape (3,)
    true_C_b_to_n : np.ndarray
        True body-to-NED coordinate transformation matrix, shape (3, 3)

    Returns
    -------
    delta_r_eb_n : np.ndarray
        Position error resolved along NED (m), shape (3,)
    delta_v_eb_n : np.ndarray
        Velocity error resolved along NED (m/s), shape (3,)
    delta_eul_nb_n : np.ndarray
        Attitude error as NED Euler angles (rad), shape (3,)
        These are expressed about north, east, and down
    """

    # Begins

    # Position error calculation, using (2.119) - convert lat/lon difference to meters
    R_N, R_E = radii_of_curvature(true_L_b)
    delta_r_eb_n = np.zeros(3)
    delta_r_eb_n[0] = (est_L_b - true_L_b) * R_N                                     # North (m)
    delta_r_eb_n[1] = (est_lambda_b - true_lambda_b) * (R_E + true_h_b) * np.cos(true_L_b)  # East (m)
    delta_r_eb_n[2] = est_h_b - true_h_b                                              # Down (m)

    # Velocity error calculation
    delta_v_eb_n = est_v_eb_n - true_v_eb_n

    # Attitude error calculation, using (5.109) and (5.111)
    delta_C_b_to_n = est_C_b_to_n @ true_C_b_to_n.T
    delta_eul_nb_n = -ctm_to_euler(delta_C_b_to_n) #Note that maybe we want to show delta_eul_bn_n instead

    # Ends

    return delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n
