"""
Initialize_LC_P_matrix - Initializes the loosely coupled INS/GNSS KF
error covariance matrix

Inputs:
  LC_KF_config
    .init_att_unc           Initial attitude uncertainty per axis (rad)
    .init_vel_unc           Initial velocity uncertainty per axis (m/s)
    .init_pos_unc           Initial position uncertainty per axis (m)
    .init_b_a_unc           Initial accel. bias uncertainty (m/s^2)
    .init_b_g_unc           Initial gyro. bias uncertainty (rad/s)

  lat (optional)  Initial latitude (rad) — if provided with h, P[6:9]
  h   (optional)  Initial altitude (m)   — is initialized in radians

Outputs:
  P_matrix              state estimation error covariance matrix
"""

import math
import numpy as np
from typing import Dict, Optional


def initialize_lc_p_matrix(
        LC_KF_config: Dict[str, float],
        lat: Optional[float] = None,
        h: Optional[float] = None,
) -> np.ndarray:
    """
    Initialize the loosely coupled INS/GNSS KF error covariance matrix.

    Parameters
    ----------
    LC_KF_config : dict
        Configuration dictionary containing initial uncertainties
    lat : float, optional
        Initial latitude in radians. When provided together with h,
        P[6:9,6:9] is expressed in radians² to match the radians-based
        position error state used in P_predict.
    h : float, optional
        Initial altitude in metres.

    Returns
    -------
    P_matrix : np.ndarray
        State estimation error covariance matrix, shape (15, 15)
    """

    P_matrix = np.zeros((15, 15))
    P_matrix[0:3, 0:3] = np.eye(3) * LC_KF_config['init_att_unc'] ** 2
    P_matrix[3:6, 3:6] = np.eye(3) * LC_KF_config['init_vel_unc'] ** 2

    if lat is not None and h is not None:
        from radii_of_curvature import radii_of_curvature
        R_N, R_E = radii_of_curvature(lat)
        sig_lat = LC_KF_config['init_pos_unc'] / R_N
        sig_lon = LC_KF_config['init_pos_unc'] / ((R_E + h) * math.cos(lat))
        P_matrix[6, 6] = sig_lat ** 2
        P_matrix[7, 7] = sig_lon ** 2
        P_matrix[8, 8] = LC_KF_config['init_pos_unc'] ** 2
    else:
        P_matrix[6:9, 6:9] = np.eye(3) * LC_KF_config['init_pos_unc'] ** 2

    P_matrix[9:12, 9:12] = np.eye(3) * LC_KF_config['init_b_a_unc'] ** 2
    P_matrix[12:15, 12:15] = np.eye(3) * LC_KF_config['init_b_g_unc'] ** 2

    return P_matrix