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

Outputs:
  P_matrix              state estimation error covariance matrix
"""

import numpy as np
from typing import Dict


def initialize_lc_p_matrix(LC_KF_config: Dict[str, float]) -> np.ndarray:
    """
    Initialize the loosely coupled INS/GNSS KF error covariance matrix.

    Parameters
    ----------
    LC_KF_config : dict
        Configuration dictionary containing initial uncertainties

    Returns
    -------
    P_matrix : np.ndarray
        State estimation error covariance matrix, shape (15, 15)
    """

    # Begins

    # Initialize error covariance matrix
    P_matrix = np.zeros((15, 15))
    P_matrix[0:3, 0:3] = np.eye(3) * LC_KF_config['init_att_unc'] ** 2
    P_matrix[3:6, 3:6] = np.eye(3) * LC_KF_config['init_vel_unc'] ** 2
    P_matrix[6:9, 6:9] = np.eye(3) * LC_KF_config['init_pos_unc'] ** 2
    P_matrix[9:12, 9:12] = np.eye(3) * LC_KF_config['init_b_a_unc'] ** 2
    P_matrix[12:15, 12:15] = np.eye(3) * LC_KF_config['init_b_g_unc'] ** 2

    # Ends

    return P_matrix