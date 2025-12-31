"""
Author: Guy Damari
Date: December 15, 2025
Description:

Initialize_NED_attitude - Initializes the attitude solution by adding
errors to the truth.


Inputs:
  C_b_n         true body-to-NED coordinate transformation matrix
  initialization_errors
    .delta_eul_nb_n   attitude error as NED Euler angles (rad)

Outputs:
  est_C_b_n     body-to-NED coordinate transformation matrix solution

"""

import numpy as np
from typing import Dict
from euler_to_ctm import euler_to_ctm


def initialize_ned_attitude(C_b_n: np.ndarray,
                            initialization_errors: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Initialize the attitude solution by adding errors to the truth.

    Parameters
    ----------
    C_b_n : np.ndarray
        True body-to-NED coordinate transformation matrix, shape (3, 3)
    initialization_errors : dict
        Dictionary containing:
        - 'delta_eul_nb_n': attitude error as NED Euler angles (rad), shape (3,)

    Returns
    -------
    est_C_b_n : np.ndarray
        Body-to-NED coordinate transformation matrix solution, shape (3, 3)
    """

    # Begins

    # Attitude initialization, using (5.109) and (5.111)
    delta_C_b_n = euler_to_ctm(-initialization_errors['delta_eul_nb_n'])
    est_C_b_n = delta_C_b_n @ C_b_n

    # Ends

    return est_C_b_n
