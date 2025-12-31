"""
Author: Guy Damari
Date: December 15, 2025
Description:

CTM_to_Euler - Converts a coordinate transformation matrix to the
corresponding set of Euler angles


Inputs:
  C       coordinate transformation matrix describing transformation from
          beta to alpha

Outputs:
  eul     Euler angles describing rotation from beta to alpha in the
          order roll, pitch, yaw (rad)

"""

import numpy as np


def ctm_to_euler(C: np.ndarray) -> np.ndarray:
    """
    Convert a coordinate transformation matrix to Euler angles.

    Parameters
    ----------
    C : np.ndarray
        Coordinate transformation matrix describing transformation from
        beta to alpha, shape (3, 3)

    Returns
    -------
    eul : np.ndarray
        Euler angles [roll, pitch, yaw] in radians, shape (3,)
    """

    # Begins

    # Calculate Euler angles using (2.23)
    eul = np.zeros(3)
    eul[0] = np.arctan2(C[1, 2], C[2, 2])  # roll
    # eul[1] = -np.arcsin(C[0, 2])  # pitch
    eul[1] = -np.arcsin(np.clip(C[0, 2], -1.0, 1.0))  # pitch
    eul[2] = np.arctan2(C[0, 1], C[0, 0])  # yaw

    # Ends

    return eul
