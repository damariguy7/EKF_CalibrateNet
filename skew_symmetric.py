"""
Author: Guy Damari
Date: December 15, 2025
Description:

Skew_symmetric - Calculates skew-symmetric matrix

Inputs:
  a       3-element vector

Outputs:
  A       3x3 matrix
"""

import numpy as np


def skew_symmetric(a: np.ndarray) -> np.ndarray:
    """
    Calculate skew-symmetric matrix from a 3D vector.

    Parameters
    ----------
    a : np.ndarray
        3-element vector, shape (3,)

    Returns
    -------
    A : np.ndarray
        3x3 skew-symmetric matrix
    """

    # Begins

    A = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

    # Ends

    return A