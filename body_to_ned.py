
import numpy as np
from euler_to_ctm import euler_to_ctm

def body_to_ned(vec_b, eul_ned_to_body_rad):
    """
    Transform velocity from body frame to NED frame.

    Parameters
    ----------
    v_b : np.ndarray
        Velocity in body frame, shape (no_epochs, 3)
    eul_ned_to_body_rad : np.ndarray
        Euler angles from NED to body frame in radians, shape (no_epochs, 3)


    Returns
    -------
    vec_n : np.ndarray
        Velocity in NED frame, shape (no_epochs, 3)
    """
    vec_n = np.zeros_like(vec_b)

    C_ned_to_body = euler_to_ctm(eul_ned_to_body_rad)
    vec_n = (C_ned_to_body.T @ vec_b.T).T

    return vec_n