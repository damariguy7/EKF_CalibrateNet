

"""
Read_profile - inputs a motion profile in the following .csv format

Column 0: time (sec)
Column 1: latitude (rad)
Column 2: longitude (rad)
Column 3: height (m)
Column 4: north velocity (m/s)
Column 5: east velocity (m/s)
Column 6: down velocity (m/s)
Column 7: roll angle of body w.r.t NED (rad)
Column 8: pitch angle of body w.r.t NED (rad)
Column 9: yaw angle of body w.r.t NED (rad)

Inputs:
  filename     Name of file to read

Outputs:
  in_profile   Array of data from the file
  no_epochs    Number of epochs of data in the file
  ok           Indicates file has the expected number of columns

"""

import numpy as np
from typing import Tuple
def read_profile(filename: str) -> Tuple[np.ndarray, int]:
    """
    Read a motion profile from a CSV file.

    Parameters
    ----------
    filename : str
        Name of file to read

    Returns
    -------
    in_profile : np.ndarray
        Array of data from the file (no_epochs x 10)
    no_epochs : int
        Number of epochs of data in the file
    ok : bool
        Indicates file has the expected number of columns
    """

    # Begins

    # Parameters
    deg_to_rad = 0.01745329252

    # Read in the profile in .csv format
    # Skip first row (header), read from row 1 onwards (MATLAB: csvread(filename, 1, 0))
    in_profile = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Determine size of file
    no_epochs, no_columns = in_profile.shape

    # Ends

    return in_profile, no_epochs