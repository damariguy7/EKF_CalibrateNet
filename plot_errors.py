import numpy as np
import matplotlib.pyplot as plt


def plot_errors_with_std(out_errors, out_KF_SD, out_IMU_bias_est):
    """
    Plot navigation errors with ±1-sigma standard deviation bounds.

    Parameters
    ----------
    out_errors : np.ndarray
        Navigation errors (columns: time, pos_N, pos_E, pos_D, vel_N, vel_E, vel_D, att_X, att_Y, att_Z)
    out_KF_SD : np.ndarray
        Kalman filter standard deviations (columns: time, ..., att_X_std, att_Y_std, att_Z_std,
                                          vel_X_std, vel_Y_std, vel_Z_std, pos_X_std, pos_Y_std, pos_Z_std,
                                          bias_a_X_std, bias_a_Y_std, bias_a_Z_std,
                                          bias_g_X_std, bias_g_Y_std, bias_g_Z_std)
    out_IMU_bias_est : np.ndarray
        IMU bias estimates (columns: time, bias_a_X, bias_a_Y, bias_a_Z, bias_g_X, bias_g_Y, bias_g_Z)

    Returns
    -------
    tuple
        (fig_pos, fig_vel, fig_att, fig_bias_a, fig_bias_g) - 5 matplotlib figures
    """

    # Extract time vectors
    time_errors = out_errors[:, 0]
    time_kf = out_KF_SD[:, 0]

    # Constants for unit conversions
    micro_g_to_meters_per_second_squared = 9.80665e-6
    deg_to_rad = 0.01745329252

    # ========== POSITION ERRORS WITH STD ==========
    fig_pos, axes_pos = plt.subplots(3, 1, figsize=(12, 10))
    fig_pos.suptitle('Position Errors with ±1σ Standard Deviation', fontsize=14, fontweight='bold')

    # Position errors (columns 1, 2, 3 for North, East, Down)
    pos_errors = out_errors[:, 1:4]
    # Position STD (columns 7, 8, 9)
    pos_std = out_KF_SD[:, 7:10]

    pos_labels = ['North', 'East', 'Down']
    colors = ['r', 'g', 'b']

    for i, (ax, label, color) in enumerate(zip(axes_pos, pos_labels, colors)):
        # Plot error
        ax.plot(time_errors, pos_errors[:, i], color=color, linewidth=1.5, label=f'{label} Error')

        # Plot ±1σ bounds
        ax.plot(time_kf, pos_std[:, i], '--', color=color, alpha=0.7, linewidth=1, label='+1σ')
        ax.plot(time_kf, -pos_std[:, i], '--', color=color, alpha=0.7, linewidth=1, label='-1σ')

        # Fill between ±σ bounds
        ax.fill_between(time_kf, -pos_std[:, i], pos_std[:, i],
                        color=color, alpha=0.2, label='±1σ bounds')

        ax.set_ylabel(f'{label} Position Error (m)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Position Error', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ========== VELOCITY ERRORS WITH STD ==========
    fig_vel, axes_vel = plt.subplots(3, 1, figsize=(12, 10))
    fig_vel.suptitle('Velocity Errors with ±1σ Standard Deviation', fontsize=14, fontweight='bold')

    # Velocity errors (columns 4, 5, 6 for North, East, Down)
    vel_errors = out_errors[:, 4:7]
    # Velocity STD (columns 4, 5, 6)
    vel_std = out_KF_SD[:, 4:7]

    vel_labels = ['North', 'East', 'Down']

    for i, (ax, label, color) in enumerate(zip(axes_vel, vel_labels, colors)):
        # Plot error
        ax.plot(time_errors, vel_errors[:, i], color=color, linewidth=1.5, label=f'{label} Error')

        # Plot ±1σ bounds
        ax.plot(time_kf, vel_std[:, i], '--', color=color, alpha=0.7, linewidth=1, label='+1σ')
        ax.plot(time_kf, -vel_std[:, i], '--', color=color, alpha=0.7, linewidth=1, label='-1σ')

        # Fill between ±σ bounds
        ax.fill_between(time_kf, -vel_std[:, i], vel_std[:, i],
                        color=color, alpha=0.2, label='±1σ bounds')

        ax.set_ylabel(f'{label} Velocity Error (m/s)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Velocity Error', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ========== ATTITUDE ERRORS WITH STD ==========
    fig_att, axes_att = plt.subplots(3, 1, figsize=(12, 10))
    fig_att.suptitle('Attitude Errors with ±1σ Standard Deviation', fontsize=14, fontweight='bold')

    # Attitude errors (columns 7, 8, 9 for Roll, Pitch, Yaw) - convert to degrees
    att_errors = np.rad2deg(out_errors[:, 7:10])
    # Attitude STD (columns 1, 2, 3)
    att_std = np.rad2deg(out_KF_SD[:, 1:4])

    att_labels = ['Roll', 'Pitch', 'Yaw']

    for i, (ax, label, color) in enumerate(zip(axes_att, att_labels, colors)):
        # Plot error
        ax.plot(time_errors, att_errors[:, i], color=color, linewidth=1.5, label=f'{label} Error')

        # Plot ±1σ bounds
        ax.plot(time_kf, att_std[:, i], '--', color=color, alpha=0.7, linewidth=1, label='+1σ')
        ax.plot(time_kf, -att_std[:, i], '--', color=color, alpha=0.7, linewidth=1, label='-1σ')

        # Fill between ±σ bounds
        ax.fill_between(time_kf, -att_std[:, i], att_std[:, i],
                        color=color, alpha=0.2, label='±1σ bounds')

        ax.set_ylabel(f'{label} Error (deg)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Attitude Error', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ========== ACCELEROMETER BIAS ESTIMATES WITH STD ==========
    fig_bias_a, axes_bias_a = plt.subplots(3, 1, figsize=(12, 10))
    fig_bias_a.suptitle('Accelerometer Bias Estimates with ±1σ Standard Deviation',
                        fontsize=14, fontweight='bold')

    # Extract time and bias estimates
    time_bias = out_IMU_bias_est[:, 0]
    bias_a_est = out_IMU_bias_est[:, 1:4]  # Columns 1, 2, 3 for X, Y, Z accel biases

    # Convert to micro-g for better readability
    bias_a_est_ug = bias_a_est / micro_g_to_meters_per_second_squared

    # Accelerometer bias STD (columns 10, 11, 12)
    bias_a_std = out_KF_SD[:, 10:13]
    bias_a_std_ug = bias_a_std / micro_g_to_meters_per_second_squared

    bias_a_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for i, (ax, label, color) in enumerate(zip(axes_bias_a, bias_a_labels, colors)):
        # Plot bias estimate
        ax.plot(time_bias, bias_a_est_ug[:, i], color=color, linewidth=1.5,
                label=f'{label} Bias Estimate')

        # Plot ±1σ bounds
        ax.plot(time_kf, bias_a_std_ug[:, i], '--', color=color, alpha=0.7,
                linewidth=1, label='+1σ')
        ax.plot(time_kf, -bias_a_std_ug[:, i], '--', color=color, alpha=0.7,
                linewidth=1, label='-1σ')

        # Fill between ±σ bounds
        ax.fill_between(time_kf, -bias_a_std_ug[:, i], bias_a_std_ug[:, i],
                        color=color, alpha=0.2, label='±1σ bounds')

        ax.set_ylabel(f'{label} Accel Bias (μg)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Accelerometer Bias Estimate', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ========== GYRO BIAS ESTIMATES WITH STD ==========
    fig_bias_g, axes_bias_g = plt.subplots(3, 1, figsize=(12, 10))
    fig_bias_g.suptitle('Gyro Bias Estimates with ±1σ Standard Deviation',
                        fontsize=14, fontweight='bold')

    # Gyro bias estimates (columns 4, 5, 6 for X, Y, Z gyro biases)
    bias_g_est = out_IMU_bias_est[:, 4:7]

    # Convert to deg/hour for better readability
    bias_g_est_dph = bias_g_est / (deg_to_rad / 3600)

    # Gyro bias STD (columns 13, 14, 15)
    bias_g_std = out_KF_SD[:, 13:16]
    bias_g_std_dph = bias_g_std / (deg_to_rad / 3600)

    bias_g_labels = ['X-axis', 'Y-axis', 'Z-axis']

    for i, (ax, label, color) in enumerate(zip(axes_bias_g, bias_g_labels, colors)):
        # Plot bias estimate
        ax.plot(time_bias, bias_g_est_dph[:, i], color=color, linewidth=1.5,
                label=f'{label} Bias Estimate')

        # Plot ±1σ bounds
        ax.plot(time_kf, bias_g_std_dph[:, i], '--', color=color, alpha=0.7,
                linewidth=1, label='+1σ')
        ax.plot(time_kf, -bias_g_std_dph[:, i], '--', color=color, alpha=0.7,
                linewidth=1, label='-1σ')

        # Fill between ±σ bounds
        ax.fill_between(time_kf, -bias_g_std_dph[:, i], bias_g_std_dph[:, i],
                        color=color, alpha=0.2, label='±1σ bounds')

        ax.set_ylabel(f'{label} Gyro Bias (deg/h)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Gyro Bias Estimate', fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Always return all 5 figures
    return fig_pos, fig_vel, fig_att, fig_bias_a, fig_bias_g


def plot_results(out_errors, out_KF_SD):
    """
    Plot the simulation results.

    Parameters
    ----------
    in_profile : np.ndarray
        Input motion profile
    out_errors : np.ndarray
        Navigation errors
    out_KF_SD : np.ndarray
        Kalman filter standard deviations
    """

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    # fig.suptitle('INS/GNSS Demo 3 Results', fontsize=16, fontweight='bold')

    time = out_errors[:, 0]

    # Position errors
    axes[0, 0].plot(time, out_errors[:, 1], 'r-', label='North')
    axes[0, 0].plot(time, out_errors[:, 2], 'g-', label='East')
    axes[0, 0].plot(time, out_errors[:, 3], 'b-', label='Down')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position Error (m)')
    axes[0, 0].set_title('Position Errors')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Velocity errors
    axes[0, 1].plot(time, out_errors[:, 4], 'r-', label='North')
    axes[0, 1].plot(time, out_errors[:, 5], 'g-', label='East')
    axes[0, 1].plot(time, out_errors[:, 6], 'b-', label='Down')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity Error (m/s)')
    axes[0, 1].set_title('Velocity Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Attitude errors
    axes[1, 0].plot(time, np.rad2deg(out_errors[:, 7]), 'r-', label='Roll')
    axes[1, 0].plot(time, np.rad2deg(out_errors[:, 8]), 'g-', label='Pitch')
    axes[1, 0].plot(time, np.rad2deg(out_errors[:, 9]), 'b-', label='Yaw')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Attitude Error (deg)')
    axes[1, 0].set_title('Attitude Errors')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Position uncertainties
    kf_time = out_KF_SD[:, 0]
    axes[1, 1].plot(kf_time, out_KF_SD[:, 7], 'r-', label='N')
    axes[1, 1].plot(kf_time, out_KF_SD[:, 8], 'g-', label='E')
    axes[1, 1].plot(kf_time, out_KF_SD[:, 9], 'b-', label='D')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position Uncertainty (m)')
    axes[1, 1].set_title('Position Uncertainties (1-sigma)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Velocity uncertainties
    axes[2, 0].plot(kf_time, out_KF_SD[:, 4], 'r-', label='N')
    axes[2, 0].plot(kf_time, out_KF_SD[:, 5], 'g-', label='E')
    axes[2, 0].plot(kf_time, out_KF_SD[:, 6], 'b-', label='D')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Velocity Uncertainty (m/s)')
    axes[2, 0].set_title('Velocity Uncertainties (1-sigma)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Attitude uncertainties
    axes[2, 1].plot(kf_time, np.rad2deg(out_KF_SD[:, 1]), 'r-', label='Roll')
    axes[2, 1].plot(kf_time, np.rad2deg(out_KF_SD[:, 2]), 'g-', label='Pitch')
    axes[2, 1].plot(kf_time, np.rad2deg(out_KF_SD[:, 3]), 'b-', label='Yaw')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Attitude Uncertainty (deg)')
    axes[2, 1].set_title('Attitude Uncertainties (1-sigma)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    # plt.savefig('INS_GNSS_Demo_3_Results.png', dpi=300, bbox_inches='tight')
    # print("Plot saved as: INS_GNSS_Demo_3_Results.png")
    # plt.show()