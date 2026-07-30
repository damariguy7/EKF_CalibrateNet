import numpy as np
import matplotlib.pyplot as plt
import navpy


def _sigma_pct(error, time_e, std, time_s):
    """Return % of |error| samples within the ±1σ bound (std interpolated onto time_e)."""
    std_i = np.interp(time_e, time_s, std)
    return 100.0 * np.mean(np.abs(error) <= std_i)


def _improvement_pct(base, dnn):
    """% error reduction of the DNN over the baseline (positive = DNN is better).

    improvement % = 100 * (baseline - dnn) / baseline; returns None if baseline≈0.
    """
    if base is None or dnn is None or abs(base) < 1e-12:
        return None
    return 100.0 * (base - dnn) / base


def _annotate_improvement(ax, bar, base_val):
    """Label a DNN bar with its % improvement vs baseline (green ↑good / red ↓worse)."""
    imp = _improvement_pct(base_val, bar.get_height())
    if imp is None:
        return
    color = 'green' if imp >= 0 else 'firebrick'
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.06,
            f'{imp:+.1f}%', ha='center', va='bottom',
            fontsize=7, fontweight='bold', color=color)


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

        pct = _sigma_pct(pos_errors[:, i], time_errors, pos_std[:, i], time_kf)
        ax.set_ylabel(f'{label} Position Error (m)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Position Error  [{pct:.1f}% within ±1σ]', fontsize=11)
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

        pct = _sigma_pct(vel_errors[:, i], time_errors, vel_std[:, i], time_kf)
        ax.set_ylabel(f'{label} Velocity Error (m/s)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Velocity Error  [{pct:.1f}% within ±1σ]', fontsize=11)
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

        pct = _sigma_pct(att_errors[:, i], time_errors, att_std[:, i], time_kf)
        ax.set_ylabel(f'{label} Error (deg)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Attitude Error  [{pct:.1f}% within ±1σ]', fontsize=11)
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

        pct = _sigma_pct(bias_a_est_ug[:, i], time_bias, bias_a_std_ug[:, i], time_kf)
        ax.set_ylabel(f'{label} Accel Bias (μg)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Accelerometer Bias Estimate  [{pct:.1f}% within ±1σ]', fontsize=11)
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

        pct = _sigma_pct(bias_g_est_dph[:, i], time_bias, bias_g_std_dph[:, i], time_kf)
        ax.set_ylabel(f'{label} Gyro Bias (deg/h)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_title(f'{label} Gyro Bias Estimate  [{pct:.1f}% within ±1σ]', fontsize=11)
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


def plot_errors_comparison(out_errors_base, out_sd_base,
                           out_errors_dnn_np, out_sd_dnn_np,
                           out_errors_dnn_p, out_sd_dnn_p):
    """
    Overlay position, velocity, and attitude errors for three runs:
      - Baseline EKF (grey)
      - DNN, no P update (blue)
      - DNN, with P update (red)

    Returns
    -------
    fig_pos, fig_vel, fig_att : three matplotlib figures
    """
    runs = [
        ('Baseline',         out_errors_base,    out_sd_base,    'dimgrey', '--'),
        ('DNN (no P-update)', out_errors_dnn_np, out_sd_dnn_np,  'steelblue', '-'),
        ('DNN (P-update)',    out_errors_dnn_p,  out_sd_dnn_p,   'crimson',   '-'),
    ]

    deg_to_rad = 0.01745329252

    # ---- Position ----
    fig_pos, axes_pos = plt.subplots(3, 1, figsize=(13, 11))
    fig_pos.suptitle('Position Errors: Baseline vs DNN vs DNN+P-update', fontsize=13, fontweight='bold')
    pos_labels = ['North (m)', 'East (m)', 'Down (m)']
    for i, ax in enumerate(axes_pos):
        for label, errs, sds, color, ls in runs:
            t_e = errs[:, 0]
            t_s = sds[:, 0]
            pct = _sigma_pct(errs[:, 1 + i], t_e, sds[:, 7 + i], t_s)
            ax.plot(t_e, errs[:, 1 + i], color=color, linewidth=1.5, linestyle=ls,
                    label=f'{label} ({pct:.0f}%)')
            ax.fill_between(t_s, -sds[:, 7 + i], sds[:, 7 + i], color=color, alpha=0.12)
        ax.set_ylabel(pos_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---- Velocity ----
    fig_vel, axes_vel = plt.subplots(3, 1, figsize=(13, 11))
    fig_vel.suptitle('Velocity Errors: Baseline vs DNN vs DNN+P-update', fontsize=13, fontweight='bold')
    vel_labels = ['North (m/s)', 'East (m/s)', 'Down (m/s)']
    for i, ax in enumerate(axes_vel):
        for label, errs, sds, color, ls in runs:
            t_e = errs[:, 0]
            t_s = sds[:, 0]
            pct = _sigma_pct(errs[:, 4 + i], t_e, sds[:, 4 + i], t_s)
            ax.plot(t_e, errs[:, 4 + i], color=color, linewidth=1.5, linestyle=ls,
                    label=f'{label} ({pct:.0f}%)')
            ax.fill_between(t_s, -sds[:, 4 + i], sds[:, 4 + i], color=color, alpha=0.12)
        ax.set_ylabel(vel_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---- Attitude ----
    fig_att, axes_att = plt.subplots(3, 1, figsize=(13, 11))
    fig_att.suptitle('Attitude Errors: Baseline vs DNN vs DNN+P-update', fontsize=13, fontweight='bold')
    att_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    for i, ax in enumerate(axes_att):
        for label, errs, sds, color, ls in runs:
            t_e = errs[:, 0]
            t_s = sds[:, 0]
            pct = _sigma_pct(np.rad2deg(errs[:, 7 + i]), t_e, np.rad2deg(sds[:, 1 + i]), t_s)
            ax.plot(t_e, np.rad2deg(errs[:, 7 + i]), color=color, linewidth=1.5, linestyle=ls,
                    label=f'{label} ({pct:.0f}%)')
            ax.fill_between(t_s, -np.rad2deg(sds[:, 1 + i]), np.rad2deg(sds[:, 1 + i]),
                            color=color, alpha=0.12)
        ax.set_ylabel(att_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig_pos, fig_vel, fig_att


def plot_trajectory_2d(in_gt_profile, out_errors):
    """
    Plot 2D top-down GT and EKF-estimated trajectory (North vs East).

    Parameters
    ----------
    in_gt_profile : np.ndarray
        Ground truth profile (cols: time, lat_rad, lon_rad, h, ...)
    out_errors : np.ndarray
        Navigation errors (cols: time, pos_N_err, pos_E_err, pos_D_err, ...)
        pos error = estimated - true, so estimated = true + error

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    lat_ref = np.rad2deg(in_gt_profile[0, 1])
    lon_ref = np.rad2deg(in_gt_profile[0, 2])
    h_ref   = in_gt_profile[0, 3]

    gt_ned = np.array([
        navpy.lla2ned(
            np.rad2deg(in_gt_profile[i, 1]),
            np.rad2deg(in_gt_profile[i, 2]),
            in_gt_profile[i, 3],
            lat_ref, lon_ref, h_ref
        )
        for i in range(len(in_gt_profile))
    ])
    gt_n, gt_e = gt_ned[:, 0], gt_ned[:, 1]

    # Estimated trajectory: GT position + position error
    # out_errors[:, 1] = est_N - true_N, out_errors[:, 2] = est_E - true_E
    # Subtract initial error so both trajectories share the same starting point
    est_n = gt_n + out_errors[:, 1] - out_errors[0, 1]
    est_e = gt_e + out_errors[:, 2] - out_errors[0, 2]
    est_d = gt_e + out_errors[:, 3] - out_errors[0, 3]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_e,  gt_n,  'b-',  linewidth=2,   label='Ground Truth')
    ax.plot(est_e, est_n, 'r--', linewidth=1.5, label='EKF Estimate')
    ax.scatter(gt_e[0],  gt_n[0],  color='green', s=100, zorder=5, label='Start')
    ax.scatter(gt_e[-1], gt_n[-1], color='blue',  s=100, marker='s', zorder=5, label='GT End')
    ax.scatter(est_e[-1], est_n[-1], color='red', s=100, marker='s', zorder=5, label='EKF End')

    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.set_title('Trajectory: Ground Truth vs EKF Estimate (Top-Down)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()

    return fig


def plot_trajectory_2d_comparison(in_gt_profile, out_errors_base,
                                   out_errors_dnn_np, out_errors_dnn_p):
    """2D trajectory: GT, baseline EKF, DNN (no P-update), DNN (with P-update) on one axes."""
    lat_ref = np.rad2deg(in_gt_profile[0, 1])
    lon_ref = np.rad2deg(in_gt_profile[0, 2])
    h_ref   = in_gt_profile[0, 3]

    gt_ned = np.array([
        navpy.lla2ned(
            np.rad2deg(in_gt_profile[i, 1]),
            np.rad2deg(in_gt_profile[i, 2]),
            in_gt_profile[i, 3],
            lat_ref, lon_ref, h_ref
        )
        for i in range(len(in_gt_profile))
    ])
    gt_n, gt_e = gt_ned[:, 0], gt_ned[:, 1]

    def _traj(errs):
        return (gt_n + errs[:, 1] - errs[0, 1],
                gt_e + errs[:, 2] - errs[0, 2])

    base_n,   base_e   = _traj(out_errors_base)
    dnn_np_n, dnn_np_e = _traj(out_errors_dnn_np)
    dnn_p_n,  dnn_p_e  = _traj(out_errors_dnn_p)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_e,     gt_n,     'b-',  linewidth=2,   label='Ground Truth')
    ax.plot(base_e,   base_n,   color='dimgrey',   linewidth=1.5, linestyle='--', label='Baseline EKF')
    ax.plot(dnn_np_e, dnn_np_n, color='steelblue', linewidth=1.5, linestyle='-',  label='DNN (no P-update)')
    ax.plot(dnn_p_e,  dnn_p_n,  color='crimson',   linewidth=1.5, linestyle='-',  label='DNN (P-update)')
    ax.scatter(gt_e[0], gt_n[0], color='black', s=100, zorder=5, label='Start')

    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.set_title('Trajectory: GT vs Baseline vs DNN vs DNN+P-update', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


def plot_pos_vel_two_runs(out_err_base, out_sd_base,
                          out_err_dnn_p=None, out_sd_dnn_p=None,
                          scenario='', arch='lstm',
                          out_err_dnn_np=None, out_sd_dnn_np=None,
                          base_label='Baseline'):
    """
    Overlay position and velocity errors for one, two, or three runs on the same
    axes, with ±1σ shading. base_label customises the baseline-style legend
    entry (e.g. 'Nadav EKF'). DNN runs are drawn only when their args are given.

    Returns
    -------
    fig_pos, fig_vel : two matplotlib figures
    """
    runs = [(base_label, out_err_base, out_sd_base, 'dimgrey', '--')]
    if out_err_dnn_p is not None and out_sd_dnn_p is not None:
        runs.append((f'DNN ({arch}, P-update)', out_err_dnn_p, out_sd_dnn_p, 'crimson', '-'))
    if out_err_dnn_np is not None and out_sd_dnn_np is not None:
        runs.insert(1, (f'DNN ({arch}, no-P)', out_err_dnn_np, out_sd_dnn_np, 'steelblue', '-'))
    title_suffix = f' — {scenario}' if scenario else ''

    # ---- Position ----
    fig_pos, axes_pos = plt.subplots(3, 1, figsize=(13, 11))
    fig_pos.suptitle(f'Position Errors{title_suffix}', fontsize=13, fontweight='bold')
    pos_labels = ['North (m)', 'East (m)', 'Down (m)']
    for i, ax in enumerate(axes_pos):
        for label, errs, sds, color, ls in runs:
            pct = _sigma_pct(errs[:, 1 + i], errs[:, 0], sds[:, 7 + i], sds[:, 0])
            ax.plot(errs[:, 0], errs[:, 1 + i], color=color, linewidth=1.5, linestyle=ls,
                    label=f'{label} ({pct:.0f}%)')
            ax.fill_between(sds[:, 0], -sds[:, 7 + i], sds[:, 7 + i], color=color, alpha=0.12)
        ax.set_ylabel(pos_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---- Velocity ----
    fig_vel, axes_vel = plt.subplots(3, 1, figsize=(13, 11))
    fig_vel.suptitle(f'Velocity Errors{title_suffix}', fontsize=13, fontweight='bold')
    vel_labels = ['North (m/s)', 'East (m/s)', 'Down (m/s)']
    for i, ax in enumerate(axes_vel):
        for label, errs, sds, color, ls in runs:
            pct = _sigma_pct(errs[:, 4 + i], errs[:, 0], sds[:, 4 + i], sds[:, 0])
            ax.plot(errs[:, 0], errs[:, 4 + i], color=color, linewidth=1.5, linestyle=ls,
                    label=f'{label} ({pct:.0f}%)')
            ax.fill_between(sds[:, 0], -sds[:, 4 + i], sds[:, 4 + i], color=color, alpha=0.12)
        ax.set_ylabel(vel_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig_pos, fig_vel


def plot_att_bias_two_runs(out_err_base, out_sd_base, out_bias_base,
                           out_err_dnn_p=None, out_sd_dnn_p=None, out_bias_dnn_p=None,
                           scenario='', arch='lstm',
                           base_label='Baseline'):
    """
    Overlay attitude errors and IMU bias estimates for one or two runs
    with ±1σ shading. base_label customises the baseline-style legend entry
    (e.g. 'Nadav EKF'). DNN run is drawn only when its args are provided.

    Column layout used:
      out_errors  : att at cols 7-9 (rad)
      out_KF_SD   : att_std at cols 1-3, accel_bias_std at 10-12, gyro_bias_std at 13-15
      out_IMU_bias: accel bias at cols 1-3 (m/s²), gyro bias at cols 4-6 (rad/s)

    Returns
    -------
    fig_att, fig_bias_a, fig_bias_g : three matplotlib figures
    """
    micro_g = 9.80665e-6
    deg_to_rad = 0.01745329252

    runs = [(base_label, out_err_base, out_sd_base, out_bias_base, 'dimgrey', '--')]
    if (out_err_dnn_p is not None and out_sd_dnn_p is not None
            and out_bias_dnn_p is not None):
        runs.append((f'DNN ({arch}, P-update)', out_err_dnn_p, out_sd_dnn_p,
                     out_bias_dnn_p, 'crimson', '-'))
    title_suffix = f' — {scenario}' if scenario else ''

    # ---- Attitude ----
    fig_att, axes_att = plt.subplots(3, 1, figsize=(13, 11))
    fig_att.suptitle(f'Attitude Errors{title_suffix}', fontsize=13, fontweight='bold')
    att_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    for i, ax in enumerate(axes_att):
        for label, errs, sds, _, color, ls in runs:
            pct = _sigma_pct(np.rad2deg(errs[:, 7 + i]), errs[:, 0],
                             np.rad2deg(sds[:, 1 + i]), sds[:, 0])
            ax.plot(errs[:, 0], np.rad2deg(errs[:, 7 + i]),
                    color=color, linewidth=1.5, linestyle=ls, label=f'{label} ({pct:.0f}%)')
            ax.fill_between(sds[:, 0],
                            -np.rad2deg(sds[:, 1 + i]), np.rad2deg(sds[:, 1 + i]),
                            color=color, alpha=0.12)
        ax.set_ylabel(att_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---- Accel bias ----
    fig_bias_a, axes_ba = plt.subplots(3, 1, figsize=(13, 11))
    fig_bias_a.suptitle(f'Accel Bias Estimate{title_suffix}', fontsize=13, fontweight='bold')
    bias_a_labels = ['X-axis (μg)', 'Y-axis (μg)', 'Z-axis (μg)']
    for i, ax in enumerate(axes_ba):
        for label, errs, sds, bias, color, ls in runs:
            ax.plot(bias[:, 0], bias[:, 1 + i] / micro_g,
                    color=color, linewidth=1.5, linestyle=ls, label=label)
            ax.fill_between(sds[:, 0],
                            -sds[:, 10 + i] / micro_g, sds[:, 10 + i] / micro_g,
                            color=color, alpha=0.12)
        ax.set_ylabel(bias_a_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # ---- Gyro bias ----
    fig_bias_g, axes_bg = plt.subplots(3, 1, figsize=(13, 11))
    fig_bias_g.suptitle(f'Gyro Bias Estimate{title_suffix}', fontsize=13, fontweight='bold')
    bias_g_labels = ['X-axis (deg/h)', 'Y-axis (deg/h)', 'Z-axis (deg/h)']
    dph = deg_to_rad / 3600
    for i, ax in enumerate(axes_bg):
        for label, errs, sds, bias, color, ls in runs:
            ax.plot(bias[:, 0], bias[:, 4 + i] / dph,
                    color=color, linewidth=1.5, linestyle=ls, label=label)
            ax.fill_between(sds[:, 0],
                            -sds[:, 13 + i] / dph, sds[:, 13 + i] / dph,
                            color=color, alpha=0.12)
        ax.set_ylabel(bias_g_labels[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig_att, fig_bias_a, fig_bias_g


def plot_trajectory_two_runs(in_gt_profile, out_err_base, out_err_dnn_p=None,
                              scenario='', arch='lstm',
                              base_label='Baseline EKF'):
    """
    Plot 2D top-down trajectory for GT and one or two estimates on the same axes.
    base_label customises the baseline-style legend entry (e.g. 'Nadav EKF').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    lat_ref = np.rad2deg(in_gt_profile[0, 1])
    lon_ref = np.rad2deg(in_gt_profile[0, 2])
    h_ref   = in_gt_profile[0, 3]

    gt_ned = np.array([
        navpy.lla2ned(
            np.rad2deg(in_gt_profile[i, 1]),
            np.rad2deg(in_gt_profile[i, 2]),
            in_gt_profile[i, 3],
            lat_ref, lon_ref, h_ref
        )
        for i in range(len(in_gt_profile))
    ])
    gt_n, gt_e = gt_ned[:, 0], gt_ned[:, 1]

    def _traj(errs):
        return (gt_n + errs[:, 1],
                gt_e + errs[:, 2])

    base_n, base_e = _traj(out_err_base)

    title_suffix = f' — {scenario}' if scenario else ''
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'Trajectory{title_suffix}', fontsize=13, fontweight='bold')
    ax.plot(gt_e,   gt_n,   'b-',  linewidth=2,   label='Ground Truth')
    ax.plot(base_e, base_n, color='dimgrey', linewidth=1.5, linestyle='--', label=base_label)
    if out_err_dnn_p is not None:
        dnn_n, dnn_e = _traj(out_err_dnn_p)
        ax.plot(dnn_e, dnn_n, color='crimson', linewidth=1.5, linestyle='-',
                label=f'DNN ({arch}, P-update)')
    ax.scatter(gt_e[0], gt_n[0], color='black', s=100, zorder=5, label='Start')
    ax.set_xlabel('East (m)', fontsize=12)
    ax.set_ylabel('North (m)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


def plot_dnn_sd_over_time(sigma_log, scenario='', arch='lstm', const_sd=None):
    """Plot the applied per-axis DNN measurement SD (learned dnn_vel_SD = sigma)
    over time for one test scenario.

    Parameters
    ----------
    sigma_log : list of (time, sd(3), omega_norm)
        One entry per sequential DNN update (from lc_ins_dvl_real's dnn_sigma_log).
        `sd` is the SD actually used in R_dnn (constant fallback during warmup, then
        the learned per-axis sigma). `omega_norm` = ||omega_ib_b|| (rad/s) for the
        turn-rate overlay.
    const_sd : float or None
        The constant dnn_vel_SD; drawn as a reference line for scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    t     = np.array([e[0] for e in sigma_log], dtype=float)
    sd    = np.array([e[1] for e in sigma_log], dtype=float)          # (N, 3)
    wmag  = np.array([e[2] for e in sigma_log], dtype=float)
    axis_labels = ['σ North', 'σ East', 'σ Down']
    axis_colors = ['tab:blue', 'tab:green', 'tab:red']

    suffix = f' — {scenario}' if scenario else ''
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(3):
        ax.plot(t, sd[:, i], color=axis_colors[i], linewidth=1.4, label=axis_labels[i])
    if const_sd is not None:
        ax.axhline(const_sd, color='gray', ls='--', lw=1.0,
                   label=f'constant dnn_vel_SD = {const_sd:g}')
    ax.set_ylabel('Learned dnn_vel_SD  σ  [m/s]', fontsize=11)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_title(f'Learned dnn_vel_SD (σ) over time  [{arch}]{suffix}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Turn-rate overlay on a secondary axis: does σ rise where the vehicle turns?
    ax_w = ax.twinx()
    ax_w.plot(t, np.rad2deg(wmag), color='darkorange', ls=':', lw=1.0, alpha=0.6,
              label='‖ω‖ (turn rate)')
    ax_w.set_ylabel('‖ω_ib_b‖  [deg/s]', fontsize=10, color='darkorange')
    ax_w.tick_params(axis='y', labelcolor='darkorange')

    # Merge legends from both axes into one box.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_w.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    plt.tight_layout()
    return fig


def plot_rmse_over_time(time, pos_rmse_base, vel_rmse_base,
                        pos_rmse_dnn, vel_rmse_dnn,
                        arch='lstm', n_scenarios=0):
    """
    Plot 3D position and velocity RMSE over time, averaged across all test scenarios.

    At each time step t:
        pos_RMSE(t) = sqrt( mean over scenarios of (eN(t)^2 + eE(t)^2 + eD(t)^2) )
        vel_RMSE(t) = sqrt( mean over scenarios of (evN(t)^2 + evE(t)^2 + evD(t)^2) )

    Parameters
    ----------
    time : np.ndarray  shape (T,)
    pos_rmse_base, vel_rmse_base : np.ndarray  shape (T,)  — baseline EKF
    pos_rmse_dnn,  vel_rmse_dnn  : np.ndarray  shape (T,)  — DNN with P-update
    arch : str   DNN architecture label for legend
    n_scenarios : int  number of test scenarios (for title)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    title = f'RMSE over Test Set ({n_scenarios} scenarios)' if n_scenarios else 'RMSE over Test Set'
    dnn_label = f'DNN ({arch}, P-update)'

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    ax_pos.plot(time, pos_rmse_base, color='steelblue', linewidth=1.8, label='Baseline EKF')
    ax_pos.plot(time, pos_rmse_dnn,  color='crimson',   linewidth=1.8, label=dnn_label)
    ax_pos.set_ylabel('Position RMSE [m]', fontsize=11)
    ax_pos.legend(fontsize=10)
    ax_pos.grid(True, alpha=0.3)

    ax_vel.plot(time, vel_rmse_base, color='steelblue', linewidth=1.8, label='Baseline EKF')
    ax_vel.plot(time, vel_rmse_dnn,  color='crimson',   linewidth=1.8, label=dnn_label)
    ax_vel.set_ylabel('Velocity RMSE [m/s]', fontsize=11)
    ax_vel.set_xlabel('Time [s]', fontsize=11)
    ax_vel.legend(fontsize=10)
    ax_vel.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prmse_vrmse_per_trajectory(scenario_names, prmse_base, vrmse_base,
                                    prmse_dnn_p, vrmse_dnn_p, arch='lstm',
                                    prmse_dnn_np=None, vrmse_dnn_np=None):
    """
    Two separate bar-chart figures showing per-trajectory scalar PRMSE and VRMSE.

    For each trajectory i:
        PRMSE_i = sqrt( mean_over_time( eN^2 + eE^2 + eD^2 ) )
        VRMSE_i = sqrt( mean_over_time( evN^2 + evE^2 + evD^2 ) )

    Parameters
    ----------
    scenario_names        : list of str
    prmse_base, vrmse_base: list of float  — baseline EKF scalar per trajectory
    prmse_dnn_p, vrmse_dnn_p : list of float  — DNN with P-update
    arch                  : str  DNN architecture label for legend
    prmse_dnn_np, vrmse_dnn_np : list of float or None — DNN without P-update (optional)

    Returns
    -------
    fig_prmse, fig_vrmse : matplotlib.figure.Figure
    """
    n = len(scenario_names)
    x = np.arange(n)
    has_np = prmse_dnn_np is not None and vrmse_dnn_np is not None
    n_bars = 3 if has_np else 2
    width = 0.22 if has_np else 0.35
    short_names = [s.split('_s')[-1] if '_s' in s else s for s in scenario_names]

    def _add_labels(ax, bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    def _make_bar_fig(base_vals, dnn_np_vals, dnn_p_vals, ylabel, title):
        fig, ax = plt.subplots(figsize=(max(7, n * 1.3), 5))
        if has_np:
            offsets = [-width, 0, width]
        else:
            offsets = [-width / 2, width / 2]
        bars_base = ax.bar(x + offsets[0], base_vals, width,
                           color='dimgrey',   label='Baseline EKF')
        _add_labels(ax, bars_base)
        if has_np:
            bars_np = ax.bar(x + offsets[1], dnn_np_vals, width,
                             color='steelblue', label=f'DNN ({arch}, no-P)')
            _add_labels(ax, bars_np)
            bars_p = ax.bar(x + offsets[2], dnn_p_vals, width,
                            color='crimson',   label=f'DNN ({arch}, P-update)')
            _add_labels(ax, bars_p)
            for bar, b in zip(bars_np, base_vals):
                _annotate_improvement(ax, bar, b)
            for bar, b in zip(bars_p, base_vals):
                _annotate_improvement(ax, bar, b)
        else:
            bars_p = ax.bar(x + offsets[1], dnn_p_vals, width,
                            color='crimson',   label=f'DNN ({arch}, P-update)')
            _add_labels(ax, bars_p)
            for bar, b in zip(bars_p, base_vals):
                _annotate_improvement(ax, bar, b)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        return fig

    fig_prmse = _make_bar_fig(prmse_base, prmse_dnn_np, prmse_dnn_p,
                               'Position RMSE [m]',
                               f'PRMSE per Trajectory ({n} scenarios)')
    fig_vrmse = _make_bar_fig(vrmse_base, vrmse_dnn_np, vrmse_dnn_p,
                               'Velocity RMSE [m/s]',
                               f'VRMSE per Trajectory ({n} scenarios)')
    return fig_prmse, fig_vrmse


def plot_prmse_vrmse_per_axis(out_err_base, out_err_dnn, scenario='', arch='lstm'):
    """
    Two bar-chart figures showing per-axis (North/East/Down) RMSE for a single
    trajectory: baseline EKF vs DNN, in the same style as
    plot_prmse_vrmse_per_trajectory.

        PRMSE_axis = sqrt( mean_over_time( e_axis^2 ) )   for axis in {N, E, D}
        VRMSE_axis = sqrt( mean_over_time( ev_axis^2 ) )

    Parameters
    ----------
    out_err_base : ndarray  baseline error array; cols [1:4]=pos N,E,D, [4:7]=vel N,E,D
    out_err_dnn  : ndarray  DNN error array, same layout
    scenario     : str   scenario name for the title
    arch         : str   DNN architecture label for the legend

    Returns
    -------
    fig_prmse, fig_vrmse : matplotlib.figure.Figure
    """
    axes_labels = ['North', 'East', 'Down']
    x = np.arange(3)
    width = 0.35

    def _rmse_per_axis(err, cols):
        return [float(np.sqrt(np.mean(err[:, c] ** 2))) for c in cols]

    pos_base = _rmse_per_axis(out_err_base, [1, 2, 3])
    pos_dnn  = _rmse_per_axis(out_err_dnn,  [1, 2, 3])
    vel_base = _rmse_per_axis(out_err_base, [4, 5, 6])
    vel_dnn  = _rmse_per_axis(out_err_dnn,  [4, 5, 6])

    def _add_labels(ax, bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    def _make_bar_fig(base_vals, dnn_vals, ylabel, title):
        fig, ax = plt.subplots(figsize=(7, 5))
        bars_base = ax.bar(x - width / 2, base_vals, width,
                           color='dimgrey', label='Baseline EKF')
        _add_labels(ax, bars_base)
        bars_dnn = ax.bar(x + width / 2, dnn_vals, width,
                          color='crimson', label=f'DNN ({arch}, P-update)')
        _add_labels(ax, bars_dnn)
        for bar, b in zip(bars_dnn, base_vals):
            _annotate_improvement(ax, bar, b)
        ax.set_xticks(x)
        ax.set_xticklabels(axes_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        return fig

    suffix = f' — {scenario}' if scenario else ''
    fig_prmse = _make_bar_fig(pos_base, pos_dnn, 'Position RMSE [m]',
                              f'PRMSE per Axis{suffix}')
    fig_vrmse = _make_bar_fig(vel_base, vel_dnn, 'Velocity RMSE [m/s]',
                              f'VRMSE per Axis{suffix}')
    return fig_prmse, fig_vrmse