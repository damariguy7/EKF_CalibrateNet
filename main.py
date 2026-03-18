# Author: Guy Damari
# Date: December 15, 2025


import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from lc_ins_dvl_real import lc_ins_dvl_real
from lc_ins_dvl_sim import lc_ins_dvl_sim
from lc_ins_dvl_sim_nadav import lc_ins_dvl_sim_nadav
from plot_errors import plot_errors_with_std, plot_results, plot_trajectory_2d
from dnn_vel_compensator import (train_vel_compensator, save_compensator,
                                  load_compensator)


# Constants
deg_to_rad = 0.01745329252
rad_to_deg = 1 / deg_to_rad
micro_g_to_meters_per_second_squared = 9.80665e-6


# ============================================================================
# DVL CONFIG
# ============================================================================
#Interval between DVL epochs (s)
DVL_config = {
        'epoch_interval': 1.002506
}

# ============================================================================
# KALMAN FILTER CONFIGURATION
# ============================================================================

LC_KF_config = {

    # Initial attitude uncertainty per axis (deg, converted to rad)
    # 2 deg too tight: P[att] shrinks faster than error converges after first DVL update
    'init_att_unc': np.deg2rad(2.0),

    # Initial velocity uncertainty per axis (m/s)
    # 3.0 too tight: draw (~3 m/s) + gravity mis-projection (g*sin(att)*dt ~ 0.34 m/s) = 3.34 m/s
    # bound grows in quadrature (3.02 m/s) while error grows additively (3.34 m/s) -> outside
    'init_vel_unc': 3.0,

    # Initial position uncertainty per axis (m)
    # 1.0 m too tight: init vel transient (3 m/s x 20s) causes ~12 m position error in first 50s
    'init_pos_unc': 10.0,

    # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
    'init_b_a_unc': 9.80665e-4,

    # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
    'init_b_g_unc': 4.848e-6,

    'gyro_noise_PSD': 1.0e-6,
    'accel_noise_PSD': 9.617e-9,

    # Bias PSDs: small but non-zero
    # 1e-6/2e-9 -> P[bias] grows too fast -> biases absorb yaw error -> diverge
    # 1e-14 -> P[bias] collapses -> unstable cross-covariances
    # 1e-10/1e-12 -> P[bias] stays bounded but doesn't grow
    # 1e-10 -> P[acc_bias] collapses -> bounds shrink while estimate drifts -> inconsistent
    # 1e-8 -> keeps bounds wide enough to contain the estimate
    'accel_bias_PSD': 1.0e-6,
    'gyro_bias_PSD': 1.0e-8,

    # vel_meas_SD: 2.0 -> K[att] too small -> yaw can't be corrected -> diverges
    # 0.5 -> K[att] larger -> yaw correctable; K[bias] controlled by small bias_PSDs above
    'vel_meas_SD': 0.5
}


def main(config):
    # Set random seed for reproducibility
    np.random.seed(2)

    data_path = config['data_path']

    if config['data_type'] == "real":
        real_data_trajectory_index = config['real_data_trajectory_index']
        gt_pd = pd.read_csv(os.path.join(data_path, f'real_data', f'GT_trajectory{real_data_trajectory_index}.csv'),
                            header=0, names=None)
        imu_pd = pd.read_csv(os.path.join(data_path, f'real_data', f'IMU_trajectory{real_data_trajectory_index}.csv'),
                             header=0, names=None)
        dvl_pd = pd.read_csv(os.path.join(data_path, f'real_data', f'DVL_trajectory{real_data_trajectory_index}.csv'),
                             header=0, names=None)
        in_gt_profile = np.array(gt_pd.iloc[:, 0:10])
        in_imu_profile = np.array(imu_pd.iloc[:, 0:7])
        in_imu_profile[:, 1:7] *= 100
        in_dvl_profile = np.array(dvl_pd.iloc[:, 0:4])

        # Determine size of file
        no_epochs, no_columns = in_imu_profile.shape

        # Run loosely coupled NED Inertial navigation and DVL integrated navigation system
        out_profile, out_errors, out_IMU_bias_est, out_KF_SD = lc_ins_dvl_real(
            in_imu_profile,
            in_dvl_profile,
            in_gt_profile,
            no_epochs,
            DVL_config,
            LC_KF_config,
        )

    elif config['data_type'] == "sim":
        simulated_data_file_name = config['simulated_data_file_name']
        gt_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'GT_{simulated_data_file_name}.csv'),
                            header=0, names=None)
        imu_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'IMU_{simulated_data_file_name}.csv'),
                             header=0, names=None)
        dvl_pd = pd.read_csv(os.path.join(data_path, f'simulated_data', f'DVL_{simulated_data_file_name}.csv'),
                             header=0, names=None)
        # Skip first row: first simulated sample is corrupted (initialization shock)
        in_gt_profile = np.array(gt_pd.iloc[:, 0:10])
        in_imu_profile = np.array(imu_pd.iloc[:, 0:7])
        in_dvl_profile = np.array(dvl_pd.iloc[:, 0:4])

        # Trim leading seconds if requested
        trim_start = config.get('trim_start_seconds', 0)
        if trim_start > 0:
            mask_gt = in_gt_profile[:, 0] >= trim_start
            mask_imu = in_imu_profile[:, 0] >= trim_start
            mask_dvl = in_dvl_profile[:, 0] >= trim_start
            in_gt_profile = in_gt_profile[mask_gt]
            in_imu_profile = in_imu_profile[mask_imu]
            in_dvl_profile = in_dvl_profile[mask_dvl]

        # Determine size of file
        no_epochs, no_columns = in_imu_profile.shape

        # Run Guy's EKF
        out_profile, out_errors, out_IMU_bias_est, out_KF_SD = lc_ins_dvl_sim(
            in_imu_profile,
            in_dvl_profile,
            in_gt_profile,
            no_epochs,
            DVL_config,
            LC_KF_config,
        )

        # Run Nadav's EKF
        # out_profile_n, out_errors_n, out_IMU_bias_est_n, out_KF_SD_n = lc_ins_dvl_sim_nadav(
        #     in_imu_profile,
        #     in_dvl_profile,
        #     in_gt_profile,
        #     no_epochs,
        #     DVL_config,
        #     LC_KF_config,
        # )


    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    scenario  = config.get('simulated_data_file_name', config.get('real_data_trajectory_index', 'unknown'))

    def _save_plots(out_err, out_sd, out_bias, gt_profile, label):
        """Save all 6 plots for one EKF run into its own sub-folder."""
        plots_dir = os.path.join(config['data_path'], 'plots', label)
        os.makedirs(plots_dir, exist_ok=True)

        fig_pos, fig_vel, fig_att, fig_bias_a, fig_bias_g = plot_errors_with_std(
            out_err, out_sd, out_bias)
        fig_traj = plot_trajectory_2d(gt_profile, out_err)

        fig_pos.savefig(   os.path.join(plots_dir, 'position_errors.png'), dpi=150, bbox_inches='tight')
        fig_vel.savefig(   os.path.join(plots_dir, 'velocity_errors.png'), dpi=150, bbox_inches='tight')
        fig_att.savefig(   os.path.join(plots_dir, 'attitude_errors.png'), dpi=150, bbox_inches='tight')
        fig_bias_a.savefig(os.path.join(plots_dir, 'accel_bias.png'),      dpi=150, bbox_inches='tight')
        fig_bias_g.savefig(os.path.join(plots_dir, 'gyro_bias.png'),       dpi=150, bbox_inches='tight')
        fig_traj.savefig(  os.path.join(plots_dir, 'trajectory.png'),      dpi=150, bbox_inches='tight')

        with open(os.path.join(plots_dir, 'params_summary.txt'), 'w') as f:
            f.write(f"Run: {label}\n{'='*50}\n\nDVL CONFIG\n{'-'*30}\n")
            for k, v in DVL_config.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nKALMAN FILTER CONFIG\n{'-'*30}\n")
            for k, v in LC_KF_config.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nRUN CONFIG\n{'-'*30}\n  data_type: {config['data_type']}\n  scenario: {scenario}\n")

        plt.close('all')
        print(f"Plots saved to: {plots_dir}")

    if config['data_type'] == "sim":
        _save_plots(out_errors,   out_KF_SD,   out_IMU_bias_est,   in_gt_profile,
                    f"sim_{scenario}_{timestamp}_guy")
        # _save_plots(out_errors_n, out_KF_SD_n, out_IMU_bias_est_n, in_gt_profile,
        #             f"sim_{scenario}_{timestamp}_nadav")
    else:
        _save_plots(out_errors, out_KF_SD, out_IMU_bias_est, in_gt_profile,
                    f"real_{scenario}_{timestamp}")

    # =========================================================================
    # TRAIN DNN VELOCITY COMPENSATOR
    # =========================================================================
    if config['train_model'] and config['data_type'] == 'sim':
        print('Collecting training data...')
        _, _, _, _, dvl_features, dvl_labels = lc_ins_dvl_sim(
            in_imu_profile, in_dvl_profile, in_gt_profile,
            no_epochs, DVL_config, LC_KF_config,
            collect_data=True,
        )

        dnn_config = config.get('dnn_config', {})
        print('Training DNN velocity compensator...')
        model, norm_stats = train_vel_compensator(dvl_features, dvl_labels, dnn_config)

        model_path = os.path.join(config['data_path'], 'trained_model', 'vel_compensator.pt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_compensator(model, norm_stats, dnn_config, model_path)

    # =========================================================================
    # TEST DNN VELOCITY COMPENSATOR
    # =========================================================================
    if config['test_model'] and config['data_type'] == 'sim':
        model_path = os.path.join(config['data_path'], 'trained_model', 'vel_compensator.pt')
        print(f'Loading model from: {model_path}')
        compensator = load_compensator(model_path)

        print('Running EKF + DNN compensator...')
        out_profile_dnn, out_errors_dnn, out_IMU_bias_est_dnn, out_KF_SD_dnn = lc_ins_dvl_sim(
            in_imu_profile, in_dvl_profile, in_gt_profile,
            no_epochs, DVL_config, LC_KF_config,
            compensator=compensator,
        )

        _save_plots(out_errors_dnn, out_KF_SD_dnn, out_IMU_bias_est_dnn, in_gt_profile,
                    f"sim_{scenario}_{timestamp}_dnn")


if __name__ == '__main__':

    # User-defined configuration (can be read from a config file or command-line arguments)
    user_config = {
        'data_path': "C:\\Users\\damar\\PycharmProjects\\EKF_CalibrateNet",
        'data_type': 'sim',  # "sim" or "real"
        'simulated_data_file_name': 'long_turn', #'static', 'straight', 'lawn_mower_50', 'lawn_mower600' 'long_turn', 'turn_n_straight'
        'real_data_trajectory_index': '1',  # 1-13
        'trim_start_seconds': 50,  # cut first N seconds (0 = no trim)
        'train_model': False,
        'test_model': False,
        'test_baseline_model': False,
        'trained_model_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work\\trained_model",
        'dnn_config': {
            'arch':        'lstm',  # 'lstm' | 'gru' | 'mlp'
            'window_size':  10,     # DVL epochs per input window (~10s)
            'hidden_size':  64,
            'num_layers':    2,
            'batch_size':   32,
            'epochs':      100,
            'lr':          1e-3,
        },
    }


    # Merge default and user configurations
    config = {**user_config}

    main(config)

    # plt.show()


