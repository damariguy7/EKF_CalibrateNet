# Author: Guy Damari
# Date: December 15, 2025


import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from lc_ins_dvl_real import lc_ins_dvl_real
from lc_ins_dvl_real_nadav import lc_ins_dvl_real_nadav
from lc_ins_dvl_sim import lc_ins_dvl_sim
from lc_ins_dvl_sim_nadav import lc_ins_dvl_sim_nadav
from plot_errors import plot_errors_with_std, plot_results, plot_trajectory_2d, plot_trajectory_2d_comparison, plot_errors_comparison, plot_rmse_over_time, plot_pos_vel_two_runs, plot_att_bias_two_runs, plot_trajectory_two_runs, plot_prmse_vrmse_per_trajectory
from dnn_vel_compensator import (train_vel_compensator, save_compensator,
                                  load_compensator)


# Constants
deg_to_rad = 0.01745329252
rad_to_deg = 1 / deg_to_rad
micro_g_to_meters_per_second_squared = 9.80665e-6


def _dnn_config_token(dnn_config, dnn_vel_sd=None):
    """Return a directory-friendly DNN config string used for plot folder names.

    Format: <arch>_w<W>_h<H>_l<L>_e<E>[_sd<dnn_vel_SD>][_<tag>]

    `dnn_vel_sd` is an EKF-side parameter (lives in LC_KF_config_*, not
    dnn_config) but it changes the run's behaviour at the DVL update step,
    so we include it in the folder name so different SD values don't
    overwrite each other. Encoded as e.g. 0.5 → 'sd0p5' (filesystem-safe).
    Pass None to omit it.
    """
    arch = dnn_config.get('arch', 'lstm')
    parts = [
        arch,
        f"w{dnn_config.get('window_size', 10)}",
        f"h{dnn_config.get('hidden_size', 64)}",
        f"l{dnn_config.get('num_layers', 2)}",
        f"e{dnn_config.get('epochs', 100)}",
    ]
    if dnn_vel_sd is not None:
        parts.append(f"sd{str(dnn_vel_sd).replace('.', 'p')}")
    tag = dnn_config.get('tag', '')
    if tag:
        parts.append(str(tag))
    return '_'.join(parts)


def _build_model_filename(data_type, dnn_config, sim_trajectory_name):
    """Build the .pth filename for the DNN compensator from training context.

    sim:  vel_compensator_sim_<arch>_<traj>_w<W>_h<H>_l<L>_e<E>[_<tag>].pth
    real: vel_compensator_real_<arch>_w<W>_h<H>_l<L>_e<E>[_<tag>].pth

    Underscores inside the trajectory name are replaced with dashes so
    underscore stays a clean field separator. Empty/missing tag drops the
    trailing _<tag> segment.
    """
    arch = dnn_config.get('arch', 'lstm')
    parts = ['vel_compensator', data_type, arch]
    if data_type == 'sim':
        traj_token = (sim_trajectory_name or 'unknown').replace('_', '-')
        parts.append(traj_token)
    parts += [
        f"w{dnn_config.get('window_size', 10)}",
        f"h{dnn_config.get('hidden_size', 64)}",
        f"l{dnn_config.get('num_layers', 2)}",
        f"e{dnn_config.get('epochs', 100)}",
    ]
    tag = dnn_config.get('tag', '')
    if tag:
        parts.append(str(tag))
    return '_'.join(parts) + '.pth'


def _discover_sim_scenarios(data_dir, trajectory_name):
    """Glob GT_{trajectory_name}_*.csv in data_dir and return scenario base names."""
    matches = glob.glob(os.path.join(data_dir, f'GT_{trajectory_name}_*.csv'))
    names = sorted(os.path.basename(p)[len('GT_'):-len('.csv')] for p in matches)
    if not names:
        print(f'[discover] No files matched GT_{trajectory_name}_*.csv in {data_dir}')
    else:
        print(f'[discover] Found {len(names)} sim scenarios for trajectory "{trajectory_name}"')
    return names


def _split_files(files, seed):
    """Shuffle and split a file list 60/20/20 → (train, val, test) deterministically.

    A single-file list is treated as a test-only request: the file is placed in
    the test split and train/val are empty. This lets the user point at one
    trajectory for ad-hoc evaluation without the split eating the only sample.
    """
    if not files:
        return [], [], []
    if len(files) == 1:
        print(f'[split] 1 file → test-only: {files}')
        return [], [], list(files)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(files)).tolist()
    n = len(shuffled)
    n_train = int(round(n * 0.6))
    n_val   = int(round(n * 0.2))
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    print(f'[split] {n} files → train={len(train)}, val={len(val)}, test={len(test)}')
    return train, val, test


def _real_split(real_files, real_test_files, seed):
    """Real-data split policy.

    If real_test_files is non-empty, treat it as the manual test set and
    auto-split the remaining (real_files − real_test_files) 80/20 into
    train/val. Otherwise fall back to the standard 60/20/20 _split_files
    behaviour on real_files.
    """
    if real_test_files:
        test = list(real_test_files)
        rest = [f for f in real_files if f not in set(test)]
        if not rest:
            print(f'[split] manual test={test}; no remaining files for train/val')
            return [], [], test
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(sorted(rest)).tolist()
        n_train = int(round(len(shuffled) * 0.8))
        train = shuffled[:n_train]
        val   = shuffled[n_train:]
        print(f'[split] manual test={len(test)} ({test}); rest {len(rest)} → train={len(train)}, val={len(val)}')
        return train, val, test
    return _split_files(real_files, seed)


def _load_scenario(data_dir, name, trim_start, scale_imu):
    """Load GT/IMU/DVL CSVs for one scenario. Optionally trim leading seconds and rescale IMU."""
    gt  = np.array(pd.read_csv(os.path.join(data_dir, f'GT_{name}.csv'),  header=0).iloc[:, 0:10])
    imu = np.array(pd.read_csv(os.path.join(data_dir, f'IMU_{name}.csv'), header=0).iloc[:, 0:7])
    dvl = np.array(pd.read_csv(os.path.join(data_dir, f'DVL_{name}.csv'), header=0).iloc[:, 0:4])
    if scale_imu:
        # Real-data only: IMU recorded at 1/100 of physical units; restore.
        imu[:, 1:7] *= 100
        # Real-data only: real GT CSV header is [Time, Longitude, Latitude, ...] but
        # the EKF and plot code assume the sim convention [Time, Latitude, Longitude, ...].
        gt[:, [1, 2]] = gt[:, [2, 1]]
    if trim_start > 0:
        gt  = gt[gt[:, 0]   >= trim_start]
        imu = imu[imu[:, 0] >= trim_start]
        dvl = dvl[dvl[:, 0] >= trim_start]
        gt[:, 0]  -= trim_start
        imu[:, 0] -= trim_start
        dvl[:, 0] -= trim_start
    return gt, imu, dvl


def _collect_scenario_data(name, data_dir, trim_start, scale_imu, ekf_fn, dvl_cfg, kf_cfg):
    """Run one scenario through the EKF in collect mode, return (features, labels)."""
    gt, imu, dvl = _load_scenario(data_dir, name, trim_start, scale_imu)
    _, _, _, _, features, labels = ekf_fn(
        imu, dvl, gt, imu.shape[0], dvl_cfg, kf_cfg, collect_data=True)
    return features, labels


def _write_config_txt(plots_dir, scenario_name, config, data_dir,
                      kf_cfg, dvl_cfg, extra_lines=None):
    """Write a human-readable config summary to <plots_dir>/config.txt."""
    micro_g    = 9.80665e-6
    deg_to_rad = np.pi / 180
    dph        = deg_to_rad / 3600

    lines = [
        f"Scenario : {scenario_name}",
        f"Datasets : {data_dir}",
        f"Trim     : {config.get('trim_start_seconds', 0)} s",
        "",
        "── DVL CONFIG ──────────────────────────────",
        f"  epoch_interval      : {dvl_cfg['epoch_interval']} s",
        "",
        "── EKF INITIAL UNCERTAINTIES ───────────────",
        f"  init_att_unc        : {np.rad2deg(kf_cfg['init_att_unc']):.4f} deg",
        f"  init_vel_unc        : {kf_cfg['init_vel_unc']:.4f} m/s",
        f"  init_pos_unc        : {kf_cfg['init_pos_unc']:.4f} m",
        f"  init_b_a_unc        : {kf_cfg['init_b_a_unc'] / micro_g:.2f} μg",
        f"  init_b_g_unc        : {kf_cfg['init_b_g_unc'] / dph:.4f} deg/h",
        "",
        "── EKF NOISE PSDs ──────────────────────────",
        f"  gyro_noise_PSD      : {kf_cfg['gyro_noise_PSD']:.3e} (rad/s)²/Hz",
        f"  accel_noise_PSD     : {kf_cfg['accel_noise_PSD']:.3e} (m/s²)²/Hz",
        f"  accel_bias_PSD      : {kf_cfg['accel_bias_PSD']:.3e}",
        f"  gyro_bias_PSD       : {kf_cfg['gyro_bias_PSD']:.3e}",
        "",
        "── EKF MEASUREMENT ─────────────────────────",
        f"  vel_meas_SD         : {kf_cfg['vel_meas_SD']:.4f} m/s",
        f"  dnn_vel_SD          : {kf_cfg.get('dnn_vel_SD', 'N/A')}",
    ]

    dnn_cfg = config.get('dnn_config')
    if dnn_cfg:
        lines += [
            "",
            "── DNN CONFIG ──────────────────────────────",
            f"  arch                : {dnn_cfg.get('arch', 'lstm')}",
            f"  window_size         : {dnn_cfg.get('window_size', 10)} DVL epochs",
            f"  hidden_size         : {dnn_cfg.get('hidden_size', 64)}",
            f"  num_layers          : {dnn_cfg.get('num_layers', 2)}",
            f"  batch_size          : {dnn_cfg.get('batch_size', 32)}",
            f"  epochs              : {dnn_cfg.get('epochs', 100)}",
            f"  lr                  : {dnn_cfg.get('lr', 1e-3)}",
        ]

    if extra_lines:
        lines += [""] + extra_lines

    with open(os.path.join(plots_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main(config):
    np.random.seed(2)

    data_type   = config['data_type']           # 'sim' | 'real'
    data_dir    = config[f'{data_type}_data_path']
    ekf_fn      = lc_ins_dvl_sim if data_type == 'sim' else lc_ins_dvl_real
    dvl_cfg     = config[f'dvl_cfg_{data_type}']
    kf_cfg      = config[f'kf_cfg_{data_type}']
    scale_imu   = (data_type == 'real')
    trim_start  = config.get('trim_start_seconds', 0) if data_type == 'sim' else 0
    output_dir  = config['output_dir']
    seed        = config.get('split_seed', 42)

    if data_type == 'sim':
        files = _discover_sim_scenarios(data_dir, config['sim_trajectory_name'])
        train_files, val_files, test_files = _split_files(files, seed)
    else:
        train_files, val_files, test_files = _real_split(
            config.get('real_files', []),
            config.get('real_test_files', []),
            seed)

    # =========================================================================
    # TRAIN DNN VELOCITY COMPENSATOR
    # =========================================================================
    if config['train_data']:
        dnn_config = config.get('dnn_config', {})

        print(f'Collecting training data from {len(train_files)} scenarios...')
        train_feats, train_lbls = [], []
        for i, sc in enumerate(train_files, 1):
            print(f'  [{i}/{len(train_files)}] {sc}')
            f, l = _collect_scenario_data(sc, data_dir, trim_start, scale_imu,
                                          ekf_fn, dvl_cfg, kf_cfg)
            train_feats.append(f); train_lbls.append(l)
        all_train_features = np.concatenate(train_feats, axis=0)
        all_train_labels   = np.concatenate(train_lbls,  axis=0)

        all_val_features, all_val_labels = None, None
        if val_files:
            print(f'Collecting validation data from {len(val_files)} scenarios...')
            val_feats, val_lbls = [], []
            for i, sc in enumerate(val_files, 1):
                print(f'  [{i}/{len(val_files)}] {sc}')
                f, l = _collect_scenario_data(sc, data_dir, trim_start, scale_imu,
                                              ekf_fn, dvl_cfg, kf_cfg)
                val_feats.append(f); val_lbls.append(l)
            all_val_features = np.concatenate(val_feats, axis=0)
            all_val_labels   = np.concatenate(val_lbls,  axis=0)

        print('Training DNN velocity compensator...')
        model, norm_stats = train_vel_compensator(
            all_train_features, all_train_labels, dnn_config,
            all_val_features, all_val_labels)

        model_filename = _build_model_filename(data_type, dnn_config,
                                               config.get('sim_trajectory_name'))
        model_path = os.path.join(output_dir, 'trained_model', model_filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_compensator(model, norm_stats, dnn_config, model_path)
        print(f'Saved model to: {model_path}')

    # =========================================================================
    # TEST DNN VELOCITY COMPENSATOR (sim or real, dispatched via data_type)
    # =========================================================================
    if config['test_data']:
        import copy
        dnn_config = config.get('dnn_config', {})
        arch       = dnn_config.get('arch', 'lstm')

        model_filename = _build_model_filename(data_type, dnn_config,
                                               config.get('sim_trajectory_name'))
        model_path = os.path.join(output_dir, 'trained_model', model_filename)
        print(f'Loading model from: {model_path}')
        compensator = load_compensator(model_path)

        all_err_base, all_err_dnn_p, all_err_dnn_np = [], [], []
        prmse_base_list,   vrmse_base_list   = [], []
        prmse_dnn_p_list,  vrmse_dnn_p_list  = [], []
        prmse_dnn_np_list, vrmse_dnn_np_list = [], []
        test_scenario_names = []

        active_test_files = test_files
        if data_type == 'sim' and config.get('sim_test_first_only', False):
            active_test_files = test_files[:1]
            print(f'[sim] sim_test_first_only=True → testing only {active_test_files}')

        for sc in active_test_files:
            print(f'Testing {data_type} scenario: {sc}')
            gt_t, imu_t, dvl_t = _load_scenario(data_dir, sc, trim_start, scale_imu)
            ne = imu_t.shape[0]

            _, out_err_base,   out_bias_base,   out_sd_base   = ekf_fn(
                imu_t, dvl_t, gt_t, ne, dvl_cfg, kf_cfg)

            _, out_err_dnn_np, out_bias_dnn_np, out_sd_dnn_np = ekf_fn(
                imu_t, dvl_t, gt_t, ne, dvl_cfg, kf_cfg,
                compensator=copy.deepcopy(compensator),
                update_P_after_dnn=False)

            _, out_err_dnn_p,  out_bias_dnn_p,  out_sd_dnn_p  = ekf_fn(
                imu_t, dvl_t, gt_t, ne, dvl_cfg, kf_cfg,
                compensator=copy.deepcopy(compensator),
                update_P_after_dnn=True)

            all_err_base.append(out_err_base)
            all_err_dnn_np.append(out_err_dnn_np)
            all_err_dnn_p.append(out_err_dnn_p)
            test_scenario_names.append(sc)

            prmse_base_list.append(float(np.sqrt(np.mean(
                out_err_base[:, 1]**2 + out_err_base[:, 2]**2 + out_err_base[:, 3]**2))))
            vrmse_base_list.append(float(np.sqrt(np.mean(
                out_err_base[:, 4]**2 + out_err_base[:, 5]**2 + out_err_base[:, 6]**2))))
            prmse_dnn_np_list.append(float(np.sqrt(np.mean(
                out_err_dnn_np[:, 1]**2 + out_err_dnn_np[:, 2]**2 + out_err_dnn_np[:, 3]**2))))
            vrmse_dnn_np_list.append(float(np.sqrt(np.mean(
                out_err_dnn_np[:, 4]**2 + out_err_dnn_np[:, 5]**2 + out_err_dnn_np[:, 6]**2))))
            prmse_dnn_p_list.append(float(np.sqrt(np.mean(
                out_err_dnn_p[:, 1]**2 + out_err_dnn_p[:, 2]**2 + out_err_dnn_p[:, 3]**2))))
            vrmse_dnn_p_list.append(float(np.sqrt(np.mean(
                out_err_dnn_p[:, 4]**2 + out_err_dnn_p[:, 5]**2 + out_err_dnn_p[:, 6]**2))))

            sc_dir = os.path.join(output_dir, 'plots', f'{data_type}_results',
                                  f"test_{sc}_{_dnn_config_token(dnn_config, kf_cfg.get('dnn_vel_SD'))}")
            os.makedirs(sc_dir, exist_ok=True)
            _write_config_txt(sc_dir, sc, config, data_dir, kf_cfg, dvl_cfg,
                              extra_lines=[f"model : {model_path}",
                                           f"arch  : {arch}",
                                           f"data  : {data_type}"])

            fig_pos_sc, fig_vel_sc = plot_pos_vel_two_runs(
                out_err_base, out_sd_base, out_err_dnn_p, out_sd_dnn_p,
                scenario=sc, arch=arch)

            if data_type == 'real':
                # Real: IMU rate >> GT rate, so downsample errors to GT times for trajectory plot.
                gt_times = gt_t[:, 0]
                idx_b = np.searchsorted(out_err_base[:, 0],  gt_times).clip(0, len(out_err_base)  - 1)
                idx_p = np.searchsorted(out_err_dnn_p[:, 0], gt_times).clip(0, len(out_err_dnn_p) - 1)
                fig_traj_sc = plot_trajectory_two_runs(
                    gt_t, out_err_base[idx_b], out_err_dnn_p[idx_p], scenario=sc, arch=arch)
            else:
                fig_traj_sc = plot_trajectory_two_runs(
                    gt_t, out_err_base, out_err_dnn_p, scenario=sc, arch=arch)

            fig_att_sc, fig_ba_sc, fig_bg_sc = plot_att_bias_two_runs(
                out_err_base, out_sd_base, out_bias_base,
                out_err_dnn_p, out_sd_dnn_p, out_bias_dnn_p,
                scenario=sc, arch=arch)

            fig_prmse_sc, fig_vrmse_sc = plot_prmse_vrmse_per_trajectory(
                [sc],
                [prmse_base_list[-1]],   [vrmse_base_list[-1]],
                [prmse_dnn_p_list[-1]],  [vrmse_dnn_p_list[-1]],
                arch=arch)

            fig_pos_sc.savefig(  os.path.join(sc_dir, 'position_errors.png'), dpi=150, bbox_inches='tight')
            fig_vel_sc.savefig(  os.path.join(sc_dir, 'velocity_errors.png'), dpi=150, bbox_inches='tight')
            fig_traj_sc.savefig( os.path.join(sc_dir, 'trajectory.png'),      dpi=150, bbox_inches='tight')
            fig_att_sc.savefig(  os.path.join(sc_dir, 'attitude_errors.png'), dpi=150, bbox_inches='tight')
            fig_ba_sc.savefig(   os.path.join(sc_dir, 'accel_bias.png'),      dpi=150, bbox_inches='tight')
            fig_bg_sc.savefig(   os.path.join(sc_dir, 'gyro_bias.png'),       dpi=150, bbox_inches='tight')
            fig_prmse_sc.savefig(os.path.join(sc_dir, 'prmse.png'),           dpi=150, bbox_inches='tight')
            fig_vrmse_sc.savefig(os.path.join(sc_dir, 'vrmse.png'),           dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f'  Scenario plots saved to {sc_dir}')

            # ----------------------------------------------------------------
            # Real-data only: also run Nadav's EKF baseline and save a sibling
            # folder test_<sc>_<ts>_nadav with the same standard plots
            # (Nadav alone vs GT — no DNN comparison).
            # ----------------------------------------------------------------
            if data_type == 'real':
                _, out_err_n, out_bias_n, out_sd_n = lc_ins_dvl_real_nadav(
                    imu_t, dvl_t, gt_t, ne, dvl_cfg, kf_cfg)

                # Nadav folder doesn't carry the DNN config token (Nadav doesn't use the DNN).
                nadav_dir = os.path.join(output_dir, 'plots', f'{data_type}_results',
                                         f"test_{sc}_nadav")
                os.makedirs(nadav_dir, exist_ok=True)
                _write_config_txt(nadav_dir, sc, config, data_dir, kf_cfg, dvl_cfg,
                                  extra_lines=[f"model : Nadav EKF baseline",
                                               f"data  : {data_type}"])

                # Use the same helpers as the Guy folder so Nadav inherits the
                # baseline colour scheme (dimgrey dashed line + grey ±σ envelope).
                fig_pos_n, fig_vel_n = plot_pos_vel_two_runs(
                    out_err_n, out_sd_n,
                    scenario=sc, arch=arch, base_label='Nadav EKF')
                fig_att_n, fig_ba_n, fig_bg_n = plot_att_bias_two_runs(
                    out_err_n, out_sd_n, out_bias_n,
                    scenario=sc, arch=arch, base_label='Nadav EKF')

                # Real data: IMU rate >> GT rate; downsample errors to GT times.
                gt_times = gt_t[:, 0]
                idx_n = np.searchsorted(out_err_n[:, 0], gt_times).clip(0, len(out_err_n) - 1)
                fig_traj_n = plot_trajectory_two_runs(
                    gt_t, out_err_n[idx_n], scenario=sc, arch=arch,
                    base_label='Nadav EKF')

                fig_pos_n.savefig( os.path.join(nadav_dir, 'position_errors.png'), dpi=150, bbox_inches='tight')
                fig_vel_n.savefig( os.path.join(nadav_dir, 'velocity_errors.png'), dpi=150, bbox_inches='tight')
                fig_att_n.savefig( os.path.join(nadav_dir, 'attitude_errors.png'), dpi=150, bbox_inches='tight')
                fig_ba_n.savefig(  os.path.join(nadav_dir, 'accel_bias.png'),      dpi=150, bbox_inches='tight')
                fig_bg_n.savefig(  os.path.join(nadav_dir, 'gyro_bias.png'),       dpi=150, bbox_inches='tight')
                fig_traj_n.savefig(os.path.join(nadav_dir, 'trajectory.png'),      dpi=150, bbox_inches='tight')
                plt.close('all')
                print(f'  Nadav baseline plots saved to {nadav_dir}')

        # Aggregate per-trajectory RMSE folder is only meaningful for 2+ scenarios.
        # For a single scenario the same single-bar plot already lives inside sc_dir.
        if len(all_err_base) >= 2:
            fig_prmse, fig_vrmse = plot_prmse_vrmse_per_trajectory(
                test_scenario_names,
                prmse_base_list,   vrmse_base_list,
                prmse_dnn_p_list,  vrmse_dnn_p_list,
                arch=arch)

            rmse_dir = os.path.join(output_dir, 'plots', f'{data_type}_results',
                                    f"test_rmse_{_dnn_config_token(dnn_config, kf_cfg.get('dnn_vel_SD'))}")
            os.makedirs(rmse_dir, exist_ok=True)
            _write_config_txt(rmse_dir, f"{len(test_scenario_names)} test scenarios",
                              config, data_dir, kf_cfg, dvl_cfg,
                              extra_lines=[f"scenarios : {', '.join(test_scenario_names)}",
                                           f"arch      : {arch}"])
            fig_prmse.savefig(os.path.join(rmse_dir, 'prmse_per_trajectory.png'), dpi=150, bbox_inches='tight')
            fig_vrmse.savefig(os.path.join(rmse_dir, 'vrmse_per_trajectory.png'), dpi=150, bbox_inches='tight')
            plt.close(fig_prmse)
            plt.close(fig_vrmse)
            print(f'PRMSE/VRMSE plots saved to {rmse_dir}')


if __name__ == '__main__':

    user_config = {
        # Where to write outputs (plots/, trained_model/)
        'output_dir': r'C:\Users\damar\PycharmProjects\EKF_CalibrateNet',

        # Active data source
        'data_type': 'real',                # 'sim' | 'real'

        # Per-type data paths (only the active one is used)
        'sim_data_path':  r'C:\Users\damar\MATLAB\Projects\EKFcompensateNet\simulated_EKF\long_turn_ba50-100-200ug_bg0p5-1-2dph_dvl1-2-5mms',
        'real_data_path': r'C:\Users\damar\PycharmProjects\EKF_CalibrateNet\real_data',

        # SIM: trajectory_name selects all GT_/IMU_/DVL_{trajectory_name}_*.csv triplets
        # under sim_data_path. The discovered scenarios are split 60/20/20.
        'sim_trajectory_name': 'long_turn',

        # REAL: explicit basenames (without GT_/IMU_/DVL_ prefix and .csv).
        # Split policy:
        #   - if real_test_files is non-empty → it IS the test split, and the
        #     remaining (real_files − real_test_files) is auto-split 80/20
        #     into train / val.
        #   - if real_test_files is empty → real_files is auto-split 60/20/20
        #     (with the 1-file = test-only shortcut still active).
        'real_files':      [f'trajectory{i}' for i in range(1, 14)],
        # 'real_test_files': ['trajectory4'],
        'real_test_files': ['trajectory4','trajectory13'],

        'split_seed': 42,                   # RNG seed for reproducible auto-splits

        # Mode flags
        'train_data': False,                # train DNN on the train split
        'test_data':  True,                 # run EKF baseline + DNN on the test split
        'sim_test_first_only': True,        # sim only: limit test loop to test_files[:1]

        'trim_start_seconds': 50,           # sim only

        'dnn_config': {
            'arch':        'tcn',          # 'lstm' | 'gru' | 'mlp' | 'bilstm' | 'bigru' | 'tcn' | 'transformer'
            'window_size':  10,             # DVL epochs per input window (~10s)
            'hidden_size':  64,
            'num_layers':    2,
            'batch_size':   32,
            'epochs':      100,
            'lr':          1e-8,
            'tag':         'lr8_traj_4_13',              # free-text suffix on the model filename (e.g. 'v2', 'tuned'). Empty = no suffix.
        },
    }

    # ========================================================================
    # DVL CONFIG — interval between DVL epochs (s)
    # ========================================================================
    DVL_config_sim = {'epoch_interval': 0.2}
    DVL_config_real = {'epoch_interval': 1.002506}

    # ========================================================================
    # KALMAN FILTER CONFIG — SIMULATED DATA
    # ========================================================================
    LC_KF_config_sim = {

        # Initial attitude uncertainty per axis (deg, converted to rad)
        # 2 deg too tight: P[att] shrinks faster than error converges after first DVL update
        'init_att_unc': np.deg2rad(2.0),

        # Initial velocity uncertainty per axis (m/s)
        # 3.0 too tight: draw (~3 m/s) + gravity mis-projection (g*sin(att)*dt ~ 0.34 m/s) = 3.34 m/s
        # bound grows in quadrature (3.02 m/s) while error grows additively (3.34 m/s) -> outside
        'init_vel_unc': 3.0,

        # Initial position uncertainty per axis (m)
        # 1.0 m too tight: init vel transient (3 m/s x 20s) causes ~12 m position error in first 50s
        'init_pos_unc': 5.0,

        # Initial accelerometer bias uncertainty per instrument (micro-g, converted to m/s^2)
        'init_b_a_unc': 9.80665e-4,

        # Initial gyro bias uncertainty per instrument (deg/hour, converted to rad/sec)
        'init_b_g_unc': 4.848e-6,

        'gyro_noise_PSD':  1.0e-6,
        'accel_noise_PSD': 9.617e-9,

        # Bias PSDs: small but non-zero
        # 1e-6/2e-9 -> P[bias] grows too fast -> biases absorb yaw error -> diverge
        # 1e-14 -> P[bias] collapses -> unstable cross-covariances
        # 1e-10/1e-12 -> P[bias] stays bounded but doesn't grow
        # 1e-10 -> P[acc_bias] collapses -> bounds shrink while estimate drifts -> inconsistent
        # 1e-8 -> keeps bounds wide enough to contain the estimate
        'accel_bias_PSD': 1.0e-6,
        'gyro_bias_PSD':  1.0e-8,

        # vel_meas_SD: 2.0 -> K[att] too small -> yaw can't be corrected -> diverges
        # 0.5 -> K[att] larger -> yaw correctable; K[bias] controlled by small bias_PSDs above
        'vel_meas_SD': 0.5,

        # dnn_vel_SD: noise assumed for the DNN correction when update_P_after_dnn=True.
        # Smaller value -> more trust in DNN -> stronger P shrinkage.
        # Start equal to vel_meas_SD and tune based on DNN correction magnitude.
        'dnn_vel_SD': 0.5,
    }

    # ========================================================================
    # KALMAN FILTER CONFIG — REAL DATA
    # ========================================================================
    LC_KF_config_real = {

        # Colleague conventional EKF: P0 = [0.2 m/s, 5 deg, 30 mg, 30 deg/h]
        'init_att_unc': np.deg2rad(2.0), # deg
        'init_vel_unc': 0.2, #m/s
        'init_pos_unc': 5.0, # m
        'init_b_a_unc': 1.0e-4, #micro-g
        'init_b_g_unc': np.deg2rad(30.0) / 3600, # deg/h

        'gyro_noise_PSD':  1.0e-5, # (rad/s)^2/HZ
        'accel_noise_PSD': 1.0e-2, # (m/s)^2/HZ

        'accel_bias_PSD': 1.0e-3,
        'gyro_bias_PSD':  1.0e-6,

        'vel_meas_SD': 0.5, #m/s
        'dnn_vel_SD':  0.5,
    }

    main({**user_config,
          'dvl_cfg_sim':  DVL_config_sim,
          'dvl_cfg_real': DVL_config_real,
          'kf_cfg_sim':   LC_KF_config_sim,
          'kf_cfg_real':  LC_KF_config_real})
