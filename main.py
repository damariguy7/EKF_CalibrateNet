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
from plot_errors import plot_errors_with_std, plot_results, plot_trajectory_2d, plot_trajectory_2d_comparison, plot_errors_comparison, plot_rmse_over_time, plot_pos_vel_two_runs, plot_att_bias_two_runs, plot_trajectory_two_runs, plot_prmse_vrmse_per_trajectory, plot_prmse_vrmse_per_axis, plot_dnn_sd_over_time
from dnn_vel_compensator import (train_vel_compensator, save_compensator,
                                  load_compensator, train_uncertainty,
                                  save_uncertainty, load_uncertainty,
                                  _UNC_DEFAULT_GROUPS)
from skew_symmetric import skew_symmetric


# Constants
deg_to_rad = 0.01745329252
rad_to_deg = 1 / deg_to_rad
micro_g_to_meters_per_second_squared = 9.80665e-6


def _lr_token(lr):
    """Encode the learning rate as 'lr<n>'.

    For a clean power of ten, n is the magnitude of the exponent, e.g.
    1e-8 -> 'lr8', 1e-3 -> 'lr3'. For a non-power-of-ten lr we fall back to a
    filesystem-safe mantissa/exponent form, e.g. 5e-4 -> 'lr5em4'.
    """
    try:
        exp = -math.log10(lr)
    except (ValueError, TypeError):
        return f"lr{str(lr).replace('.', 'p').replace('-', 'm')}"
    if abs(exp - round(exp)) < 1e-9:
        return f"lr{int(round(exp))}"
    s = f"{lr:.0e}".replace('-', 'm').replace('+', 'p').replace('.', 'p')
    return f"lr{s}"


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
    # Feature-set version: fv2 = 15-dim [innovation, v_pre, euler, f_ib_b, omega_ib_b];
    # fv3 = fv2 + 12-dim inter-epoch high-rate IMU aggregates.
    if dnn_config.get('imu_agg_features', False):
        _agg_set = dnn_config.get('imu_agg_set', 'full')
        fv_token = "fv3" if _agg_set == 'full' else f"fv3{_agg_set}"
    else:
        fv_token = "fv2"
    parts = [
        arch,
        fv_token,
        f"w{dnn_config.get('window_size', 10)}",
        f"h{dnn_config.get('hidden_size', 64)}",
        f"l{dnn_config.get('num_layers', 2)}",
        f"e{dnn_config.get('epochs', 100)}",
    ]
    if dnn_vel_sd is not None:
        parts.append(f"sd{str(dnn_vel_sd).replace('.', 'p')}")
    # Mode token: omit for 'sequential' to keep backwards-compatibility with
    # checkpoints/plot folders generated before joint mode existed.
    mode = dnn_config.get('dnn_mode', 'sequential')
    if mode != 'sequential':
        parts.append(f"m{mode}")
    if dnn_config.get('dnn_correct_attitude', False):
        parts.append("att")   # Option B: 6-dim output (velocity + attitude error)
    if dnn_config.get('dnn_apply_timing', 'epoch') != 'epoch':
        parts.append("tmidway")   # correction applied at the inter-DVL midpoint (plot only)
    # Loss token: omit for plain 'mse' (backwards-compatible); 'filter' = filter-aware loss.
    if dnn_config.get('loss_mode', 'mse') != 'mse':
        parts.append(f"l{dnn_config.get('loss_mode')}")
        parts.append(f"lam{str(dnn_config.get('filter_lambda', 1.0)).replace('.', 'p')}")
        if dnn_config.get('filter_horizon', 1) > 1:
            parts.append(f"hz{dnn_config.get('filter_horizon')}")
            parts.append(f"ld{str(dnn_config.get('filter_lambda_drift', 1.0)).replace('.', 'p')}")
    parts.append(_lr_token(dnn_config.get('lr', 1e-3)))
    tag = dnn_config.get('tag', '')
    if tag:
        parts.append(str(tag))
    return '_'.join(parts)


def _build_model_filename(data_type, dnn_config, sim_trajectory_name, dnn_vel_sd=None):
    """Build the .pth filename for the DNN compensator from training context.

    sim:  vel_compensator_sim_<arch>_fv2_<traj>_w<W>_h<H>_l<L>_e<E>[...].pth
    real: vel_compensator_real_<arch>_fv2_w<W>_h<H>_l<L>_e<E>[...].pth

    `fv2` marks the 15-dim feature set; it is always present now and keeps new
    checkpoints from colliding with / mis-loading the old 12-dim ones.
    For the filter-aware loss the name also carries `lfilter` and the training
    `dnn_vel_SD` (the loss bakes that gain), so a filter model is tied to its SD.
    Underscores inside the trajectory name are replaced with dashes so
    underscore stays a clean field separator. Empty/missing tag drops the
    trailing _<tag> segment.
    """
    arch = dnn_config.get('arch', 'lstm')
    if dnn_config.get('imu_agg_features', False):
        _agg_set = dnn_config.get('imu_agg_set', 'full')
        fv_token = "fv3" if _agg_set == 'full' else f"fv3{_agg_set}"
    else:
        fv_token = "fv2"
    parts = ['vel_compensator', data_type, arch, fv_token]
    if data_type == 'sim':
        traj_token = (sim_trajectory_name or 'unknown').replace('_', '-')
        parts.append(traj_token)
    parts += [
        f"w{dnn_config.get('window_size', 10)}",
        f"h{dnn_config.get('hidden_size', 64)}",
        f"l{dnn_config.get('num_layers', 2)}",
        f"e{dnn_config.get('epochs', 100)}",
    ]
    # Mode token: omit for 'sequential' so existing sequential checkpoints still load.
    mode = dnn_config.get('dnn_mode', 'sequential')
    if mode != 'sequential':
        parts.append(f"m{mode}")
    if dnn_config.get('dnn_correct_attitude', False):
        parts.append("att")   # Option B: 6-dim output (velocity + attitude error)
    # Loss token + baked SD + lambda: only for the filter-aware loss (mse keeps
    # SD-free names, since an mse-trained model is SD-agnostic at inference).
    if dnn_config.get('loss_mode', 'mse') != 'mse':
        parts.append(f"l{dnn_config.get('loss_mode')}")
        if dnn_vel_sd is not None:
            parts.append(f"sd{str(dnn_vel_sd).replace('.', 'p')}")
        parts.append(f"lam{str(dnn_config.get('filter_lambda', 1.0)).replace('.', 'p')}")
        if dnn_config.get('filter_horizon', 1) > 1:
            parts.append(f"hz{dnn_config.get('filter_horizon')}")
            parts.append(f"ld{str(dnn_config.get('filter_lambda_drift', 1.0)).replace('.', 'p')}")
    parts.append(_lr_token(dnn_config.get('lr', 1e-3)))
    tag = dnn_config.get('tag', '')
    if tag:
        parts.append(str(tag))
    return '_'.join(parts) + '.pth'


def _build_unc_config(dnn_config):
    """Phase-2 sigma-net config from dnn_config's unc_* overrides.

    Hyper-params default to the phase-1 values; input-design knobs
    (feature_groups / add_magnitudes / corr_skip) default per the plan.
    """
    return {
        'arch':           dnn_config.get('unc_arch', dnn_config.get('arch', 'tcn')),
        'hidden_size':    dnn_config.get('unc_hidden_size', dnn_config.get('hidden_size', 64)),
        'num_layers':     dnn_config.get('num_layers', 2),
        'window_size':    dnn_config.get('window_size', 10),
        'batch_size':     dnn_config.get('batch_size', 32),
        'epochs':         dnn_config.get('unc_epochs', dnn_config.get('epochs', 100)),
        'lr':             dnn_config.get('unc_lr', dnn_config.get('lr', 1e-3)),
        'seed':           dnn_config.get('seed', 42),
        'feature_groups': dnn_config.get('unc_feature_groups', list(_UNC_DEFAULT_GROUPS)),
        'add_magnitudes': dnn_config.get('unc_add_magnitudes', True),
        'corr_skip':      dnn_config.get('unc_corr_skip', False),
        # Early stopping: sigma-net overfits sooner, so a tighter patience by default.
        'early_stopping': dnn_config.get('early_stopping', True),
        'es_patience':    dnn_config.get('unc_es_patience', dnn_config.get('es_patience', 15)),
        'es_min_delta':   dnn_config.get('es_min_delta', 0.0),
        'es_min_epochs':  dnn_config.get('unc_es_min_epochs', dnn_config.get('es_min_epochs', 0)),
        'weight_decay':   dnn_config.get('unc_weight_decay', dnn_config.get('weight_decay', 0.0)),
        'lr_scheduler':   dnn_config.get('unc_lr_scheduler', dnn_config.get('lr_scheduler', 'none')),
        # Phase-2 target: 'sigma' (learned dnn_vel_SD) or 'fuse_gate' (learned α).
        'target':         dnn_config.get('unc_target', 'sigma'),
        'fuse_gate_max':  dnn_config.get('fuse_gate_max', 0.5),
    }


def _uncsd_path(model_path):
    """Sibling checkpoint path for the phase-2 sigma-net (model + _uncsd suffix)."""
    return model_path[:-4] + '_uncsd.pth' if model_path.endswith('.pth') \
        else model_path + '_uncsd.pth'


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


def _real_split(real_files, real_test_files, seed, val_count=None):
    """Real-data split policy.

    If real_test_files is non-empty, treat it as the manual test set and
    auto-split the remaining (real_files − real_test_files) into train/val.
    `val_count` sets the number of validation trajectories exactly (clamped to
    keep ≥1 train); when None, fall back to an 80/20 split. Otherwise (no manual
    test set) fall back to the standard 60/20/20 _split_files behaviour.
    """
    if real_test_files:
        test = list(real_test_files)
        rest = [f for f in real_files if f not in set(test)]
        if not rest:
            print(f'[split] manual test={test}; no remaining files for train/val')
            return [], [], test
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(sorted(rest)).tolist()
        if val_count is not None:
            n_val   = max(0, min(int(val_count), len(shuffled) - 1))
            n_train = len(shuffled) - n_val
        else:
            n_train = int(round(len(shuffled) * 0.8))
        train = shuffled[:n_train]
        val   = shuffled[n_train:]
        print(f'[split] manual test={len(test)} ({test}); rest {len(rest)} → train={len(train)}, val={len(val)}')
        return train, val, test
    return _split_files(real_files, seed)


def _load_scenario(data_dir, name, trim_start, scale_imu, use_noised_dvl=False):
    """Load GT/IMU/DVL CSVs for one scenario. Optionally trim leading seconds and rescale IMU.

    use_noised_dvl=True loads DVL_<name>_noised.csv (produced by noise_dvl.py)
    instead of DVL_<name>.csv, so the EKF runs on the perturbed DVL recordings.
    """
    dvl_name = f'DVL_{name}_noised.csv' if use_noised_dvl else f'DVL_{name}.csv'
    gt  = np.array(pd.read_csv(os.path.join(data_dir, f'GT_{name}.csv'),  header=0).iloc[:, 0:10])
    imu = np.array(pd.read_csv(os.path.join(data_dir, f'IMU_{name}.csv'), header=0).iloc[:, 0:7])
    dvl = np.array(pd.read_csv(os.path.join(data_dir, dvl_name), header=0).iloc[:, 0:4])
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


def _trim_end_seconds(gt, imu, dvl, seconds):
    """Drop the last `seconds` of data from a scenario's GT/IMU/DVL arrays.

    The cutoff is a single absolute time computed from the scenario's overall
    end (max time across the three sources) minus `seconds`; every array is
    filtered to times <= that cutoff so they stay aligned to the same end.
    Times are NOT shifted. seconds <= 0 is a no-op.
    """
    if not seconds or seconds <= 0:
        return gt, imu, dvl
    t_end = max(gt[:, 0].max(), imu[:, 0].max(), dvl[:, 0].max()) - seconds
    gt  = gt[gt[:, 0]   <= t_end]
    imu = imu[imu[:, 0] <= t_end]
    dvl = dvl[dvl[:, 0] <= t_end]
    return gt, imu, dvl


def _precompute_filter_targets(aux, vel_meas_SD, dnn_vel_SD, dnn_mode='joint'):
    """Build the affine post-update-velocity map v_post = c + A @ correction for
    the filter-aware / drift loss. A, c depend only on (P, H, R) — not on the DNN
    output — so they are constants the loss differentiates the correction through.

    joint:      single 6-row DVL+DNN update (mirrors lc_ekf_epoch_joint).
                aux is a-priori: v_pre, C_b_n, P_pred, dvl_v.
                A = K[3:6,3:6];  c = v_pre - K[3:6,0:3] @ (C^T v_pre - dvl_v).
    sequential: the velocity-only DNN update applied AFTER the DVL update.
                aux is POST-DVL: v_pre = post-DVL velocity (= c), P_pred = post-DVL P.
                K_dnn = P Hvel^T (Hvel P Hvel^T + R_dnn)^-1, Hvel = I on vel states.
                v_post = v_postDVL + K_dnn[3:6,:] @ correction → A = K_dnn[3:6,:], c = v_postDVL.

    Returns A (N,3,3), c (N,3), v_true (N,3).
    """
    v_pre  = aux['v_pre']      # (N,3)  pre-update (joint) / post-DVL (sequential)
    v_true = aux['v_true']     # (N,3)
    P      = aux['P_pred']     # (N,15,15)
    N = v_pre.shape[0]
    A = np.zeros((N, 3, 3))
    c = np.zeros((N, 3))

    if dnn_mode == 'sequential':
        R_dnn = np.eye(3) * dnn_vel_SD ** 2
        for i in range(N):
            Pi = P[i]
            S = Pi[3:6, 3:6] + R_dnn
            K_dnn = Pi[:, 3:6] @ np.linalg.inv(S)   # (15,3)
            A[i] = K_dnn[3:6, :]
            c[i] = v_pre[i]                          # post-DVL velocity
        return A, c, v_true

    # joint
    C     = aux['C_b_n']      # (N,3,3)
    dvl_v = aux['dvl_v']      # (N,3)
    H_dnn = np.zeros((3, 15))
    H_dnn[0:3, 3:6] = np.eye(3)
    R = np.zeros((6, 6))
    R[0:3, 0:3] = np.eye(3) * vel_meas_SD ** 2
    R[3:6, 3:6] = np.eye(3) * dnn_vel_SD ** 2
    for i in range(N):
        Ci = C[i]
        H_dvl = np.zeros((3, 15))
        H_dvl[0:3, 0:3] = -Ci.T @ skew_symmetric(v_pre[i])
        H_dvl[0:3, 3:6] = Ci.T
        H = np.vstack([H_dvl, H_dnn])
        S = H @ P[i] @ H.T + R
        K = P[i] @ H.T @ np.linalg.inv(S)         # (15,6)
        Kv = K[3:6, :]                             # (3,6) velocity rows
        delta_z_dvl = Ci.T @ v_pre[i] - dvl_v[i]   # (3,)
        A[i] = Kv[:, 3:6]
        c[i] = v_pre[i] - Kv[:, 0:3] @ delta_z_dvl
    return A, c, v_true


def _collect_scenario_data(name, data_dir, trim_start, scale_imu, ekf_fn, dvl_cfg, kf_cfg,
                           dnn_mode='sequential', capture_aux=False, correct_attitude=False,
                           imu_agg_features=False, imu_agg_set='full', use_noised_dvl=False):
    """Run one scenario through the EKF in collect mode, return (features, labels).

    `dnn_mode` selects the label expression: 'sequential' yields the post-DVL
    residual, 'joint' yields the pre-DVL residual (the target a DNN co-fused
    with DVL in a single Kalman update must learn).

    When `capture_aux=True` (joint mode, for the filter-aware loss), also returns
    a per-epoch aux dict (v_pre/v_true/C_b_n/P_pred/dvl_v) → (features, labels, aux).
    `correct_attitude=True` (Option B) appends a 3-dim attitude-error block to the
    label → 6-dim labels. `imu_agg_features=True` (fv3) appends 12-dim inter-epoch
    high-rate IMU aggregates to the feature → 27-dim features.
    """
    gt, imu, dvl = _load_scenario(data_dir, name, trim_start, scale_imu, use_noised_dvl)
    if capture_aux:
        _, _, _, _, features, labels, aux = ekf_fn(
            imu, dvl, gt, imu.shape[0], dvl_cfg, kf_cfg,
            collect_data=True, dnn_mode=dnn_mode, capture_aux=True,
            correct_attitude=correct_attitude, imu_agg_features=imu_agg_features,
            imu_agg_set=imu_agg_set)
        return features, labels, aux
    _, _, _, _, features, labels = ekf_fn(
        imu, dvl, gt, imu.shape[0], dvl_cfg, kf_cfg,
        collect_data=True, dnn_mode=dnn_mode, correct_attitude=correct_attitude,
        imu_agg_features=imu_agg_features, imu_agg_set=imu_agg_set)
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


def _training_curve_lines(history):
    """Return a per-epoch train/val loss table (list of str) for config.txt.

    Losses are on the normalised scale used during training (≈1.0 means the
    model is no better than predicting the label mean).
    """
    lines = [
        "",
        "── TRAINING CURVE (per epoch) ──────────────",
        "  epoch    train_loss     val_loss",
    ]
    for e, tr, va in zip(history['epoch'], history['train_loss'], history['val_loss']):
        va_str = '        N/A' if va is None else f"{va:12.6f}"
        lines.append(f"  {e:5d}  {tr:12.6f}  {va_str}")
    return lines


def _save_training_results(history, out_dir):
    """Persist the per-epoch train/val loss curves to <out_dir>.

    Writes two files:
      - training_curve.csv  : epoch, train_loss, val_loss (raw numbers)
      - training_curve.png  : log-scale loss-vs-epoch plot
    Losses are on the normalised scale used during training (≈1.0 means the
    model is no better than predicting the label mean).
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs     = history['epoch']
    train_loss = history['train_loss']
    val_loss   = history['val_loss']
    has_val    = any(v is not None for v in val_loss)

    csv_path = os.path.join(out_dir, 'training_curve.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss\n')
        for e, tr, va in zip(epochs, train_loss, val_loss):
            f.write(f"{e},{tr:.8f},{'' if va is None else f'{va:.8f}'}\n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label='train', color='tab:blue')
    if has_val:
        ax.plot(epochs, val_loss, label='val', color='tab:orange')
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='predict-mean baseline')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE (normalised)')
    ax.set_yscale('log')
    ax.set_title('DNN velocity compensator training')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    png_path = os.path.join(out_dir, 'training_curve.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curves saved to: {out_dir}')


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
    # Load DVL_<name>_noised.csv (from noise_dvl.py) instead of DVL_<name>.csv.
    use_noised_dvl = config.get('use_noised_dvl', False)

    # One timestamp per run, appended to every results folder so repeated runs
    # with the same config don't overwrite each other (format: YYYYMMDD_HHMMSS).
    run_stamp   = datetime.now().strftime('%Y%m%d_%H%M%S')

    if data_type == 'sim':
        files = _discover_sim_scenarios(data_dir, config['sim_trajectory_name'])
        train_files, val_files, test_files = _split_files(files, seed)
    else:
        train_files, val_files, test_files = _real_split(
            config.get('real_files', []),
            config.get('real_test_files', []),
            seed,
            val_count=config.get('real_val_count'))

    # =========================================================================
    # TRAIN DNN VELOCITY COMPENSATOR
    # =========================================================================
    if config['train_data']:
        dnn_config = config.get('dnn_config', {})
        dnn_mode   = dnn_config.get('dnn_mode', 'sequential')
        loss_mode  = dnn_config.get('loss_mode', 'mse')
        print(f'[train] dnn_mode = {dnn_mode}   loss_mode = {loss_mode}')

        # Filter-aware / drift loss needs the per-epoch EKF state captured during
        # collection (joint: a-priori; sequential: post-DVL).
        capture_aux = (loss_mode == 'filter')
        if capture_aux and dnn_mode not in ('joint', 'sequential'):
            raise ValueError(f"loss_mode='filter' not supported for dnn_mode='{dnn_mode}'.")
        # Option B: 6-dim (velocity + attitude) labels. The filter-aware loss is
        # velocity-only for now, so attitude correction uses plain MSE.
        correct_attitude = dnn_config.get('dnn_correct_attitude', False)
        if correct_attitude and loss_mode == 'filter':
            raise ValueError("dnn_correct_attitude currently supports loss_mode='mse' only "
                             "(filter-aware loss does not yet cover the attitude block).")
        vel_meas_SD = kf_cfg.get('vel_meas_SD')
        dnn_vel_SD  = kf_cfg.get('dnn_vel_SD', vel_meas_SD)
        # fv3: high-rate inter-epoch IMU aggregates appended to the feature.
        imu_agg_features = dnn_config.get('imu_agg_features', False)
        imu_agg_set = dnn_config.get('imu_agg_set', 'full')

        def _collect_split(files):
            """Collect features/labels (and, if capture_aux, filter targets) over a file list."""
            feats, lbls, auxes = [], [], []
            for i, sc in enumerate(files, 1):
                print(f'  [{i}/{len(files)}] {sc}')
                if capture_aux:
                    f, l, aux = _collect_scenario_data(
                        sc, data_dir, trim_start, scale_imu, ekf_fn, dvl_cfg, kf_cfg,
                        dnn_mode=dnn_mode, capture_aux=True, correct_attitude=correct_attitude,
                        imu_agg_features=imu_agg_features, imu_agg_set=imu_agg_set,
                        use_noised_dvl=use_noised_dvl)
                    auxes.append(aux)
                else:
                    f, l = _collect_scenario_data(
                        sc, data_dir, trim_start, scale_imu, ekf_fn, dvl_cfg, kf_cfg,
                        dnn_mode=dnn_mode, correct_attitude=correct_attitude,
                        imu_agg_features=imu_agg_features, imu_agg_set=imu_agg_set,
                        use_noised_dvl=use_noised_dvl)
                feats.append(f); lbls.append(l)
            features = np.concatenate(feats, axis=0)
            labels   = np.concatenate(lbls,  axis=0)
            targets = None
            if capture_aux:
                aux_all = {k: np.concatenate([a[k] for a in auxes], axis=0)
                           for k in auxes[0]}
                targets = _precompute_filter_targets(aux_all, vel_meas_SD, dnn_vel_SD,
                                                     dnn_mode=dnn_mode)
            return features, labels, targets

        print(f'Collecting training data from {len(train_files)} scenarios...')
        all_train_features, all_train_labels, train_targets = _collect_split(train_files)

        all_val_features, all_val_labels, val_targets = None, None, None
        if val_files:
            print(f'Collecting validation data from {len(val_files)} scenarios...')
            all_val_features, all_val_labels, val_targets = _collect_split(val_files)

        print('Training DNN velocity compensator...')
        model, norm_stats, history = train_vel_compensator(
            all_train_features, all_train_labels, dnn_config,
            all_val_features, all_val_labels,
            loss_mode=loss_mode, train_targets=train_targets, val_targets=val_targets)

        model_filename = _build_model_filename(data_type, dnn_config,
                                               config.get('sim_trajectory_name'),
                                               dnn_vel_sd=dnn_vel_SD)
        model_path = os.path.join(output_dir, 'trained_model', model_filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_compensator(model, norm_stats, dnn_config, model_path)
        print(f'Saved model to: {model_path}')

        # Persist the train/val loss curves into a results folder named after the
        # DNN config (so different arch/lr/mode runs don't overwrite each other).
        train_dir = os.path.join(output_dir, 'plots', 'trains',
                                 f"train_{_dnn_config_token(dnn_config, kf_cfg.get('dnn_vel_SD'))}_{run_stamp}")
        _save_training_results(history, train_dir)
        _write_config_txt(train_dir, f"{len(train_files)} train / "
                                     f"{len(val_files)} val scenarios",
                          config, data_dir, kf_cfg, dvl_cfg,
                          extra_lines=[f"model    : {model_path}",
                                       f"mode     : {dnn_mode}",
                                       f"final train_loss : {history['train_loss'][-1]:.6f}",
                                       f"final val_loss   : "
                                       f"{history['val_loss'][-1] if history['val_loss'][-1] is not None else 'N/A'}"]
                                      + _training_curve_lines(history))

        # ---- Phase 2: learned per-epoch, per-axis dnn_vel_SD (uncertainty net) ----
        # Freeze the just-trained velocity model and train a sigma-net that
        # predicts the per-axis log-variance of its correction (NLL on
        # r = corr - label). Sequential-only for v1; reuses the phase-1 data.
        if dnn_config.get('train_dnn_sd', False):
            if dnn_mode != 'sequential':
                print(f"[uncertainty] skip: learned dnn_vel_SD is sequential-only "
                      f"(dnn_mode={dnn_mode}).")
            else:
                unc_config = _build_unc_config(dnn_config)
                print('Training phase-2 uncertainty (sigma) net...')
                unc_model, unc_norm_stats, unc_history = train_uncertainty(
                    all_train_features, all_train_labels, unc_config,
                    model, norm_stats,
                    val_features=all_val_features, val_labels=all_val_labels)
                unc_path = _uncsd_path(model_path)
                save_uncertainty(unc_model, unc_norm_stats, unc_config, unc_path)
                print(f'Saved uncertainty model to: {unc_path}')
                _save_training_results(unc_history,
                                       os.path.join(train_dir, 'uncertainty'))

    # =========================================================================
    # TEST DNN VELOCITY COMPENSATOR (sim or real, dispatched via data_type)
    # =========================================================================
    if config['test_data']:
        import copy
        dnn_config = config.get('dnn_config', {})
        arch       = dnn_config.get('arch', 'lstm')

        # Discover which mode checkpoints exist. Both modes' files are looked up
        # by toggling 'dnn_mode' on a copy of dnn_config; whichever .pth files
        # are present participate in the comparison.
        modes_to_test = []
        for mode in ('sequential', 'joint'):
            cfg_for_mode = {**dnn_config, 'dnn_mode': mode}
            fname = _build_model_filename(data_type, cfg_for_mode,
                                          config.get('sim_trajectory_name'),
                                          dnn_vel_sd=kf_cfg.get('dnn_vel_SD'))
            path  = os.path.join(output_dir, 'trained_model', fname)
            if os.path.exists(path):
                print(f'Loading {mode} model from: {path}')
                comp = load_compensator(path)
                # Attach the sibling phase-2 net (sequential only). It serves either
                # 'kalman' (learned dnn_vel_SD σ, use_learned_dnn_sd) or 'fuse' (learned
                # gate α, use_learned_fuse_scale); VelCompensator picks σ vs α from the
                # saved target. Attach if either consumer is on.
                if (kf_cfg.get('use_learned_dnn_sd', False)
                        or kf_cfg.get('use_learned_fuse_scale', False)) and mode == 'sequential':
                    unc_path = _uncsd_path(path)
                    if os.path.exists(unc_path):
                        unc_model, unc_norm_stats = load_uncertainty(unc_path)
                        comp.attach_uncertainty(unc_model, unc_norm_stats)
                        print(f'  attached phase-2 net '
                              f'(target={unc_norm_stats.get("target", "sigma")}): {unc_path}')
                    else:
                        print(f'  [phase-2] no net at {unc_path}; falling back to constant')
                modes_to_test.append({
                    'mode':        mode,
                    'cfg':         cfg_for_mode,
                    'path':        path,
                    'compensator': comp,
                })
            else:
                print(f'[skip] {mode} checkpoint not found at: {path}')

        if not modes_to_test:
            raise FileNotFoundError(
                "No DNN checkpoint found for either 'sequential' or 'joint' mode. "
                "Train at least one mode first (set 'train_data': True with the "
                "desired 'dnn_mode' in dnn_config).")

        # Per-mode aggregation buckets keyed by mode name
        prmse_by_mode = {m['mode']: [] for m in modes_to_test}
        vrmse_by_mode = {m['mode']: [] for m in modes_to_test}
        prmse_base_list, vrmse_base_list = [], []
        test_scenario_names              = []

        active_test_files = test_files
        if data_type == 'sim' and config.get('sim_test_first_only', False):
            active_test_files = test_files[:1]
            print(f'[sim] sim_test_first_only=True → testing only {active_test_files}')

        # Optional per-test-trajectory tail trim: test_trim_end_seconds[i] cuts the
        # last i-th seconds off the i-th test trajectory (by position). Shorter
        # list / missing entries → no trim for those trajectories.
        test_trim_end = config.get('test_trim_end_seconds', [])

        for i_sc, sc in enumerate(active_test_files):
            trim_end = test_trim_end[i_sc] if i_sc < len(test_trim_end) else 0
            print(f'Testing {data_type} scenario: {sc}'
                  + (f' (trim last {trim_end}s)' if trim_end else ''))
            gt_t, imu_t, dvl_t = _load_scenario(data_dir, sc, trim_start, scale_imu, use_noised_dvl)
            gt_t, imu_t, dvl_t = _trim_end_seconds(gt_t, imu_t, dvl_t, trim_end)
            ne = imu_t.shape[0]

            # ---- Baseline: no DNN (one run per scenario, shared across modes) ----
            _, out_err_base, out_bias_base, out_sd_base = ekf_fn(
                imu_t, dvl_t, gt_t, ne, dvl_cfg, kf_cfg)

            test_scenario_names.append(sc)
            prmse_base_list.append(float(np.sqrt(np.mean(
                out_err_base[:, 1]**2 + out_err_base[:, 2]**2 + out_err_base[:, 3]**2))))
            vrmse_base_list.append(float(np.sqrt(np.mean(
                out_err_base[:, 4]**2 + out_err_base[:, 5]**2 + out_err_base[:, 6]**2))))

            # Baseline attitude error (deg) — is traj position error heading-limited?
            # out_err cols 7,8,9 = roll,pitch,yaw error (rad). Report RMSE + yaw bias.
            _att = out_err_base[:, 7:10] * rad_to_deg
            _yaw_rmse = float(np.sqrt(np.mean(_att[:, 2]**2)))
            _yaw_bias = float(np.mean(_att[:, 2]))
            print(f"  [att] {sc} baseline yaw RMSE={_yaw_rmse:.3f} deg (bias={_yaw_bias:+.3f}); "
                  f"roll RMSE={np.sqrt(np.mean(_att[:,0]**2)):.3f}, pitch RMSE={np.sqrt(np.mean(_att[:,1]**2)):.3f}")

            # ---- One inference run per available mode, in its own plot folder ----
            for entry in modes_to_test:
                mode        = entry['mode']
                cfg_for_mode = entry['cfg']
                compensator = entry['compensator']

                # Log the applied per-axis dnn_vel_SD (learned sigma) over time so it
                # can be plotted. Sequential real-data only (lc_ins_dvl_real accepts
                # the dnn_sigma_log kwarg; lc_ins_dvl_sim does not).
                dnn_sigma_log = [] if (data_type == 'real' and mode == 'sequential') else None
                _, out_err_dnn, out_bias_dnn, out_sd_dnn = ekf_fn(
                    imu_t, dvl_t, gt_t, ne, dvl_cfg, kf_cfg,
                    compensator=copy.deepcopy(compensator),
                    update_P_after_dnn=True,
                    dnn_mode=mode,
                    imu_agg_features=cfg_for_mode.get('imu_agg_features', False),
                    imu_agg_set=cfg_for_mode.get('imu_agg_set', 'full'),
                    dnn_apply_timing=cfg_for_mode.get('dnn_apply_timing', 'epoch'),
                    **({'dnn_sigma_log': dnn_sigma_log} if data_type == 'real' else {}))

                prmse_by_mode[mode].append(float(np.sqrt(np.mean(
                    out_err_dnn[:, 1]**2 + out_err_dnn[:, 2]**2 + out_err_dnn[:, 3]**2))))
                vrmse_by_mode[mode].append(float(np.sqrt(np.mean(
                    out_err_dnn[:, 4]**2 + out_err_dnn[:, 5]**2 + out_err_dnn[:, 6]**2))))

                # Console comparison so the benefit (or not) is visible without
                # opening the plots: baseline vs this mode, per scenario.
                _pb, _vb = prmse_base_list[-1], vrmse_base_list[-1]
                _pm, _vm = prmse_by_mode[mode][-1], vrmse_by_mode[mode][-1]
                print(f"  [{mode}] {sc}: "
                      f"PRMSE {_pb:.3f} -> {_pm:.3f} m ({100*(_pm-_pb)/_pb:+.1f}%)   "
                      f"VRMSE {_vb:.4f} -> {_vm:.4f} m/s ({100*(_vm-_vb)/_vb:+.1f}%)")

                # Turn-rate error slice: does the DNN help during maneuvers? Bucket
                # epochs by gyro magnitude ||omega|| (imu cols 4:6, rad/s); compare
                # baseline vs mode velocity RMSE in the high-turn (top quartile) vs
                # low-turn buckets. (GT held within intervals affects both equally,
                # so the baseline-vs-mode comparison per bucket is still meaningful.)
                _wmag = np.linalg.norm(imu_t[:out_err_base.shape[0], 4:7], axis=1)
                _thr = np.percentile(_wmag, 75)
                _hi = _wmag > _thr
                _veb = np.sqrt(out_err_base[:, 4]**2 + out_err_base[:, 5]**2 + out_err_base[:, 6]**2)
                _vem = np.sqrt(out_err_dnn[:, 4]**2 + out_err_dnn[:, 5]**2 + out_err_dnn[:, 6]**2)

                def _rmse(a, m):
                    return float(np.sqrt(np.mean(a[m]**2))) if m.any() else float('nan')
                _hb, _hm = _rmse(_veb, _hi), _rmse(_vem, _hi)
                _lb, _lm = _rmse(_veb, ~_hi), _rmse(_vem, ~_hi)
                print(f"        turn-slice VRMSE  high(>{np.rad2deg(_thr):.0f}deg/s): "
                      f"{_hb:.4f}->{_hm:.4f} ({100*(_hm-_hb)/_hb:+.1f}%)   "
                      f"low: {_lb:.4f}->{_lm:.4f} ({100*(_lm-_lb)/_lb:+.1f}%)")

                # Plot folder name carries the mode token so sequential and joint
                # plots never collide.
                sc_dir = os.path.join(output_dir, 'plots', f'{data_type}_results',
                                      f"test_{sc}_{_dnn_config_token(cfg_for_mode, kf_cfg.get('dnn_vel_SD'))}_{run_stamp}")
                os.makedirs(sc_dir, exist_ok=True)
                _write_config_txt(sc_dir, sc,
                                  {**config, 'dnn_config': cfg_for_mode},
                                  data_dir, kf_cfg, dvl_cfg,
                                  extra_lines=[f"model : {entry['path']}",
                                               f"arch  : {arch}",
                                               f"mode  : {mode}",
                                               f"data  : {data_type}",
                                               f"trim_end_seconds : {trim_end}"])

                fig_pos_sc, fig_vel_sc = plot_pos_vel_two_runs(
                    out_err_base, out_sd_base, out_err_dnn, out_sd_dnn,
                    scenario=sc, arch=arch)

                if data_type == 'real':
                    gt_times = gt_t[:, 0]
                    idx_b = np.searchsorted(out_err_base[:, 0], gt_times).clip(0, len(out_err_base) - 1)
                    idx_d = np.searchsorted(out_err_dnn[:, 0],  gt_times).clip(0, len(out_err_dnn)  - 1)
                    fig_traj_sc = plot_trajectory_two_runs(
                        gt_t, out_err_base[idx_b], out_err_dnn[idx_d], scenario=sc, arch=arch)
                else:
                    fig_traj_sc = plot_trajectory_two_runs(
                        gt_t, out_err_base, out_err_dnn, scenario=sc, arch=arch)

                fig_att_sc, fig_ba_sc, fig_bg_sc = plot_att_bias_two_runs(
                    out_err_base, out_sd_base, out_bias_base,
                    out_err_dnn,  out_sd_dnn,  out_bias_dnn,
                    scenario=sc, arch=arch)

                fig_prmse_sc, fig_vrmse_sc = plot_prmse_vrmse_per_trajectory(
                    [sc],
                    [prmse_base_list[-1]],     [vrmse_base_list[-1]],
                    [prmse_by_mode[mode][-1]], [vrmse_by_mode[mode][-1]],
                    arch=arch)

                fig_prmse_axis, fig_vrmse_axis = plot_prmse_vrmse_per_axis(
                    out_err_base, out_err_dnn, scenario=sc, arch=arch)

                # Learned dnn_vel_SD (sigma) over time — sequential real runs only.
                fig_sd = None
                if dnn_sigma_log:
                    fig_sd = plot_dnn_sd_over_time(
                        dnn_sigma_log, scenario=sc, arch=arch,
                        const_sd=kf_cfg.get('dnn_vel_SD'))

                fig_pos_sc.savefig(  os.path.join(sc_dir, 'position_errors.png'), dpi=150, bbox_inches='tight')
                fig_vel_sc.savefig(  os.path.join(sc_dir, 'velocity_errors.png'), dpi=150, bbox_inches='tight')
                fig_traj_sc.savefig( os.path.join(sc_dir, 'trajectory.png'),      dpi=150, bbox_inches='tight')
                fig_att_sc.savefig(  os.path.join(sc_dir, 'attitude_errors.png'), dpi=150, bbox_inches='tight')
                fig_ba_sc.savefig(   os.path.join(sc_dir, 'accel_bias.png'),      dpi=150, bbox_inches='tight')
                fig_bg_sc.savefig(   os.path.join(sc_dir, 'gyro_bias.png'),       dpi=150, bbox_inches='tight')
                fig_prmse_sc.savefig(os.path.join(sc_dir, 'prmse.png'),           dpi=150, bbox_inches='tight')
                fig_vrmse_sc.savefig(os.path.join(sc_dir, 'vrmse.png'),           dpi=150, bbox_inches='tight')
                fig_prmse_axis.savefig(os.path.join(sc_dir, 'prmse_per_axis.png'), dpi=150, bbox_inches='tight')
                fig_vrmse_axis.savefig(os.path.join(sc_dir, 'vrmse_per_axis.png'), dpi=150, bbox_inches='tight')
                if fig_sd is not None:
                    fig_sd.savefig(os.path.join(sc_dir, 'dnn_sd.png'), dpi=150, bbox_inches='tight')
                plt.close('all')
                print(f'  [{mode}] scenario plots saved to {sc_dir}')

            # ----------------------------------------------------------------
            # Real-data only: Nadav baseline (independent of DNN mode).
            # Gated by config 'run_nadav' (default False) so it's skipped unless
            # explicitly requested.
            # ----------------------------------------------------------------
            if data_type == 'real' and config.get('run_nadav', False):
                _, out_err_n, out_bias_n, out_sd_n = lc_ins_dvl_real_nadav(
                    imu_t, dvl_t, gt_t, ne, dvl_cfg, kf_cfg)

                nadav_dir = os.path.join(output_dir, 'plots', f'{data_type}_results',
                                         f"test_{sc}_nadav_{run_stamp}")
                os.makedirs(nadav_dir, exist_ok=True)
                _write_config_txt(nadav_dir, sc, config, data_dir, kf_cfg, dvl_cfg,
                                  extra_lines=[f"model : Nadav EKF baseline",
                                               f"data  : {data_type}"])

                fig_pos_n, fig_vel_n = plot_pos_vel_two_runs(
                    out_err_n, out_sd_n,
                    scenario=sc, arch=arch, base_label='Nadav EKF')
                fig_att_n, fig_ba_n, fig_bg_n = plot_att_bias_two_runs(
                    out_err_n, out_sd_n, out_bias_n,
                    scenario=sc, arch=arch, base_label='Nadav EKF')

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
        'real_test_files': ['trajectory7', 'trajectory9'],
        # Number of validation trajectories carved from the non-test remainder
        # (the rest go to train). None → default 80/20 split. Clamped to keep ≥1
        # train. With 11 non-test files, 3 → train=8/val=3.
        'real_val_count': 2,

        # Cut the last N seconds off each test trajectory, by position:
        # [0] → first test trajectory, [1] → second, etc. 0 (or a missing entry)
        # = no trim. Applies to whatever the active test split is (sim or real).
        'test_trim_end_seconds': [0, 0],

        'split_seed': 42,                   # RNG seed for reproducible auto-splits

        # Mode flags
        'train_data': True,                # train DNN on the train split
        'test_data':  True,                 # run EKF baseline + DNN on the test split
        'sim_test_first_only': True,        # sim only: limit test loop to test_files[:1]
        'run_nadav': False,                  # real only: also run+save the Nadav EKF baseline

        # Load DVL_<name>_noised.csv (generated by noise_dvl.py) instead of the
        # raw DVL_<name>.csv, for both training and test. Requires the _noised
        # files to exist in the data dir (currently only real_data/).
        'use_noised_dvl': False,

        'trim_start_seconds': 50,           # sim only

        'dnn_config': {
            'arch':        'tcn',          # 'lstm' | 'gru' | 'mlp' | 'bilstm' | 'bigru' | 'tcn' | 'transformer'
            'window_size':  10,             # DVL epochs per input window (~10s)
            'hidden_size':  64,
            'num_layers':    2,
            'batch_size':   32,
            'epochs':      100,
            'lr':          3e-4,           # lowered from 1e-3 (noisy/early-overfit val)
            # LR schedule for both DNNs: 'cosine' anneals lr base→~0 over 'epochs'
            # (smoother, lower val minimum); 'none' = constant lr (old behaviour).
            'lr_scheduler': 'cosine',
            'seed':         42,             # RNG seed for weight init / shuffle / dropout. Same seed + same config => identical model. Change it to sample a different random run
            'tag':         'traj_7_9',              # free-text suffix on the model filename (e.g. 'v2', 'tuned'). Empty = no suffix. The lr token (e.g. 'lr3' for 1e-3) is auto-added before this tag. Do NOT put dnn_vel_SD here — it's not part of the model and is already recorded in the plot-folder name (sd<value>).
            # dnn_mode controls how the DNN is fused with the EKF:
            #   'sequential' — DVL EKF update first, then a second Kalman update with the DNN output.
            #                  Label = true_v_eb_n - est_v_eb_n  (residual AFTER the DVL update).
            #   'joint'      — DVL + DNN measurements are STACKED into a single 6-row Kalman update.
            #                  Label = true_v_eb_n - v_ekf_pre   (residual BEFORE any measurement update).
            # In test mode, both checkpoints (if both exist) are loaded and compared
            # automatically — each mode writes plots into its own folder.
            'dnn_mode':    'sequential',
            # loss_mode controls the TRAINING objective:
            #   'mse'    — plain MSE on the normalised velocity-error label (default).
            #   'filter' — filter-aware loss on the NED velocity AFTER the joint Kalman
            #              update (single-step). JOINT ONLY; the loss bakes the current
            #              dnn_vel_SD, so train at the SD you intend to test at. Encoded
            #              in the checkpoint name as lfilter_sd<SD>.
            'loss_mode':   'mse',
            # Option B: also estimate & correct attitude error (6-dim output:
            # [velocity(3), euler(3)]). Targets the heading-driven position drift
            # that velocity-only correction can't fix. Joint-only; MSE loss only.
            # Name token 'att'; attitude trust set by dnn_att_SD in LC_KF_config_real.
            'dnn_correct_attitude': False,
            # (sequential drift-loss run: loss_mode='filter' + filter_horizon>1 trains
            #  the velocity-only sequential model to minimise accumulated drift → PRMSE.)
            # filter_lambda weights the filter-aware term vs the velocity-error
            # anchor in the blended loss: L = ||corr-(v_true-v_pre)||^2 + lambda*||v_post-v_true||^2.
            # lambda=0 == plain MSE (safe floor); raise for more filter influence.
            'filter_lambda': 1.0,
            # Multi-step (drift) loss: filter_horizon>1 trains over H consecutive epochs
            # and adds lambda_drift*||mean_h(v_post-v_true)||^2 — penalises the persistent
            # velocity bias that integrates into POSITION drift (targets PRMSE, not just VRMSE).
            # horizon=1 disables it (pure single-step). Name tokens: hz<H>_ld<drift>.
            'filter_horizon': 20,
            'filter_lambda_drift': 1.0,
            # Phase A (fv3): append 12-dim inter-epoch high-rate IMU aggregates
            # [Δθ(3), Δv(3), std(ω)(3), std(f)(3)] to the 15-dim feature → 27-dim,
            # to capture maneuvers the 1 Hz EKF state smooths away. False = fv2.
            # Distinct checkpoint name (fv3 token) — fv2 models are preserved.
            'imu_agg_features': False,
            # Which aggregates: 'full' = [Δθ,Δv,std(ω),std(f)] (12-dim, fv3);
            # 'dtheta' = just Δθ (3-dim, fv3dtheta) — the lean ablation.
            'imu_agg_set': 'dtheta',
            # Sequential DNN-correction application timing (inference-only, reuses
            # the same weights): 'epoch' = at the DVL update (current);
            # 'midway' = at the inter-DVL interval midpoint (t_k + interval/2).
            # Plot-folder token 'tmidway'; checkpoint name unchanged.
            'dnn_apply_timing': 'epoch',
            # ---- Phase 2: learn a per-epoch, per-axis dnn_vel_SD (sigma-net) ----
            # When True, after phase-1 training a second "uncertainty" net is
            # trained (frozen phase-1 model) that predicts the per-axis log-variance
            # of the velocity correction; saved next to the model as *_uncsd.pth.
            # Sequential mode only. To USE it at test, set use_learned_dnn_sd:True in
            # LC_KF_config_real. Requires train_data:True to (re)generate the file.
            'train_dnn_sd': True,
            # Phase-2 target: 'sigma' (learned per-axis dnn_vel_SD, for 'kalman' mode)
            # or 'fuse_gate' (learned scalar per-epoch fuse scale α for 'fuse' mode,
            # trained toward α* = clip(corr·label/‖corr‖², 0, fuse_gate_max)).
            'unc_target': 'fuse_gate',
            'fuse_gate_max': 0.5,       # upper clamp on the learned α
            # sigma-net hyper-params default to the phase-1 values above; override
            # here if desired: 'unc_arch','unc_hidden_size','unc_epochs','unc_lr'.
            # Input design (ablatable): which groups the sigma-net sees, whether to
            # append their scalar magnitudes, and the (reserved) corr skip path.
            'unc_feature_groups': list(_UNC_DEFAULT_GROUPS),   # drops 'euler' by default
            'unc_add_magnitudes': False,
            # Early stopping (both DNNs): restore the best-val checkpoint and stop
            # after es_patience epochs with no val improvement. Applies to phase-1
            # and phase-2; the sigma-net can use a tighter unc_es_patience. Set
            # early_stopping:False to train the full 'epochs' (old behaviour).
            'early_stopping': True,
            'es_patience':    25,    # more tolerance: wait 25 no-improve epochs
            'es_min_delta':   0.0,
            'es_min_epochs':  30,    # never stop before epoch 30 (floor for both nets)
            'unc_es_patience': 20,   # sigma-net overfits sooner, but give it room too
            # 'unc_es_min_epochs': 30,  # override the floor for the sigma-net only
            # Weight decay (L2) slows overfitting so the val minimum lands LATER —
            # this is what makes 'more tolerance' actually train longer usefully.
            # 0.0 = off. unc_weight_decay overrides it for the sigma-net.
            'weight_decay':      1e-3,   # raised from 1e-4 to delay the ~ep15 overfit
            'unc_weight_decay':  1e-3,
        },
    }

    # ========================================================================
    # DVL CONFIG — interval between DVL epochs (s)
    # ========================================================================
    DVL_config_sim = {'epoch_interval': 0.2}
    DVL_config_real = {'epoch_interval': 1.002506}

    # ========================================================================
    # KALMAN FILTER CONFIG — REAL DATA
    # ========================================================================
    LC_KF_config_real = {

        # Colleague conventional EKF: P0 = [0.2 m/s, 5 deg, 30 mg, 30 deg/h]
        'init_att_unc': np.deg2rad(2.0), # deg
        'init_vel_unc': 0.1, #m/s
        'init_pos_unc': 0.5, # m
        'init_b_a_unc': 1.0e-4, #micro-g
        'init_b_g_unc': np.deg2rad(30.0) / 3600, # deg/h

        'gyro_noise_PSD':  1.0e-10, # (rad/s)^2/HZ  — matched to colleague (sigma_gyro=1e-4)
        'accel_noise_PSD': 9.0e-3, # (m/s^2)^2/HZ — matched to colleague (sigma_acc=3e-2)

        'accel_bias_PSD': 9e-5,  # matched to colleague (sigma_ba=3e-4)
        'gyro_bias_PSD':  1e-14, # matched to colleague (sigma_bg=1e-6)

        # 'init_att_unc': np.deg2rad(2.0), # deg
        # 'init_vel_unc': 0.1, #m/s
        # 'init_pos_unc': 0.5, # m
        # 'init_b_a_unc': 1.0e-4, #micro-g
        # 'init_b_g_unc': np.deg2rad(30.0) / 3600, # deg/h
        #
        # 'gyro_noise_PSD':  1.0e-10, # (rad/s)^2/HZ  — matched to colleague (sigma_gyro=1e-4)
        # 'accel_noise_PSD': 9.0e-3, # (m/s^2)^2/HZ — matched to colleague (sigma_acc=3e-2)
        #
        # 'accel_bias_PSD': 9e-2,  # matched to colleague (sigma_ba=3e-4)
        # 'gyro_bias_PSD':  1e-10, # matched to colleague (sigma_bg=1e-6)

        # 'accel_bias_PSD': 9.0e-8,  # matched to colleague (sigma_ba=3e-4)
        # 'gyro_bias_PSD': 1.0e-12,  # matched to colleague (sigma_bg=1e-6)

        'vel_meas_SD': 0.02, #m/s
        # dnn_vel_SD raised from 0.5 to down-weight the DNN in joint mode: at 0.5
        # the DNN had equal trust to the DVL, and since the DNN measurement is
        # correlated with the DVL (it takes dvl_v + innovation as inputs) the
        # joint update over-corrected → accel-bias/attitude runaway → P overflow
        # → SVD crash. Larger SD = less DNN weight = stable. Tune via sweep
        # (2.0 → 5.0 → 10.0); EKF inference param only, no retraining needed.
        # THIS is the line to edit when sweeping dnn_vel_SD for real data — the
        # folder name's sd<value> token is generated from this value.
        # NOTE: filter-loss models BAKE this SD (checkpoint name carries sd<value>),
        # so changing it requires retraining (cheap — no re-collection). 1.5 diverged
        # at inference (SVD); 2.5 is a stable starting point, sweep down toward 2.0.
        'dnn_vel_SD':  0.02,
        # Adaptive DNN gate (inference-only, no retraining): scale the correction by
        # g = m^2/(m^2+tau^2), m=||innovation||. Suppresses the DNN where EKF<->DVL
        # agree (accurate trajectories). Measured mean ||innov||: traj4~0.097 (help),
        # traj13~0.030 (harm). tau between them gates traj13 off but keeps traj4.
        # None/0 = gate off. Sweep this freely with test-only runs.
        # dnn_gate_pow: steepness of the on/off transition (higher = sharper).
        # tau=0.07 = median baseline ||innovation|| over the 11 train/val trajectories
        # (principled, not tuned on the test pair).
        'dnn_gate_tau': 0.07,
        'dnn_gate_pow': 4,
        # Option B attitude-pseudo-measurement noise SD (rad). Trust on the DNN's
        # attitude-error estimate in the joint update. Inference-only for MSE models
        # (not baked) → sweepable test-only. Smaller = trust the DNN heading more.
        'dnn_att_SD': np.deg2rad(180.0),
        # ---- Learned per-epoch, per-axis dnn_vel_SD (phase-2 sigma-net) ----
        # When True (and a *_uncsd.pth sibling exists for the sequential model),
        # the sequential DNN update uses R_dnn = diag((clip(scale*sigma))^2) with a
        # motion-dependent sigma from the sigma-net, instead of the constant
        # dnn_vel_SD above. False = current constant-SD behaviour (exact).
        # Inference-only knob: sweep freely, no retraining. dnn_sd_scale trims/boosts
        # the learned trust globally; min/max clamp it to a sane band [m/s].
        'use_learned_dnn_sd': False,
        'dnn_sd_scale': 1.0,
        'dnn_sd_min':   0.02,
        'dnn_sd_max':   0.002,
        # How the sequential DNN correction is applied:
        #   'kalman' — DNN runs a Kalman update that ALSO shrinks P (old default);
        #              dnn_sd/σ set the gain. Risks overconfident P (correlated meas).
        #   'inject' — DNN correction added straight to the velocity state
        #              (est_v += corr), full weight; P updated by the DVL only, so the
        #              DNN never shrinks P. dnn_sd / use_learned_dnn_sd are unused here
        #              (σ-net only needed if you still want the dnn_sd.png diagnostic).
        #   'fuse'   — DNN correction folded into the single DVL update: the DVL
        #              innovation is shifted by dnn_fuse_scale·Cᵀ·corr, then ONE Kalman
        #              update runs. P shrinks by the DVL only (honest); the DNN reaches
        #              the state through the DVL gain. dnn_sd/use_learned_dnn_sd unused.
        'dnn_apply_mode': 'fuse',
        # α on the DNN's innovation shift in 'fuse' mode. 1.0 = full fold; lower it
        # (0.5, 0.2) if traj3 destabilises (the DVL velocity gain is high). Used as
        # the CONSTANT α, and as the fallback during the gate's warm-up.
        'dnn_fuse_scale': 0.2,
        # When True (fuse mode) use the learned per-epoch gate α (phase-2 with
        # unc_target:'fuse_gate') instead of the constant dnn_fuse_scale. Needs the
        # *_uncsd.pth gate net (train_dnn_sd:True). False → constant α.
        'use_learned_fuse_scale': False,
    }

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
        # Raised to 2.0 to match real config: in joint mode a small dnn_vel_SD
        # over-trusts the (DVL-correlated) DNN and destabilizes the filter.
        'dnn_vel_SD': 2.0,
    }



    main({**user_config,
          'dvl_cfg_sim':  DVL_config_sim,
          'dvl_cfg_real': DVL_config_real,
          'kf_cfg_sim':   LC_KF_config_sim,
          'kf_cfg_real':  LC_KF_config_real})
