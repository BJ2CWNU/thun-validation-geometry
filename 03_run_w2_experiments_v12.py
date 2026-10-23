"""
03 — W2 Experiment Runner (v1.2 — buffer-matched IDW + Kriging baselines)

Day 3 Red Flag 1 (buffer-matched baseline) + Red Flag 2 (seed structure) 해소.

v1.1 → v1.2:
  - U-Net inference 변경 없음 (학습 0회, v3.1 weights 재사용)
  - 각 cell에서 추가로 IDW_matched, Kriging_matched 평가 (동일 buffer-filtered train_df)
  - paired delta per cell
  - train_retention_rate 산출
  - 결과 dir: w2_block_cv_v12/, w2_loo_fl_v12/ (v1.1 결과 보존)

Hyperparameter 매칭 (v3.1 baselines 정확 재현):
  IDW    : all-points IDW, power=2, no internal buffer (외부 buffer만)
  Kriging: OrdinaryKriging, variogram_model='spherical' (config.py KRIGING_VARIOGRAM)
  v1.0 spec의 k=8 + 'exponential'은 v3.1과 불일치 — 정정함.

사용:
    python 03_run_w2_experiments_v12.py --protocol all
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist as sp_cdist

try:
    from pykrige.ok import OrdinaryKriging
    KRIGING_AVAILABLE = True
except ImportError:
    KRIGING_AVAILABLE = False
    print("[warn] PyKrige not available — Kriging baseline will be skipped")

PROJECT_ROOT = Path("C:/Users/BJ/Documents/radiation-mapping/psi_mission1")
DATA_PATH    = PROJECT_ROOT / "data/processed/measurements.csv"
WEIGHTS_PATH = PROJECT_ROOT / "data/processed/unet_m1_best.pt"
LINES_PATH   = PROJECT_ROOT / "data/processed/flight_lines.csv"
RESULTS_BLOCK = PROJECT_ROOT / "results/w2_block_cv_v12"
RESULTS_LOO   = PROJECT_ROOT / "results/w2_loo_fl_v12"
RESULTS_BLOCK.mkdir(parents=True, exist_ok=True)
RESULTS_LOO.mkdir(parents=True, exist_ok=True)

BUFFERS_M = [0, 58, 125, 250]   # v1.2: include 0 for explicit baseline reproduction
N_FOLDS = 5
RANDOM_SEED = 42

# v3.1 baseline matched hyperparameters
IDW_POWER = 2                          # config.py IDW_POWER
KRIGING_VARIOGRAM = 'spherical'        # config.py KRIGING_VARIOGRAM

INFERENCE_USE_ALT_CORRECTION = False   # G4 Scenario A

# ─── data ─────────────────────────────────────────────────
def load_measurements():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['DHSR_GND', 'E_LV95', 'N_LV95']).reset_index(drop=True)
    return df

def get_block_cv_folds(df, n_folds=5):
    """v3.1 protocol: equal-N via argsort + array_split."""
    order = np.argsort(df['E_LV95'].values)
    fold_id = np.zeros(len(df), dtype=int)
    for k, idx in enumerate(np.array_split(order, n_folds)):
        fold_id[idx] = k
    return fold_id

def apply_buffer(train_xy, test_xy, buffer_m):
    if buffer_m == 0:
        return np.ones(len(train_xy), dtype=bool)
    tree = cKDTree(test_xy)
    dist, _ = tree.query(train_xy, k=1)
    return dist > buffer_m

def load_flight_lines():
    return pd.read_csv(LINES_PATH)

# ─── U-Net (v3.1 unet_m1_best.pt 단일 인스턴스 캐시) ────────────
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import importlib.util, torch
    spec = importlib.util.spec_from_file_location(
        "_g4_unet", str(PROJECT_ROOT / "src" / "g4_audit_alt_ablation.py"))
    g4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g4)
    model = g4.PhysicsAwareUNet(in_ch=5).to(g4.DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=g4.DEVICE))
    model.eval()
    _MODEL = (model, g4)
    return _MODEL


def unet_predict(train_df, test_df, use_alt_correction=False):
    model, g4 = _get_model()
    device = g4.DEVICE
    train_xy   = train_df[['E_LV95', 'N_LV95']].values
    train_vals = train_df['DHSR_GND'].values.astype(np.float64)
    test_xy    = test_df[['E_LV95', 'N_LV95']].values
    train_ph   = (train_df['PH'].values if 'PH' in train_df.columns
                  else np.full(len(train_df), g4.ALTITUDE_REF_M))
    all_xy = np.vstack([train_xy, test_xy])
    e_min, e_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    n_min, n_max = all_xy[:, 1].min(), all_xy[:, 1].max()
    e_g = np.arange(e_min, e_max + 50, 50)
    n_g = np.arange(n_min, n_max + 50, 50)
    ee, nn_m = np.meshgrid(e_g, n_g)
    gp = np.stack([ee.ravel(), nn_m.ravel()], axis=1)
    return g4.predict_unet_at_points(
        model, test_xy, train_xy, train_vals, train_ph,
        gp, e_g, n_g, device,
        apply_alt_correction=use_alt_correction,
    )

# ─── IDW (v3.1 baseline matched: all-points, power=2, no internal buffer) ──
def idw_predict(train_df, test_df, power=IDW_POWER):
    """
    Buffer-matched IDW: all-points IDW with no INTERNAL buffer.
    외부 buffer는 호출 측에서 train_df 필터링으로 이미 적용됨.

    v3.1 02_p4_baselines.py:50의 buffer_m 인자는 leakage 방지 목적이었으나,
    W2 v1.2에서는 외부 apply_buffer로 동일 효과를 명시적으로 처리하므로 내부
    buffer는 0으로 (== 모든 train point 사용).
    """
    train_xy   = train_df[['E_LV95', 'N_LV95']].values
    test_xy    = test_df[['E_LV95', 'N_LV95']].values
    train_vals = train_df['DHSR_GND'].values
    d = sp_cdist(test_xy, train_xy)
    d = np.maximum(d, 1e-6)
    w = 1.0 / d**power
    pred = (w * train_vals[np.newaxis, :]).sum(1) / w.sum(1)
    return pred

# ─── Kriging (v3.1 matched: spherical variogram, all-points) ──────────
def kriging_predict(train_df, test_df, variogram_model=KRIGING_VARIOGRAM):
    if not KRIGING_AVAILABLE:
        return None
    train_xy   = train_df[['E_LV95', 'N_LV95']].values
    test_xy    = test_df[['E_LV95', 'N_LV95']].values
    train_vals = train_df['DHSR_GND'].values
    try:
        ok = OrdinaryKriging(
            train_xy[:, 0], train_xy[:, 1], train_vals,
            variogram_model=variogram_model,
            verbose=False, enable_plotting=False,
            pseudo_inv=True,
        )
        z, _ = ok.execute('points', test_xy[:, 0], test_xy[:, 1])
        return np.asarray(z).flatten()
    except Exception as e:
        print(f"  [warn] Kriging failed: {e}")
        return None

# ─── Metrics ───────────────────────────────────────────────
def compute_metrics(pred, obs, coords=None, k=5):
    pred = np.asarray(pred); obs = np.asarray(obs)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae = float(np.mean(np.abs(pred - obs)))
    pearson_r = float(np.corrcoef(pred, obs)[0, 1])
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = float(1.0 - np.sum((obs - pred) ** 2) / ss_tot) if ss_tot > 0 else float('nan')
    bias = float((pred - obs).mean())
    sigma_ratio = float(pred.std() / obs.std()) if obs.std() > 0 else float('nan')
    morans_i = float('nan')
    if coords is not None and len(coords) >= k + 1:
        r = pred - obs
        r = r - r.mean()
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1)
        idx = idx[:, 1:]
        n = len(r)
        num = sum(r[i] * r[idx[i]].sum() / k for i in range(n))
        denom = (r ** 2).sum()
        morans_i = float(num / denom) if denom > 0 else float('nan')
    return {"rmse": rmse, "mae": mae,
            "pearson_r": pearson_r, "r_squared": r2,
            "bias": bias, "sigma_ratio": sigma_ratio,
            "morans_i_residual": morans_i}

# ─── Cell runners ──────────────────────────────────────────
def _eval_three(train_df, test_df):
    """U-Net, IDW_matched, Kriging_matched 동시 평가."""
    obs = test_df['DHSR_GND'].values.astype(np.float64)
    coords = test_df[['E_LV95', 'N_LV95']].values

    t0 = time.time()
    pred_u = unet_predict(train_df, test_df, use_alt_correction=INFERENCE_USE_ALT_CORRECTION)
    m_u = compute_metrics(pred_u, obs, coords=coords)
    el_u = time.time() - t0

    t0 = time.time()
    pred_i = idw_predict(train_df, test_df, power=IDW_POWER)
    m_i = compute_metrics(pred_i, obs, coords=coords)
    el_i = time.time() - t0

    t0 = time.time()
    pred_k = kriging_predict(train_df, test_df, variogram_model=KRIGING_VARIOGRAM)
    if pred_k is not None:
        m_k = compute_metrics(pred_k, obs, coords=coords)
    else:
        m_k = None
    el_k = time.time() - t0

    return m_u, m_i, m_k, dict(unet=el_u, idw=el_i, kriging=el_k)


def run_block_cv(buffer_m, fold_id, df, fold_arr):
    test_mask = (fold_arr == fold_id)
    test_df = df[test_mask].reset_index(drop=True)
    train_df_raw = df[~test_mask].reset_index(drop=True)
    n_train_raw = len(train_df_raw)

    train_xy = train_df_raw[['E_LV95', 'N_LV95']].values
    test_xy  = test_df[['E_LV95', 'N_LV95']].values
    keep = apply_buffer(train_xy, test_xy, buffer_m)
    train_df = train_df_raw[keep].reset_index(drop=True)

    m_u, m_i, m_k, el = _eval_three(train_df, test_df)
    d_ui = m_u['rmse'] - m_i['rmse']
    d_uk = (m_u['rmse'] - m_k['rmse']) if m_k is not None else None

    result = {
        "protocol": "block_cv",
        "buffer_m": buffer_m,
        "fold_id": int(fold_id),
        "random_seed": RANDOM_SEED,
        "n_train_raw": int(n_train_raw),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_train_excluded_by_buffer": int((~keep).sum()),
        "train_retention_rate": float(len(train_df) / n_train_raw),
        "hyperparams": {
            "idw_power": IDW_POWER,
            "idw_internal_buffer": 0,
            "kriging_variogram": KRIGING_VARIOGRAM,
            "alt_correction": INFERENCE_USE_ALT_CORRECTION,
        },
        "metrics_unet": m_u,
        "metrics_idw_matched": m_i,
        "metrics_kriging_matched": m_k,
        "delta_unet_minus_idw_matched_rmse": float(d_ui),
        "delta_unet_minus_kriging_matched_rmse": (float(d_uk) if d_uk is not None else None),
        "elapsed_seconds": el,
    }
    out = RESULTS_BLOCK / f"runs_b{buffer_m:03d}_f{fold_id}.json"
    out.write_text(json.dumps(result, indent=2))
    krig_str = f"K={m_k['rmse']:.5f}" if m_k else "K=skip"
    print(f"  ✓ b{buffer_m:03d} f{fold_id}: "
          f"U={m_u['rmse']:.5f} I={m_i['rmse']:.5f} {krig_str} "
          f"retain={result['train_retention_rate']:.2f}")
    return result


def run_loo_fl(line_id, df, line_arr):
    test_mask = (line_arr == line_id)
    test_df = df[test_mask].reset_index(drop=True)
    train_df = df[~test_mask].reset_index(drop=True)
    m_u, m_i, m_k, el = _eval_three(train_df, test_df)
    d_ui = m_u['rmse'] - m_i['rmse']
    d_uk = (m_u['rmse'] - m_k['rmse']) if m_k is not None else None
    result = {
        "protocol": "loo_fl",
        "line_id": int(line_id),
        "random_seed": RANDOM_SEED,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "hyperparams": {"idw_power": IDW_POWER, "idw_internal_buffer": 0,
                        "kriging_variogram": KRIGING_VARIOGRAM,
                        "alt_correction": INFERENCE_USE_ALT_CORRECTION},
        "metrics_unet": m_u,
        "metrics_idw_matched": m_i,
        "metrics_kriging_matched": m_k,
        "delta_unet_minus_idw_matched_rmse": float(d_ui),
        "delta_unet_minus_kriging_matched_rmse": (float(d_uk) if d_uk is not None else None),
        "elapsed_seconds": el,
    }
    out = RESULTS_LOO / f"runs_line{line_id:02d}.json"
    out.write_text(json.dumps(result, indent=2))
    krig_str = f"K={m_k['rmse']:.5f}" if m_k else "K=skip"
    print(f"  ✓ line{line_id:02d}: U={m_u['rmse']:.5f} I={m_i['rmse']:.5f} {krig_str}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["block_cv", "loo_fl", "all"], required=True)
    ap.add_argument("--buffer", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--line", type=int, default=None)
    args = ap.parse_args()

    df = load_measurements()
    print(f"Loaded {len(df)} measurement points | RANDOM_SEED={RANDOM_SEED} | "
          f"IDW power={IDW_POWER} buf=0 (external) | Kriging={KRIGING_VARIOGRAM}")

    if args.protocol in ("block_cv", "all"):
        fold_arr = get_block_cv_folds(df, N_FOLDS)
        if args.buffer is not None and args.fold is not None:
            run_block_cv(args.buffer, args.fold, df, fold_arr)
        else:
            print(f"\nBlock CV: {len(BUFFERS_M)} buffers × {N_FOLDS} folds = "
                  f"{len(BUFFERS_M)*N_FOLDS} cells (incl. buffer=0 baseline)")
            for b in BUFFERS_M:
                for f in range(N_FOLDS):
                    run_block_cv(b, f, df, fold_arr)

    if args.protocol in ("loo_fl", "all"):
        lines_df = load_flight_lines()
        line_arr = lines_df['line_id'].values
        n_lines = len(np.unique(line_arr))
        if args.line is not None:
            run_loo_fl(args.line, df, line_arr)
        else:
            print(f"\nLOO-FL: {n_lines} cells")
            for l in range(n_lines):
                run_loo_fl(l, df, line_arr)

    if args.protocol == "all":
        n_lines = len(np.unique(pd.read_csv(LINES_PATH)['line_id'].values))
        print(f"\n✓ v1.2 complete: {len(BUFFERS_M)*N_FOLDS} block + {n_lines} loo = "
              f"{len(BUFFERS_M)*N_FOLDS + n_lines} cells × 3 methods")


if __name__ == "__main__":
    main()
