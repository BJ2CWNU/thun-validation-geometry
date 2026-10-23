# psi_mission1/src/02_p4_baselines.py
"""
P4 Baseline 비교: IDW vs Ordinary Kriging
- Spatial Block CV (E축 등분)
- 테스트 포인트 BUFFER_M 이내 훈련점 제외 → leakage 방지
- 잔차 공간 분포 시각화
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MEAS_CSV, RES_DIR, TARGET_COL,
    CV_N_SPLITS, CV_BUFFER_M, CV_BLOCK_AXIS,
    RANDOM_SEED, IDW_POWER, KRIGING_VARIOGRAM,
    MATPLOTLIB_FONT,
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = MATPLOTLIB_FONT
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

np.random.seed(RANDOM_SEED)

# ── 데이터 로드 ────────────────────────────────────────────
df = pd.read_csv(MEAS_CSV)
df = df.dropna(subset=[TARGET_COL, 'E_LV95', 'N_LV95']).reset_index(drop=True)
X = df[['E_LV95', 'N_LV95']].values   # (N, 2)
y = df[TARGET_COL].values             # (N,)
N = len(df)
print(f"유효 측정점: {N}개  |  타깃: {TARGET_COL}")

# ── Spatial Block CV 분할 ──────────────────────────────────
def make_spatial_blocks(X, n_splits, axis='E'):
    """axis='E' → 0번 컬럼 정렬, 'N' → 1번 컬럼 정렬"""
    col = 0 if axis == 'E' else 1
    order = np.argsort(X[:, col])
    return np.array_split(order, n_splits)

blocks = make_spatial_blocks(X, CV_N_SPLITS, axis=CV_BLOCK_AXIS)
fold_ids = np.empty(N, dtype=int)
for i, b in enumerate(blocks):
    fold_ids[b] = i

# ── IDW ───────────────────────────────────────────────────
def idw_predict(X_train, y_train, X_test, power=IDW_POWER, buffer_m=CV_BUFFER_M):
    """test 포인트별 IDW. buffer_m 이내 훈련점 제외 (leakage 방지)."""
    preds = np.empty(len(X_test))
    for i, xt in enumerate(X_test):
        dist = np.linalg.norm(X_train - xt, axis=1)
        mask = dist > buffer_m
        if mask.sum() == 0:
            mask = np.ones(len(dist), dtype=bool)
        d = dist[mask]
        w = 1.0 / np.maximum(d, 1e-6) ** power
        preds[i] = np.dot(w, y_train[mask]) / w.sum()
    return preds

# ── Kriging ───────────────────────────────────────────────
def kriging_predict(X_train, y_train, X_test):
    ok = OrdinaryKriging(
        X_train[:, 0], X_train[:, 1], y_train,
        variogram_model=KRIGING_VARIOGRAM,
        verbose=False, enable_plotting=False,
    )
    z, _ = ok.execute('points', X_test[:, 0], X_test[:, 1])
    return z.data

# ── Spatial Block CV 실행 ─────────────────────────────────
results = {'fold': [], 'n_test': [],
           'rmse_idw': [], 'rmse_kriging': [],
           'mae_idw': [], 'mae_kriging': []}

print(f"\n{'Fold':>4}  {'N_test':>6}  {'RMSE_IDW':>10}  {'RMSE_Kriging':>12}  {'개선율':>7}")
print("-" * 55)

all_pred_idw     = np.empty(N)
all_pred_kriging = np.empty(N)

for fold in range(CV_N_SPLITS):
    test_idx  = np.where(fold_ids == fold)[0]
    train_idx = np.where(fold_ids != fold)[0]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx],  y[test_idx]

    p_idw = idw_predict(X_tr, y_tr, X_te)
    p_kri = kriging_predict(X_tr, y_tr, X_te)

    rmse_idw = np.sqrt(np.mean((p_idw - y_te) ** 2))
    rmse_kri = np.sqrt(np.mean((p_kri - y_te) ** 2))
    mae_idw  = np.mean(np.abs(p_idw - y_te))
    mae_kri  = np.mean(np.abs(p_kri - y_te))
    impr     = (rmse_idw - rmse_kri) / rmse_idw * 100

    all_pred_idw[test_idx]     = p_idw
    all_pred_kriging[test_idx] = p_kri

    results['fold'].append(fold)
    results['n_test'].append(len(test_idx))
    results['rmse_idw'].append(rmse_idw)
    results['rmse_kriging'].append(rmse_kri)
    results['mae_idw'].append(mae_idw)
    results['mae_kriging'].append(mae_kri)

    print(f"{fold:>4}  {len(test_idx):>6}  {rmse_idw:>10.5f}  {rmse_kri:>12.5f}  {impr:>6.1f}%")

# 전체 요약
res_df = pd.DataFrame(results)
print("\n[전체 평균 ± std]")
for col in ['rmse_idw', 'rmse_kriging', 'mae_idw', 'mae_kriging']:
    print(f"  {col:20s}: {res_df[col].mean():.5f} ± {res_df[col].std():.5f}")

overall_impr = (res_df['rmse_idw'].mean() - res_df['rmse_kriging'].mean()) \
               / res_df['rmse_idw'].mean() * 100
print(f"\n  Kriging vs IDW RMSE 개선율: {overall_impr:.1f}%")

# ── 잔차 공간 분포 시각화 ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sc0 = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlGn_r', s=12, alpha=0.8)
plt.colorbar(sc0, ax=axes[0], label='μSv/h')
axes[0].set_title(f'(a) 측정값 {TARGET_COL}')
axes[0].set_aspect('equal')

res_idw = all_pred_idw - y
sc1 = axes[1].scatter(X[:, 0], X[:, 1], c=res_idw,
                      cmap='RdBu_r', s=12, alpha=0.8,
                      vmin=-3 * res_idw.std(), vmax=3 * res_idw.std())
plt.colorbar(sc1, ax=axes[1], label='잔차 (μSv/h)')
axes[1].set_title(f'(b) IDW 잔차  RMSE={res_df["rmse_idw"].mean():.5f}')
axes[1].set_aspect('equal')

res_kri = all_pred_kriging - y
sc2 = axes[2].scatter(X[:, 0], X[:, 1], c=res_kri,
                      cmap='RdBu_r', s=12, alpha=0.8,
                      vmin=-3 * res_kri.std(), vmax=3 * res_kri.std())
plt.colorbar(sc2, ax=axes[2], label='잔차 (μSv/h)')
axes[2].set_title(f'(c) Kriging 잔차  RMSE={res_df["rmse_kriging"].mean():.5f}')
axes[2].set_aspect('equal')

plt.suptitle('PSI Mission 1 — P4 Baseline CV 결과', fontsize=13)
plt.tight_layout()
fig.savefig(RES_DIR / '02_p4_baselines.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 결과 저장 ─────────────────────────────────────────────
res_df.to_csv(RES_DIR / 'p4_cv_results.csv', index=False)
pd.DataFrame({
    'E_LV95': X[:, 0], 'N_LV95': X[:, 1],
    'y_true': y,
    'pred_idw': all_pred_idw,
    'pred_kriging': all_pred_kriging,
    'resid_idw': res_idw,
    'resid_kriging': res_kri,
    'fold': fold_ids,
}).to_csv(RES_DIR / 'p4_predictions.csv', index=False)

print("\n저장 완료: 02_p4_baselines.png, p4_cv_results.csv, p4_predictions.csv")
