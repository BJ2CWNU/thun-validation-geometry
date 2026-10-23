# psi_mission1/src/03_p3_synthesize.py
"""
M1 Type A 배경 필드용 합성 훈련 데이터 생성.
Gaussian Process로 smooth background 필드 샘플링.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (PROC_DIR, RES_DIR, MATPLOTLIB_FONT,
                    TARGET_MEAN, TARGET_STD, TARGET_MIN, TARGET_MAX,
                    SYNTH_GP_LENGTH_SCALE, SYNTH_N_TRAIN, SYNTH_N_VAL,
                    UNET_N_INPUT_CHANNELS, RANDOM_SEED,
                    CV_N_SPLITS, CV_BUFFER_M, CV_BLOCK_AXIS)

plt.rcParams['font.family'] = MATPLOTLIB_FONT
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(RANDOM_SEED)

# ── 실제 측정 영역 파악 ────────────────────────────────────
df = pd.read_csv(PROC_DIR / 'measurements.csv')
E  = df['E_LV95'].values
N  = df['N_LV95'].values

E_min, E_max = E.min(), E.max()
N_min, N_max = N.min(), N.max()
E_span = E_max - E_min   # ~3 km
N_span = N_max - N_min   # ~1.3 km
print(f"측정 영역: E {E_span:.0f}m × N {N_span:.0f}m")
print(f"측정점 수: {len(df)}")

# ── GP 커널 (Squared Exponential) ─────────────────────────
def se_kernel(X1, X2, length_scale, sigma=1.0):
    """Squared Exponential (RBF) 커널."""
    D2 = cdist(X1, X2, 'sqeuclidean')
    return sigma**2 * np.exp(-0.5 * D2 / length_scale**2)

def sample_gp_field(grid_pts, length_scale, n_samples, seed=None):
    """
    GP에서 smooth 필드 샘플링.
    반환: zero-mean, unit-variance 샘플 (N, G).
    스케일링은 make_one_sample에서 수행.
    """
    rng = np.random.default_rng(seed)
    K   = se_kernel(grid_pts, grid_pts, length_scale)
    K  += 1e-6 * np.eye(len(grid_pts))
    L   = np.linalg.cholesky(K)
    z   = rng.standard_normal((len(grid_pts), n_samples))
    samples = (L @ z).T   # (n_samples, G), zero-mean, unit-variance 근사

    # 샘플별 표준화 (zero-mean, unit-std) — [0,1] stretch 제거
    mu  = samples.mean(axis=1, keepdims=True)
    sig = samples.std(axis=1, keepdims=True) + 1e-12
    return (samples - mu) / sig   # 각 샘플: mean≈0, std≈1

# ── 합성 샘플 1개 생성 함수 ───────────────────────────────
def sample_meas_indices(M, E_coords, N_coords, rng,
                        block_prob=0.4, sparsity_range=(0.1, 0.5)):
    """
    block_prob 확률로 E축 슬랩 마스킹 (Spatial Block 누락 모의).
    나머지는 기존 random sparsity.
    """
    if rng.random() < block_prob:
        # Spatial block 제거: E축을 k개 슬랩으로 나눠 1~2개 제거
        k          = rng.integers(3, 6)          # 슬랩 수
        slab_edges = np.linspace(E_coords.min(), E_coords.max(), k+1)
        n_remove   = rng.integers(1, max(2, k//2))
        remove_idx = rng.choice(k, n_remove, replace=False)

        keep = np.ones(M, dtype=bool)
        for ri in remove_idx:
            in_slab = ((E_coords >= slab_edges[ri]) &
                       (E_coords <  slab_edges[ri+1]))
            keep[in_slab] = False

        # keep된 포인트 중 추가 랜덤 sparsity
        keep_idx = np.where(keep)[0]
        if len(keep_idx) > 5:
            n_keep = max(5, int(len(keep_idx) *
                                rng.uniform(*sparsity_range)))
            keep_idx = rng.choice(keep_idx, n_keep, replace=False)
        return keep_idx
    else:
        # 기존: 완전 랜덤
        sparsity = rng.uniform(*sparsity_range)
        n_meas   = max(5, int(M * sparsity))
        return rng.choice(M, n_meas, replace=False)


def make_one_sample(grid_pts, field_zscore, meas_pts_E, meas_pts_N,
                    sparsity_range=(0.1, 0.5), rng=None):
    """
    Args:
        field_zscore: (G,) zero-mean unit-std GP 샘플
    """
    if rng is None:
        rng = np.random.default_rng()

    G = len(grid_pts)
    M = len(meas_pts_E)

    # ── 핵심 수정: 실측 분포와 동일한 std·mean으로 스케일 ──
    # 샘플별로 std를 TARGET_STD 근처에서 랜덤 샘플링 (±30%)
    # → 훈련 분포가 실측 std와 자연스럽게 일치
    std_factor = rng.uniform(0.7, 1.3)
    field_scaled = field_zscore * (TARGET_STD * std_factor) + TARGET_MEAN

    # mean도 소폭 변동 (측정 지역 내 배경 레벨 변동 반영)
    mean_shift = rng.uniform(-TARGET_STD, TARGET_STD)
    field_scaled = field_scaled + mean_shift

    # 물리적 허용 범위로 클리핑 (음수 방지)
    field_scaled = np.clip(field_scaled, TARGET_MIN * 0.5, TARGET_MAX * 1.5)

    # ── 측정점 샘플링: 40% 확률로 spatial-block 누락 ──
    meas_idx = sample_meas_indices(M, meas_pts_E, meas_pts_N, rng,
                                   block_prob=0.4,
                                   sparsity_range=sparsity_range)
    n_meas   = len(meas_idx)
    sparsity = n_meas / M

    meas_E = meas_pts_E[meas_idx]
    meas_N = meas_pts_N[meas_idx]

    meas_coords = np.stack([meas_E, meas_N], axis=1)
    dist_mat    = cdist(meas_coords, grid_pts)
    nn_idx      = dist_mat.argmin(axis=1)
    meas_vals   = field_scaled[nn_idx]

    ch_sparse = np.zeros(G)
    ch_sparse[nn_idx] = meas_vals

    dist_to_meas = cdist(grid_pts, meas_coords)
    dist_to_meas = np.maximum(dist_to_meas, 1e-6)
    w     = 1.0 / dist_to_meas**2
    ch_idw = (w * meas_vals[np.newaxis, :]).sum(axis=1) / w.sum(axis=1)

    ch_prior = np.full(G, TARGET_MEAN)
    ch_mask  = np.zeros(G);  ch_mask[nn_idx] = 1.0
    ch_water = np.zeros(G)   # TODO: DEM 기반 수역 마스크

    return {
        'ch_sparse': ch_sparse, 'ch_idw': ch_idw,
        'ch_prior':  ch_prior,  'ch_mask': ch_mask,
        'ch_water':  ch_water,  'label':   field_scaled,
        'sparsity':  sparsity,  'n_meas':  n_meas,
    }

# ── 그리드 설정 ───────────────────────────────────────────
GRID_RES = 50   # m (그리드 해상도)
e_grid   = np.arange(E_min, E_max + GRID_RES, GRID_RES)
n_grid   = np.arange(N_min, N_max + GRID_RES, GRID_RES)
ee, nn   = np.meshgrid(e_grid, n_grid)
grid_pts = np.stack([ee.ravel(), nn.ravel()], axis=1)
G        = len(grid_pts)
print(f"그리드: {len(e_grid)}×{len(n_grid)} = {G}점  (해상도 {GRID_RES}m)")

# ── GP 필드 샘플링 (배치로 처리) ─────────────────────────
print(f"\nGP 필드 샘플링 중... (length_scale={SYNTH_GP_LENGTH_SCALE}m)")
print("(그리드가 크면 Cholesky 분해에 수 분 소요될 수 있음)")

BATCH    = 100
N_TOTAL  = SYNTH_N_TRAIN + SYNTH_N_VAL
n_batches = (N_TOTAL + BATCH - 1) // BATCH

all_fields = []
for b in range(n_batches):
    n = min(BATCH, N_TOTAL - b * BATCH)
    fields = sample_gp_field(grid_pts, SYNTH_GP_LENGTH_SCALE, n,
                             seed=RANDOM_SEED + b)
    all_fields.append(fields)
    if (b + 1) % 10 == 0:
        print(f"  {(b+1)*BATCH}/{N_TOTAL} 샘플 완료")

all_fields = np.vstack(all_fields)   # (N_TOTAL, G)
print(f"GP 샘플링 완료: {all_fields.shape}")

# ── 합성 데이터셋 구성 ────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

def build_dataset(fields, meas_E, meas_N, n_samples, rng):
    channels = []
    labels   = []
    sparsities = []
    for i in range(n_samples):
        s = make_one_sample(grid_pts, fields[i], meas_E, meas_N,
                            sparsity_range=(0.1, 0.6), rng=rng)
        ch = np.stack([s['ch_sparse'], s['ch_idw'], s['ch_prior'],
                       s['ch_mask'],  s['ch_water']], axis=0)
        channels.append(ch)
        labels.append(s['label'])
        sparsities.append(s['sparsity'])
    return (np.array(channels),   # (N, 5, G)
            np.array(labels),     # (N, G)
            np.array(sparsities)) # (N,)

print("\n훈련 데이터셋 구성 중...")
X_train, y_train, sp_train = build_dataset(
    all_fields[:SYNTH_N_TRAIN], E, N, SYNTH_N_TRAIN, rng)

print("검증 데이터셋 구성 중...")
X_val, y_val, sp_val = build_dataset(
    all_fields[SYNTH_N_TRAIN:], E, N, SYNTH_N_VAL, rng)

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
print(f"sparsity train: {sp_train.mean():.2f} ± {sp_train.std():.2f}")

# ── 저장 ─────────────────────────────────────────────────
out = PROC_DIR / 'synthetic_m1.npz'
np.savez_compressed(out,
    X_train=X_train, y_train=y_train, sp_train=sp_train,
    X_val=X_val,     y_val=y_val,     sp_val=sp_val,
    grid_pts=grid_pts, e_grid=e_grid, n_grid=n_grid,
    meta=np.array([SYNTH_GP_LENGTH_SCALE, GRID_RES, G]))
print(f"\n저장 완료: {out}  ({out.stat().st_size/1e6:.1f} MB)")

# ── 시각화 (샘플 4개 확인) ────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i in range(4):
    z = y_train[i].reshape(len(n_grid), len(e_grid))
    im = axes[0, i].imshow(z, origin='lower', cmap='RdYlGn_r',
                           extent=[E_min, E_max, N_min, N_max],
                           aspect='auto')
    axes[0, i].set_title(f'샘플 {i+1} — label\n(sparsity={sp_train[i]:.2f})')
    plt.colorbar(im, ax=axes[0, i], label='μSv/h')

    z2 = X_train[i, 1].reshape(len(n_grid), len(e_grid))
    im2 = axes[1, i].imshow(z2, origin='lower', cmap='RdYlGn_r',
                             extent=[E_min, E_max, N_min, N_max],
                             aspect='auto')
    axes[1, i].set_title(f'샘플 {i+1} — IDW 채널')
    plt.colorbar(im2, ax=axes[1, i], label='μSv/h')

plt.suptitle('PSI Mission 1 — P3 합성 데이터 시각화 (smooth GP)', fontsize=13)
plt.tight_layout()
fig.savefig(RES_DIR / '03_p3_synthetic_check.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"시각화 저장: 03_p3_synthetic_check.png")
