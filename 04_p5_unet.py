# psi_mission1/src/04_p5_unet.py
"""
PSI Mission 1 — Physics-Aware U-Net 훈련 및 Spatial Block CV 평가.
입력: (N, 5, H, W)  출력: (N, 1, H, W)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import (PROC_DIR, RES_DIR, MATPLOTLIB_FONT,
                    TARGET_COL, TARGET_MEAN, TARGET_STD,
                    MU_AIR, ALTITUDE_REF_M,
                    CV_N_SPLITS, CV_BUFFER_M,
                    UNET_N_INPUT_CHANNELS, RANDOM_SEED)

plt.rcParams['font.family'] = MATPLOTLIB_FONT
plt.rcParams['axes.unicode_minus'] = False
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── U-Net 아키텍처 ────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class PhysicsAwareUNet(nn.Module):
    """
    Residual U-Net: 출력 = ch_idw + δ
    - δ ∈ [-2σ, +2σ]  (Tanh × delta_scale)
    - IDW 베이스라인에 앵커링되어 외삽 안정성 확보
    Physics hook: ch_sparse(ch0)에 고도 감쇠 보정.
    Forward는 임의 H/W 대응 (multiple-of-8 zero-pad → crop).
    """
    def __init__(self, in_ch=5, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.pool  = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        self.up3   = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3  = ConvBlock(base_ch*8, base_ch*4)
        self.up2   = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2  = ConvBlock(base_ch*4, base_ch*2)
        self.up1   = nn.ConvTranspose2d(base_ch*2, base_ch,   2, stride=2)
        self.dec1  = ConvBlock(base_ch*2, base_ch)
        # δ 예측: tanh → [-1,1] → 스케일
        self.out   = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Tanh()
        )
        self.delta_scale = 2.0 * TARGET_STD   # ±2σ 범위로 클리핑

    def physics_correction(self, x, alt_map):
        """
        ch_sparse (ch0)에 고도 감쇠 보정 적용.
        I_corrected = I_measured × exp(mu_air × (alt - alt_ref))
        """
        correction = torch.exp(
            torch.tensor(MU_AIR, device=x.device, dtype=x.dtype) *
            (alt_map - ALTITUDE_REF_M)
        )
        x = x.clone()
        x[:, 0:1] = x[:, 0:1] * correction
        return x

    def forward(self, x, alt_map=None):
        if alt_map is not None:
            x = self.physics_correction(x, alt_map)

        # ch_idw 베이스라인 (un-padded 원본에서 추출)
        idw_baseline = x[:, 1:2]

        # multiple-of-8 zero-pad → 추론 → crop
        h, w = x.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        x_in = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x

        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out(d1)

        if pad_h or pad_w:
            out = out[..., :h, :w]

        delta = out * self.delta_scale            # [-2σ, +2σ]
        return idw_baseline + delta               # IDW + 보정


# ── 데이터 로드 ────────────────────────────────────────────
print("\n합성 데이터 로드 중...")
data = np.load(PROC_DIR / 'synthetic_m1.npz', allow_pickle=True)

e_grid = data['e_grid']
n_grid = data['n_grid']
H, W   = len(n_grid), len(e_grid)
print(f"그리드: H={H}, W={W}")

def to_hw(arr_flat, H, W):
    """(N, C, G) → (N, C, H, W)"""
    N, C, G = arr_flat.shape
    return arr_flat.reshape(N, C, H, W)

X_train = to_hw(data['X_train'].astype(np.float32), H, W)
y_train = data['y_train'].astype(np.float32).reshape(-1, 1, H, W)
X_val   = to_hw(data['X_val'].astype(np.float32),   H, W)
y_val   = data['y_val'].astype(np.float32).reshape(-1, 1,   H, W)

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"label std (train): {y_train.std(axis=(1,2,3)).mean():.5f}  "
      f"(목표 ~{TARGET_STD:.5f})")

# ── 훈련 ──────────────────────────────────────────────────
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 1e-3
PATIENCE   = 10   # early stopping

model = PhysicsAwareUNet(in_ch=UNET_N_INPUT_CHANNELS).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n모델 파라미터: {n_params:,}")

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss()

train_ds = TensorDataset(torch.from_numpy(X_train),
                         torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),
                         torch.from_numpy(y_val))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

best_val_loss = float('inf')
no_improve    = 0
train_losses, val_losses = [], []

print(f"\n훈련 시작 (epochs={EPOCHS}, batch={BATCH_SIZE}, device={DEVICE})")
for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    tr_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)          # alt_map=None (합성 데이터는 고도 채널 없음)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item() * len(xb)
    tr_loss /= len(train_ds)

    # Validate
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            va_loss += criterion(pred, yb).item() * len(xb)
    va_loss /= len(val_ds)

    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    scheduler.step(va_loss)

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}: train={tr_loss:.6f}  val={va_loss:.6f}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        torch.save(model.state_dict(), PROC_DIR / 'unet_m1_best.pt')
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

print(f"\n최선 val_loss: {best_val_loss:.6f}")

# ── 실측 데이터에 대한 Spatial Block CV ───────────────────
print("\n실측 데이터 Spatial Block CV 평가 중...")
model.load_state_dict(torch.load(PROC_DIR / 'unet_m1_best.pt',
                                 map_location=DEVICE))
model.eval()

df_meas = pd.read_csv(PROC_DIR / 'measurements.csv')
df_meas = df_meas.dropna(subset=[TARGET_COL, 'E_LV95', 'N_LV95']).reset_index(drop=True)
X_coord = df_meas[['E_LV95', 'N_LV95']].values
y_true  = df_meas[TARGET_COL].values
ph_vals = df_meas['PH'].values if 'PH' in df_meas.columns else \
          np.full(len(df_meas), ALTITUDE_REF_M)

order    = np.argsort(X_coord[:, 0])
blocks   = np.array_split(order, CV_N_SPLITS)
fold_ids = np.empty(len(df_meas), dtype=int)
for i, b in enumerate(blocks):
    fold_ids[b] = i

def predict_unet_at_points(model, test_coords, train_coords, train_vals,
                            train_ph, grid_pts, e_grid, n_grid, device):
    """훈련 측정점으로 입력 채널 구성 → U-Net 예측 → test 좌표 nearest-grid 샘플링."""
    from scipy.spatial.distance import cdist as sp_cdist

    H_g, W_g = len(n_grid), len(e_grid)
    G        = H_g * W_g

    ch_sparse = np.zeros(G)
    ch_mask   = np.zeros(G)

    dist_tr = sp_cdist(train_coords, grid_pts)
    nn_idx  = dist_tr.argmin(axis=1)
    ch_sparse[nn_idx] = train_vals
    ch_mask[nn_idx]   = 1.0

    d2g = sp_cdist(grid_pts, train_coords)
    d2g = np.maximum(d2g, 1e-6)
    w   = 1.0 / d2g**2
    ch_idw   = (w * train_vals[np.newaxis, :]).sum(1) / w.sum(1)
    ch_prior = np.full(G, TARGET_MEAN)
    ch_water = np.zeros(G)

    inp = np.stack([ch_sparse, ch_idw, ch_prior, ch_mask, ch_water],
                   axis=0).reshape(1, 5, H_g, W_g).astype(np.float32)

    mean_alt = train_ph.mean() if len(train_ph) > 0 else ALTITUDE_REF_M
    alt_map  = torch.full((1, 1, H_g, W_g), mean_alt, dtype=torch.float32)

    with torch.no_grad():
        out = model(torch.from_numpy(inp).to(device),
                    alt_map.to(device))           # (1,1,H,W)
    pred_grid = out.cpu().numpy().reshape(G)

    dist_te  = sp_cdist(test_coords, grid_pts)
    te_nn    = dist_te.argmin(axis=1)

    return pred_grid[te_nn]

# E/N 그리드 좌표 재구성
e_min, e_max = X_coord[:,0].min(), X_coord[:,0].max()
n_min, n_max = X_coord[:,1].min(), X_coord[:,1].max()
e_g = np.arange(e_min, e_max + 50, 50)
n_g = np.arange(n_min, n_max + 50, 50)
ee, nn_m = np.meshgrid(e_g, n_g)
gp = np.stack([ee.ravel(), nn_m.ravel()], axis=1)

cv_results = []
all_pred_unet = np.empty(len(df_meas))

print(f"{'Fold':>4}  {'N_test':>6}  {'RMSE_UNet':>10}  vs Kriging(0.00360)")
print("-" * 50)

for fold in range(CV_N_SPLITS):
    test_idx  = np.where(fold_ids == fold)[0]
    train_idx = np.where(fold_ids != fold)[0]

    preds = predict_unet_at_points(
        model,
        X_coord[test_idx], X_coord[train_idx],
        y_true[train_idx], ph_vals[train_idx],
        gp, e_g, n_g, DEVICE
    )
    all_pred_unet[test_idx] = preds

    rmse = np.sqrt(np.mean((preds - y_true[test_idx])**2))
    cv_results.append({'fold': fold, 'n_test': len(test_idx), 'rmse_unet': rmse})
    print(f"{fold:>4}  {len(test_idx):>6}  {rmse:>10.5f}")

cv_df = pd.DataFrame(cv_results)
print(f"\n평균 RMSE_UNet: {cv_df['rmse_unet'].mean():.5f} ± {cv_df['rmse_unet'].std():.5f}")
print(f"vs IDW:    0.00439  vs Kriging: 0.00360")

# ── 최종 시각화 ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(train_losses, label='train')
axes[0].plot(val_losses,   label='val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
axes[0].set_title('학습 곡선')
axes[0].legend()

sc1 = axes[1].scatter(X_coord[:,0], X_coord[:,1],
                      c=all_pred_unet, cmap='RdYlGn_r', s=12, alpha=0.8)
plt.colorbar(sc1, ax=axes[1], label='μSv/h')
axes[1].set_title('U-Net 예측값')
axes[1].set_aspect('equal')

resid = all_pred_unet - y_true
sc2 = axes[2].scatter(X_coord[:,0], X_coord[:,1],
                      c=resid, cmap='RdBu_r', s=12, alpha=0.8,
                      vmin=-3*resid.std(), vmax=3*resid.std())
plt.colorbar(sc2, ax=axes[2], label='잔차 (μSv/h)')
axes[2].set_title(f'U-Net 잔차  RMSE={cv_df["rmse_unet"].mean():.5f}')
axes[2].set_aspect('equal')

plt.suptitle('PSI Mission 1 — P5 U-Net 결과', fontsize=13)
plt.tight_layout()
fig.savefig(RES_DIR / '04_p5_unet.png', dpi=150, bbox_inches='tight')
plt.close()

cv_df.to_csv(RES_DIR / 'p5_cv_results.csv', index=False)
pd.DataFrame({'E': X_coord[:,0], 'N': X_coord[:,1],
              'y_true': y_true, 'pred_unet': all_pred_unet,
              'resid': resid}).to_csv(RES_DIR / 'p5_predictions.csv', index=False)
print(f"\n저장 완료: 04_p5_unet.png, p5_cv_results.csv, p5_predictions.csv")
