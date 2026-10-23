# psi_mission1/src/config.py
"""
PSI Mission 1 — 전체 파이프라인 공유 상수
작성: 2026-05-02
"""
from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
BASE_DIR  = Path(r"C:\Users\BJ\Documents\radiation-mapping\psi_mission1")
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
RES_DIR   = BASE_DIR / "results"

ERS_FILE  = RAW_DIR / "Mission1_CHE1_AGS_CH_newBG_NewC.ERS"
MEAS_CSV  = PROC_DIR / "measurements.csv"

# ── 맵핑 타깃 ─────────────────────────────────────────────
TARGET_COL = 'DHSR_GND'       # 맵핑 대상 컬럼 (μSv/h)

# EDA P0 실측 통계 (DHSR_GND, N=578)
TARGET_MEAN  = 0.0309          # μSv/h
TARGET_STD   = 0.0051
TARGET_MIN   = 0.0189
TARGET_MAX   = 0.0437
TARGET_CV    = 0.165           # 16.5% → Type A 확정

# ── 물리 파라미터 ──────────────────────────────────────────
# [주의] M4의 mu_eff=0.0824 /m은 점선원 phenomenological 파라미터.
#        M1 배경 맵핑에는 사용 불가.
#        고도 보정에는 순수 공기 감쇠계수(mu_air)를 사용한다.

MU_AIR      = 9.5e-3           # /m  — 662 keV 공기 감쇠계수 (물리값)
E_GAMMA     = 0.85             # Cs-137 branching ratio (참고용, M1 배경에서는 비중 낮음)

# M4에서 calibrate된 값 — M1 직접 사용 금지, 참조용으로만 보존
MU_EFF_M4_REFERENCE = 0.0824  # /m  (점선원 전용, Type B 파이프라인 전용)
PSF_FWHM_M  = 116.5            # m  (M4 calibration, M1 Type A에서는 적용 안 함)

# 비행 파라미터 (EDA P0 실측)
ALTITUDE_MEAN_M = 92.7         # m  (PH 평균)
ALTITUDE_MIN_M  = 71.0
ALTITUDE_MAX_M  = 113.0
ALTITUDE_REF_M  = 90.0         # 기준 고도 (보정 기준점)

# ── 좌표계 ────────────────────────────────────────────────
CRS_INPUT  = "EPSG:4326"       # WGS84
CRS_OUTPUT = "EPSG:2056"       # LV95 (Swiss)

# ── 검증 설계 (P2.5) ──────────────────────────────────────
CV_N_SPLITS   = 5              # Spatial Block CV fold 수
CV_BUFFER_M   = 58.0           # 테스트 포인트 주변 제외 반경 = PSF_FWHM/2
CV_BLOCK_AXIS = 'E'            # 블록 분할 축 ('E' = 서→동, flight line 방향 추정)
RANDOM_SEED   = 42

# ── 알고리즘 설정 (P2.6) ─────────────────────────────────
ALGORITHMS = ['IDW', 'Kriging', 'UNet']   # 비교 대상 알고리즘 목록
IDW_POWER  = 2                 # IDW 거리 지수
KRIGING_VARIOGRAM = 'spherical'  # smooth background에 적합

# ── U-Net 학습 설계 (P2.6) ────────────────────────────────
# 합성 훈련 데이터 파라미터
SYNTH_N_TRAIN     = 5000
SYNTH_N_VAL       = 500
SYNTH_FIELD_TYPE  = 'smooth_gp'  # 'smooth_gp' | 'peaky_hotspot'(M4 전용)

# GP 파라미터: CV=16% + 측정 영역 ~3km×1.3km 기반
SYNTH_GP_LENGTH_SCALE = 400    # m  — smooth field (CV 16% 근거)
SYNTH_CPS_MEAN        = TARGET_MEAN
SYNTH_CPS_STD         = TARGET_STD
SYNTH_CPS_RANGE       = (TARGET_MIN, TARGET_MAX)

# 물리 인코딩 (M4 대비 변경사항)
UNET_USE_PSF_KERNEL   = False  # M4: True → M1: False (점선원 없음)
UNET_USE_ALTITUDE_ATT = True   # 고도 감쇠는 mu_air 기준으로 유지
UNET_N_INPUT_CHANNELS = 5      # sparse_aerial, idw, site_prior, meas_mask, water_mask

# ── 출력 설정 ─────────────────────────────────────────────
MATPLOTLIB_FONT = 'Malgun Gothic'


# =====================================================================
# config.py — DEM 통합을 위해 추가할 상수들
# 기존 config.py의 맨 아래에 붙여넣을 것
# =====================================================================

# ── DEM 파일 경로 ─────────────────────────────────────────────────────
DEM_FILE = RAW_DIR / "mission1_dem_mosaic.tif"
# 확인: C:\Users\BJ\Documents\radiation-mapping\psi_mission1\data\raw\
# 에 mission1_dem_mosaic.tif 를 복사해 둘 것

# ── 수역 임계 고도 ────────────────────────────────────────────────────
LAKE_ELEV_M = 558.0          # Thunersee 수면 고도 (m MSL)
# DEM 분석: 전체 DEM에서 ≤558m = 8.4%
#           측량 영역 내에서는 ≤558m ≈ 0.4% (대부분 육지)

# ── U-Net 채널 수 업데이트 ────────────────────────────────────────────
UNET_N_INPUT_CHANNELS = 6    # 5 → 6 으로 변경
# ch0: sparse_aerial  (forward model 시뮬레이션 출력)
# ch1: idw            (IDW 초기 추정)
# ch2: dem_normalized (지형 고도 정규화, 이전: 상수 prior)
# ch3: measurement_mask
# ch4: water_mask     (DEM 기반, 이전: 상수 0)
# ch5: visibility_map (ray-casting 평균 가시성, 신규)

# ── Ray-casting 설정 ─────────────────────────────────────────────────
RAYCAST_N_STEPS = 20         # 광선 이산화 단계 수
#   20 = 정확도/속도 균형. 더 정밀하게 하려면 40~50

# ── 물리 forward model 설정 ──────────────────────────────────────────
FORWARD_USE_TERRAIN   = True   # True: DEM 기반 r², False: flat terrain (비교용)
FORWARD_USE_RAYCAST   = True   # True: V=ray_cast, False: V=1 everywhere (비교용)

# ── 합성 데이터 sparsity 설정 ────────────────────────────────────────
SYNTH_SPARSITY_RANGE  = (0.1, 0.6)   # 기존 유지
SYNTH_BLOCK_PROB      = 0.4          # 기존 유지

# ── Ablation 실험 플래그 (P7 통합 평가용) ────────────────────────────
# True/False 조합으로 물리 효과 분리 실험
# FORWARD_USE_TERRAIN=False, FORWARD_USE_RAYCAST=False → v2 기준선 재현

# ================================================================
# config.py — CPS round-trip 재설계를 위해 추가할 상수들
# 기존 config.py 맨 아래에 붙여넣을 것
# ================================================================

# ── 새 타깃 변수: 우주선 제거 CPS ───────────────────────────────
CPS_TARGET_COL  = 'CPS_terrestrial'   # measurements.csv에 새 열 추가
# CPS_terrestrial = AW_K-40 + AW_U-238 + AW_Th-232 + AW_Cs-137
# (ERS stripping이 이미 우주선 제거 완료된 핵종별 계수율 합산)

# EDA 실측 통계 (계산 전 추정값 — 06번 스크립트 실행 후 확정)
CPS_MEAN    = 226.73   # CPS  (187.81 + 18.29 + 16.98 + 3.65)
CPS_STD     = 72.0     # CPS  (06 스크립트에서 실측 후 업데이트)
CPS_MIN     = 0.0      # 음수 clip 후 최소값
CPS_MAX     = 500.0    # 추정 최대값 (실측 후 업데이트)

# ── 합성 데이터 파라미터 (CPS 스케일) ────────────────────────────
CPS_SYNTH_GP_LENGTH_SCALE = 400    # m — smooth field (기존 유지)
CPS_SYNTH_N_TRAIN         = 5000
CPS_SYNTH_N_VAL           = 500
CPS_SYNTH_FIELD_TYPE      = 'smooth_gp'

# ── Round-trip 평가 설계 ──────────────────────────────────────────
ROUNDTRIP_RATIOS  = [0.10, 0.30, 0.50]  # Group 1 비율
ROUNDTRIP_N_SEEDS = 5                   # 반복 seed 수
ROUNDTRIP_SEEDS   = [42, 7, 13, 99, 27] # 각 seed 값
ALTITUDE_MEAN_AGL = 92.7                # m (비행고도 above ground)

# ── 파일 경로 (CPS 버전) ─────────────────────────────────────────
# PROC_DIR / 'synthetic_m1_cps.npz'    — CPS 합성 데이터
# PROC_DIR / 'unet_m1_cps_best.pt'     — CPS U-Net 가중치
# RES_DIR  / 'p7_roundtrip_results.csv' — round-trip 평가 결과
# RES_DIR  / '08_p7_roundtrip.png'     — 최종 결과 그림