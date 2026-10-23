#!/usr/bin/env python3
"""claim_tests.py v0.12 — v4.0 manuscript claim-level assertion tests.

v0.11 → v0.12 (2026-10-03, 워크스테이션 Final Build STEP 1-b):
    [C15] §5.4.3 파생 % 12개 assert 추가 (검토 #7 F5 잔여) — canonical block
          CSV의 full-precision fold-mean 경로로 재계산해 본문 기재값과 ±0.05%p
          대조 (CSV 부재 시 5dp-표 경로 ±0.15%p fallback).

검토 #4 Q1 반영: 숫자 보존·표 재계산 검사가 잡지 못하는 '파생 통계·부호·방향
서술' 오류(정오 2건의 유형)를 자동 검증한다.

프로세스 규율 8 (2026-08-15): 표에서 파생된 서술 통계는 본 테스트를 통과해야
PROMOTED 승격 가능.

사용법:
    python3 claim_tests.py <assembly.md>

워크스테이션 확장 TODO:
    [WS-1] 완료 2026-08-18 — 비반올림 per-line RMSE에서 median(Δ(U−K)) 재계산, 원값 ±1e-8 assert
    [WS-2] 완료 2026-08-18 — fold-assignment 재실행 → fold centroid E-좌표 단조성 assert (C8)
           (measurements.csv 부재 시 skip; f0 최서단, f4 최동단)
    [WS-3] claim registry: 각 assert에 claim ID + source script + input hash 연결 (미착수)
"""
import re, sys, statistics, os

# [WS-1] W2 v1.2 canonical per-line LOO-FL 원값 (워크스테이션 경로; 노트북에서는 부재 → skip)
RAW_LOO_CSV = os.environ.get(
    'PSI_LOO_RAW_CSV',
    r'C:\Users\BJ\Documents\radiation-mapping\psi_mission1\results\w2_summary_v12\summary_loo_fl_v12.csv')
# [WS-2] processed survey df (E_LV95 열) — fold centroid 단조성 검사용
MEAS_CSV = os.environ.get(
    'PSI_MEAS_CSV',
    r'C:\Users\BJ\Documents\radiation-mapping\psi_mission1\data\processed\measurements.csv')

def fail(msg):
    print(f"  FAIL: {msg}"); return 1

def main(path):
    t = open(path, encoding='utf-8').read().replace('−', '-')
    errors = 0

    # ---------- parse Table 5.4.A (block CV) ----------
    cells = []
    for m in re.finditer(
            r'\| (\d+) \| (\d) \| (\d+) \| ([\d.]+)% \| (0\.\d{5}) \| (0\.\d{5}) \| (0\.\d{5}) \|', t):
        buf, fold, ntr, ret, u, i, k = m.groups()
        cells.append(dict(buf=int(buf), fold=int(fold), ntr=int(ntr),
                          ret=float(ret), U=float(u), I=float(i), K=float(k)))
    print(f"parsed block-CV cells: {len(cells)}")
    if len(cells) != 20:
        errors += fail(f"expected 20 block-CV cells, got {len(cells)}")

    # ---------- parse Table 5.4.C (LOO-FL) ----------
    lines = []
    for m in re.finditer(r'\| L(\d) \| (\d+) \| (0\.\d{5}) \| (0\.\d{5}) \| (0\.\d{5}) \|', t):
        ln, n, u, i, k = m.groups()
        lines.append(dict(line=int(ln), n=int(n), U=float(u), I=float(i), K=float(k)))
    print(f"parsed LOO-FL lines: {len(lines)}")
    if len(lines) != 8:
        errors += fail(f"expected 8 LOO-FL lines, got {len(lines)}")

    # ================= CLAIM TESTS =================
    # 주석 블록(〔...〕: erratum 인용 등)은 주장 검사에서 제외
    t_claims = re.sub(r'〔.*?〕', '', t, flags=re.S)

    print("\n[C1] LOO-FL median direction & value (정오 1의 재발 방지)")
    dUK = [r['U'] - r['K'] for r in lines]
    med = statistics.median(dUK)
    npos = sum(1 for d in dUK if d > 0)
    if not med > 0:
        errors += fail(f"median(dUK)={med:+.5f} must be > 0 (5 of 8 deltas positive)")
    if npos != 5:
        errors += fail(f"positive dUK count {npos} != 5")
    # [WS-1 완료 2026-08-15, 워크스테이션] 비반올림 확정값 (W2 v1.2 canonical:
    #   results/w2_summary_v12/summary_loo_fl_v12.csv, presend_checks.py CHECK-1)
    #   median Δ(U−K) = +0.0001174759066775392 → 5자리 +0.00012
    # 주의: 여기의 med는 반올림 표(RMSE 5자리)에서 파생 → +0.00011 (RMSE열 경로).
    #   원값 ±1e-8 assert는 원값 CSV에 대해 수행하고, 표-파생 med는 5자리 반올림
    #   오차(≤1.5e-5) 안에서 확정값과 정합하는지만 확인한다.
    CONFIRMED_MEDIAN_RAW = 0.0001174759066775392
    if abs(med - CONFIRMED_MEDIAN_RAW) > 1.5e-5:
        errors += fail(f"table-derived median {med:+.5f} not consistent with confirmed "
                       f"raw median {CONFIRMED_MEDIAN_RAW:+.8f} (rounding tolerance 1.5e-5)")
    raw_csv = RAW_LOO_CSV if os.path.exists(RAW_LOO_CSV) else None
    if raw_csv:
        import csv as _csv
        raw_dUK = []
        for row in _csv.DictReader(open(raw_csv, encoding='utf-8')):
            raw_dUK.append(float(row['rmse_unet']) - float(row['rmse_kriging_matched']))
        raw_med = statistics.median(raw_dUK)
        if abs(raw_med - CONFIRMED_MEDIAN_RAW) >= 1e-8:
            errors += fail(f"raw median {raw_med:+.10f} != confirmed {CONFIRMED_MEDIAN_RAW:+.10f} (±1e-8)")
        else:
            print(f"  raw median (canonical CSV) = {raw_med:+.10f} ✓ (±1e-8)")
    else:
        print(f"  (raw CSV not found at {RAW_LOO_CSV} — raw ±1e-8 assert skipped; "
              f"table-derived consistency check only)")
    if not re.search(r'median[^.]{0,60}\+0\.0001[12]', t_claims):
        errors += fail("text does not report median as +0.00011/+0.00012")
    # 본문 주장(주석 제외)에서 음수 median·opposite direction 금지
    if re.search(r'median[^.]{0,80}(is|sits at|=)\s*-0\.00003', t_claims):
        errors += fail("text still claims median = -0.00003")
    if 'opposite direction' in t_claims:
        errors += fail('"opposite direction" contrast still present in claims')
    print(f"  median={med:+.5f}, positives={npos}/8")

    print("[C2] LOO-FL means & L5-exclusion (본문 서술 수치)")
    mU = statistics.mean(r['U'] for r in lines); mK = statistics.mean(r['K'] for r in lines)
    assert_close = lambda a, b, tol, name: None if abs(a-b) <= tol else fail(f"{name}: {a:.5f} vs {b:.5f}")
    e = assert_close(mU, 0.00294, 6e-6, "mean RMSE_U");        errors += e or 0
    e = assert_close(mK, 0.00291, 6e-6, "mean RMSE_K");        errors += e or 0
    excl5 = statistics.mean(r['U']-r['K'] for r in lines if r['line'] != 5)
    e = assert_close(excl5, 0.00016, 6e-6, "excl-L5 mean dUK"); errors += e or 0
    if sum(r['n'] for r in lines) != 578:
        errors += fail("sum n_test != 578")

    print("[C3] block-CV sign counts (서술 '2 of 5 folds at every buffer')")
    for buf in (0, 58, 125, 250):
        sub = [c for c in cells if c['buf'] == buf]
        kpos = sum(1 for c in sub if c['U'] > c['K'])
        ipos = sum(1 for c in sub if c['U'] > c['I'])
        if kpos != 2:
            errors += fail(f"buffer {buf}: Kriging-favored folds {kpos} != 2")
        expect_i = 2 if buf == 250 else 0
        if ipos != expect_i:
            errors += fail(f"buffer {buf}: IDW-favored folds {ipos} != {expect_i}")
    b250 = {c['fold'] for c in cells if c['buf'] == 250 and c['U'] > c['I']}
    if b250 != {0, 3}:
        errors += fail(f"250m IDW-favored folds {b250} != {{0,3}}")

    print("[C4] Table B aggregates from Table A (재계산 일치)")
    tblB = {0: (0.00324, 0.00434, 0.00360), 58: (0.00324, 0.00469, 0.00370),
            125: (0.00357, 0.00525, 0.00410), 250: (0.00539, 0.00617, 0.00554)}
    for buf, (eu, ei, ek) in tblB.items():
        sub = [c for c in cells if c['buf'] == buf]
        for key, exp in (('U', eu), ('I', ei), ('K', ek)):
            got = statistics.mean(c[key] for c in sub)
            if abs(got - exp) > 1e-5:
                errors += fail(f"buffer {buf} {key}: mean {got:.5f} vs table {exp:.5f}")

    print("[C5] improvement definition (positive-improvement, ratio-of-fold-means)")
    imp_v31 = 100*(0.00443-0.00335)/0.00443
    imp_w2b0 = 100*(0.00434-0.00324)/0.00434
    e = assert_close(imp_v31, 24.4, 0.05, "v3.1 improvement %");  errors += e or 0
    e = assert_close(imp_w2b0, 25.3, 0.15, "W2 b0 improvement %"); errors += e or 0
    peak = max(100*(statistics.mean(c['I'] for c in cells if c['buf']==b) -
                    statistics.mean(c['U'] for c in cells if c['buf']==b)) /
                    statistics.mean(c['I'] for c in cells if c['buf']==b) for b in (0,58,125,250))
    e = assert_close(peak, 31.9, 0.15, "peak improvement %");      errors += e or 0

    print("[C6] §5.2 canonical table (W2 buffer=0 유도값과 일치)")
    b0 = sorted((c for c in cells if c['buf'] == 0), key=lambda c: c['fold'])
    k03 = statistics.mean(c['K'] for c in b0[:4]); u03 = statistics.mean(c['U'] for c in b0[:4])
    k4, u4 = b0[4]['K'], b0[4]['U']
    e = assert_close(k03, 0.00305, 6e-6, "§5.2 K folds0-3");  errors += e or 0
    e = assert_close(u03, 0.00297, 6e-6, "§5.2 U folds0-3");  errors += e or 0
    if '| Ordinary Kriging | 0.00305 | 0.00582 | +91.1% |' not in t:
        errors += fail("§5.2 Kriging row not canonical W2-derived")
    if '| Residual U-Net | 0.00297 | 0.00435 | +46.7% |' not in t:
        errors += fail("§5.2 U-Net row not canonical W2-derived")
    e = assert_close(100*(k4/k03-1), 91.1, 0.2, "§5.2 K degradation %"); errors += e or 0
    e = assert_close(100*(u4/u03-1), 46.7, 0.2, "§5.2 U degradation %"); errors += e or 0

    print("[C7] equivalence-vocabulary control (규율 9)")
    body_new = t_claims  # 주석 제외; locked verbatim 예외는 조화 태그로 관리
    for phrase in ('practically tied', 'essentially tied'):
        if phrase in body_new:
            errors += fail(f'forbidden equivalence phrase present: "{phrase}"')

    print("[C8] geometry statements (정오 2의 재발 방지 — 텍스트 수준)")
    if 'lowest and highest E-coordinate' in t_claims:
        errors += fail('"lowest and highest E-coordinate quintiles" present outside erratum annotation')
    if 'Fold 4 is the easternmost' not in t:
        errors += fail("§5.2 must state fold 4 = easternmost")
    # [WS-2] fold centroid monotonicity from coordinates (argsort(E_LV95) + array_split, k=5)
    if os.path.exists(MEAS_CSV):
        import csv as _csv
        E = [float(r['E_LV95']) for r in _csv.DictReader(open(MEAS_CSV, encoding='utf-8'))
             if r.get('E_LV95') not in (None, '')]
        order = sorted(range(len(E)), key=lambda i: E[i])
        k = 5; n = len(order)
        # np.array_split 동등: 앞 (n % k)개 fold가 1개 더 큼
        sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        cents, pos = [], 0
        for s in sizes:
            idx = order[pos:pos + s]; pos += s
            cents.append(sum(E[i] for i in idx) / s)
        mono = all(cents[i] < cents[i + 1] for i in range(k - 1))
        if not mono:
            errors += fail(f"fold centroids not monotonic west→east: {[round(c) for c in cents]}")
        if sizes != [116, 116, 116, 115, 115]:
            errors += fail(f"fold sizes {sizes} != [116,116,116,115,115]")
        print(f"  fold centroids E = {[round(c) for c in cents]} (monotonic={mono}, sizes={sizes})")
    else:
        print(f"  (measurements.csv not found at {MEAS_CSV} — centroid assert skipped)")

    print("[C9] LOO aggregation estimands (검토 #5 F3 — 정오 6 재발 방지)")
    import math
    pooled = lambda key: math.sqrt(sum(r['n']*r[key]*r[key] for r in lines)/sum(r['n'] for r in lines))
    pU, pK = pooled('U'), pooled('K')
    e = assert_close(pU, 0.00313, 1e-5, "pooled RMSE_U");  errors += e or 0
    e = assert_close(pK, 0.00330, 1e-5, "pooled RMSE_K");  errors += e or 0
    if not pU < pK:
        errors += fail("pooled must favor U-Net (text claims reversal)")
    l7 = [r for r in lines if r['line'] != 5]
    p7 = lambda key: math.sqrt(sum(r['n']*r[key]*r[key] for r in l7)/sum(r['n'] for r in l7))
    if not p7('K') < p7('U'):
        errors += fail("pooled excl-L5 must favor Kriging (L5 attribution claim)")
    if '0.00313' in t_claims and '0.00330' not in t_claims:
        errors += fail("pooled U reported without pooled K")

    print("[C10] equivalence vocabulary — extended ban (규율 9, 검토 #5 F5)")
    for phrase in ('statistically indistinguishable', 'statistical tie', 'practically indistinguishable', ' indistinguishable at', 'amounting to a tie', ' near-tie'):
        if phrase in t_claims:
            errors += fail(f'forbidden phrase in claims: "{phrase}"')

    print("[C11] L5 weighting misstatement ban (정오 6)")
    if 'more strongly weights the mean aggregate' in t_claims:
        errors += fail('L5 size-weighting misstatement present in claims')

    # ================= v0.8 신규 (Analysis Bundle 2026-08-18) =================
    ANALYSIS_DIR = os.environ.get(
        'PSI_ANALYSIS_DIR',
        r'C:\Users\BJ\Documents\radiation-mapping\psi_mission1\results\analysis_2026-08-18')
    import csv as _csv

    print("[C12] W2-B partition sensitivity (PR-2 파생 서술)")
    w2b_csv = os.path.join(ANALYSIS_DIR, 'w2b_partition_results.csv')
    if os.path.exists(w2b_csv):
        rows = list(_csv.DictReader(open(w2b_csv, encoding='utf-8')))
        if len(rows) != 35:
            errors += fail(f"W2-B cells {len(rows)} != 35")
        parts = {}
        for r in rows:
            p = r['partition_id']
            parts.setdefault(p, []).append(
                (float(r['rmse_u']), float(r['rmse_i']), float(r['rmse_k'])))
        n_ui_neg = sum(1 for p, cs in parts.items()
                       if statistics.mean(u - i for u, i, k in cs) < 0)
        n_uk_neg = sum(1 for p, cs in parts.items()
                       if statistics.mean(u - k for u, i, k in cs) < 0)
        if n_ui_neg != 7:
            errors += fail(f"W2-B: mean dUI negative in {n_ui_neg}/7 != 7/7")
        if n_uk_neg != 6:
            errors += fail(f"W2-B: mean dUK negative in {n_uk_neg}/7 != 6/7")
        na = parts['naxis']
        duk_na = statistics.mean(u - k for u, i, k in na)
        if not (0 < duk_na < 0.0001):
            errors += fail(f"W2-B: naxis dUK {duk_na:+.5f} not in (0, +0.0001)")
        i_na = statistics.mean(i for u, i, k in na)
        if abs(i_na - 0.00349) > 1e-5:
            errors += fail(f"W2-B: naxis IDW mean {i_na:.5f} != 0.00349")
        kf = [sum(1 for u, i, k in cs if u > k) for cs in parts.values()]
        if not all(2 <= x <= 3 for x in kf):
            errors += fail(f"W2-B: K-favored folds {kf} outside 2-3 band")
        for claim in ('7/7 partitions', '6/7', '0.00434', '0.00349'):
            if claim not in t:
                errors += fail(f"W2-B claim '{claim}' missing from text")
        print(f"  dUI neg {n_ui_neg}/7, dUK neg {n_uk_neg}/7, "
              f"naxis dUK={duk_na:+.5f}, K-fav band {min(kf)}-{max(kf)}")
    else:
        errors += fail(f"w2b CSV not found: {w2b_csv}")

    print("[C13] W2-C fold-safe sensitivity (PR-3 파생 서술)")
    w2c_csv = os.path.join(ANALYSIS_DIR, 'w2c_results.csv')
    RUN_SD = 0.000201
    if os.path.exists(w2c_csv):
        rows = list(_csv.DictReader(open(w2c_csv, encoding='utf-8')))
        if len(rows) != 15:
            errors += fail(f"W2-C rows {len(rows)} != 15 (3 arms × 5 folds)")
        arm = {}
        for r in rows:
            arm.setdefault(r['arm'], {})[int(r['fold'])] = float(r['rmse_u'])
        mA0 = statistics.mean(arm['Arm0'].values())
        dA = statistics.mean(arm['ArmA'].values()) - mA0
        dB = statistics.mean(arm['ArmB'].values()) - mA0
        if abs(dA / RUN_SD - 0.1) > 0.15:
            errors += fail(f"W2-C: ArmA mean delta {dA/RUN_SD:+.2f} SD != +0.1")
        if abs(dB / RUN_SD - (-1.6)) > 0.15:
            errors += fail(f"W2-C: ArmB mean delta {dB/RUN_SD:+.2f} SD != -1.6")
        if not (abs(dA) < 2 * RUN_SD and abs(dB) < 2 * RUN_SD):
            errors += fail("W2-C: mean deltas not below pre-registered 2 run-SD")
        idw = {int(r['fold']): float(r['rmse_i']) for r in rows if r['arm'] == 'Arm0'}
        for a in ('Arm0', 'ArmA', 'ArmB'):
            wins = sum(1 for f in range(5) if arm[a][f] < idw[f])
            if wins < 4:
                errors += fail(f"W2-C: {a} U<I in {wins}/5 — mean-rank claim at risk")
        for claim_re, label in ((r'\+0\.1 run-to-run\s+SD', '+0.1 run-to-run SD'),
                                (r'1\.6 run-to-run\s+SD', '1.6 run-to-run SD')):
            if not re.search(claim_re, t):
                errors += fail(f"W2-C claim '{label}' missing from text")
        print(f"  dA={dA/RUN_SD:+.2f} SD, dB={dB/RUN_SD:+.2f} SD (threshold 2 SD)")
    else:
        errors += fail(f"w2c CSV not found: {w2c_csv}")

    print("[C14] Layer C in-situ validation (PR-1 파생 서술)")
    lc_csv = os.path.join(ANALYSIS_DIR, 'layerc_results.csv')
    if os.path.exists(lc_csv):
        rows = list(_csv.DictReader(open(lc_csv, encoding='utf-8')))
        if len(rows) != 23:
            errors += fail(f"Layer C points {len(rows)} != 23")
        mae = {}
        bias = {}
        for m, col in (('U-Net', 'resid_unet'), ('IDW', 'resid_idw'),
                       ('Kriging', 'resid_kriging')):
            rs = [float(r[col]) for r in rows]
            mae[m] = statistics.mean(abs(x) for x in rs)
            bias[m] = statistics.mean(rs)
        exp_mae = {'U-Net': 5.50, 'IDW': 5.47, 'Kriging': 5.41}
        for m, e_v in exp_mae.items():
            if abs(mae[m] - e_v) > 0.02:
                errors += fail(f"Layer C: {m} MAE {mae[m]:.2f} != {e_v}")
        if not all(4.2 < b < 5.0 for b in bias.values()):
            errors += fail(f"Layer C: biases {bias} outside +4.3~+4.9 claim band")
        # 정합 상수 재계산: reconciled = raw − 49.39 (±0.01)
        d0 = float(rows[0]['insitu_raw']) - float(rows[0]['insitu_reconciled'])
        if abs(d0 - 49.39) > 0.02:
            errors += fail(f"Layer C: reconciliation constant {d0:.2f} != 49.39")
        for claim in ('49.39', '5.41', '5.50', '+4.3 to', '+4.9'):
            if claim not in t:
                errors += fail(f"Layer C claim '{claim}' missing from text")
        print(f"  MAE U/I/K = {mae['U-Net']:.2f}/{mae['IDW']:.2f}/{mae['Kriging']:.2f}, "
              f"bias band {min(bias.values()):+.2f}~{max(bias.values()):+.2f}")
    else:
        errors += fail(f"layerc CSV not found: {lc_csv}")


    print("[C15] §5.4.3 derived percentages (12 asserts — v0.12, 검토 #7 F5)")
    def _pct_list(anchor, n=4):
        i = t.find(anchor)
        if i < 0:
            return None
        vals = re.findall(r'Buffer \d+ m: ([+-]\d+\.\d)%', t[i:i + 600])
        return [float(v) for v in vals[:n]] if len(vals) >= n else None
    stated_relI = _pct_list('Relative U-Net advantage over IDW')
    stated_relK = _pct_list('Relative U-Net advantage over Kriging')
    m_abs = re.search(r'U-Net: 0\.00324 . 0\.00539, \+(\d+\.\d)%', t)
    m_absK = re.search(r'Kriging: 0\.00360 . 0\.00554, \+(\d+\.\d)%', t)
    m_absI = re.search(r'IDW: 0\.00434 . 0\.00617, \+(\d+\.\d)%', t)
    if not (stated_relI and stated_relK and m_abs and m_absK and m_absI):
        errors += fail("C15: §5.4.3 파생 % 본문 파싱 실패")
    else:
        stated_abs = {'U': float(m_abs.group(1)), 'K': float(m_absK.group(1)),
                      'I': float(m_absI.group(1))}
        RAW_BLOCK_CSV = os.environ.get(
            'PSI_BLOCK_RAW_CSV',
            r'C:\Users\BJ\Documents\radiation-mapping\psi_mission1\results'
            r'\w2_summary_v12\summary_block_cv_v12.csv')
        if os.path.exists(RAW_BLOCK_CSV):
            import csv as _csv
            rows = list(_csv.DictReader(open(RAW_BLOCK_CSV, encoding='utf-8')))
            mean = {}
            for buf in (0, 58, 125, 250):
                sub = [r for r in rows if int(r['buffer_m']) == buf]
                mean[buf] = (statistics.mean(float(r['rmse_unet']) for r in sub),
                             statistics.mean(float(r['rmse_idw_matched']) for r in sub),
                             statistics.mean(float(r['rmse_kriging_matched']) for r in sub))
            tol, src_lbl = 0.05, 'full-precision CSV'
        else:
            byb = {}
            for c in cells:
                byb.setdefault(c['buf'], []).append(c)
            mean = {b: (statistics.mean(x['U'] for x in v),
                        statistics.mean(x['I'] for x in v),
                        statistics.mean(x['K'] for x in v)) for b, v in byb.items()}
            tol, src_lbl = 0.15, '5dp-table fallback'
        relI = {b: 100 * (mean[b][0] - mean[b][1]) / mean[b][1] for b in mean}
        relK = {b: 100 * (mean[b][0] - mean[b][2]) / mean[b][2] for b in mean}
        absn = {'U': 100 * (mean[250][0] / mean[0][0] - 1),
                'I': 100 * (mean[250][1] / mean[0][1] - 1),
                'K': 100 * (mean[250][2] / mean[0][2] - 1)}
        for j, b in enumerate((0, 58, 125, 250)):
            if abs(relI[b] - stated_relI[j]) > tol:
                errors += fail(f"C15 relI buf{b}: text {stated_relI[j]:+.1f} vs calc {relI[b]:+.2f}")
            if abs(relK[b] - stated_relK[j]) > tol:
                errors += fail(f"C15 relK buf{b}: text {stated_relK[j]:+.1f} vs calc {relK[b]:+.2f}")
        for k in 'UIK':
            if abs(absn[k] - stated_abs[k]) > tol:
                errors += fail(f"C15 abs {k}: text +{stated_abs[k]:.1f} vs calc +{absn[k]:.2f}")
        print(f"  [{src_lbl}] relI " + '/'.join(f'{relI[b]:+.2f}' for b in (0, 58, 125, 250)) +
              "  relK " + '/'.join(f'{relK[b]:+.2f}' for b in (0, 58, 125, 250)) +
              f"  abs U/I/K {absn['U']:.2f}/{absn['I']:.2f}/{absn['K']:.2f}")

    print()
    if errors:
        print(f"RESULT: {errors} claim test failure(s)"); sys.exit(1)
    print("RESULT: all claim tests PASS")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'v40_manuscript_assembly_v03.md')
