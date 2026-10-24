# thun-validation-geometry

Analysis code and canonical result tables for the manuscript:

> **Validation Geometry and Aggregation Estimand Determine Method Rankings
> in Airborne Radiation Mapping: An Audited Benchmark of Learned versus
> Geostatistical Reconstruction** — Byoung-Jik Kim (Changwon National
> University), under review at *Computers & Geosciences*, 2026.

The study benchmarks a residual U-Net (trained entirely on
survey-conditioned synthetic fields) against inverse-distance weighting
(IDW) and Ordinary Kriging on the 578-point AGC25 Mission 1 airborne
gamma-ray survey of the Thun reference area (Switzerland), and shows that
the cross-method ranking is a property of the (validation geometry,
aggregation estimand) pair.

**Archived snapshot:** this repository is archived on Zenodo —
DOI [10.5281/zenodo.23122892](https://doi.org/10.5281/zenodo.23122892).
SHA-256 checksums of the archived files are listed in
`MANIFEST_sha256.md`. License: MIT (`LICENSE`); `quick_test.py` and this
README are repository additions outside the Zenodo manifest, and
`README_deposit.md` (listed in the manifest) is the Zenodo deposit's own
README, superseded here by this file.

## Contents

| File | Role |
|---|---|
| `03_p3_synthesize.py` | Survey-conditioned synthetic-field generator (training data) |
| `04_p5_unet.py` | Residual U-Net definition, training, and inference |
| `02_p4_baselines.py` | IDW and Ordinary Kriging baselines (v3.1 protocol) |
| `03_run_w2_experiments_v12.py` | Buffer-matched block-CV and LOO-FL evaluation harness (paper §5.4) |
| `config.py` | Locked v3.1 configuration constants |
| `claim_tests_v0.12.py` | Automated claim verification: 15 assertion groups re-derive the paper's principal derived numerical claims (coverage note below) |
| `quick_test.py` | **Quick test** (standard library only; see below) |
| `summary_block_cv_v12.csv` | Canonical block-CV results — paper Tables 5.4.A/B |
| `summary_loo_fl_v12.csv` | Canonical LOO-FL results — paper Table 5.4.C |
| `w2b_partition_results.csv` | Partition-sensitivity results — paper Table S2.1 |
| `w2c_results.csv` | Fold-safe conditioning results — paper Table S2.2 |
| `fold_line_index.csv` | Coordinate-free fold and flight-line assignment per measurement row |
| `MANIFEST_sha256.md` | SHA-256 manifest of the archived set |
| `LICENSE` | MIT |

## Quick test (no dependencies, < 1 s)

Run from the repository root:

```
python quick_test.py
```

Recomputes the paper's headline aggregates directly from the included
canonical tables and checks them against the published values — the
buffer-level means, the −31.9% peak U-Net advantage over IDW, the 2-of-5
fold-paired Kriging counts, the LOO-FL line-macro means (0.00294 vs
0.00291), the 5-of-8 line counts, and the point-pooled estimand reversal
(+5.0%). Expected output: 10 checks, all OK, exit code 0.

## Full claim-verification suite

`claim_tests_v0.12.py` re-derives the manuscript's principal derived
numerical claims from the canonical tables and fails on any inconsistency.
Its text input must be the **full submission text: the manuscript and the
Supplementary Material concatenated into one file** (one verified value
appears only in supplementary Table S2.1a; run on the manuscript alone,
the suite adds a spurious C12 failure). The submission text itself is not
part of this repository; the suite is provided for editors and reviewers
who hold the submission files:

```
cat manuscript.md supplementary.md > submission.md
```

The CSV locations are read from environment variables, which **must be
set** (without them the script falls back to the author's local paths and
reports spurious failures). From the repository root —

Linux/macOS:

```
export PSI_ANALYSIS_DIR="$(pwd)"
export PSI_BLOCK_RAW_CSV="$(pwd)/summary_block_cv_v12.csv"
export PSI_LOO_RAW_CSV="$(pwd)/summary_loo_fl_v12.csv"
python claim_tests_v0.12.py submission.md
```

Windows (cmd):

```
set PSI_ANALYSIS_DIR=%cd%
set PSI_BLOCK_RAW_CSV=%cd%\summary_block_cv_v12.csv
set PSI_LOO_RAW_CSV=%cd%\summary_loo_fl_v12.csv
python claim_tests_v0.12.py submission.md
```

`PSI_LOO_RAW_CSV` activates C1's full-precision (±1e-8) raw-median
assertion against the public LOO-FL table; without it C1 still runs its
table-derived consistency check.

**Expected outcome on this public set:** one assertion group (C14, the
in-situ Layer C check) requires `layerc_results.csv`, which contains
surveyed coordinates and is therefore not publicly distributable (see
below). It reports a single missing-file failure — the final line reads
`RESULT: 1 claim test failure(s)` and the exit code is 1. All other groups
(C1–C13, C15) run normally and pass. C8's fold-centroid monotonicity
assertion requires the gated `measurements.csv` and is skipped on the
public set (its text-level geometry checks still run).

**Coverage note:** the suite verifies the §5.2–§5.4 headline aggregates,
paired sign counts and derived percentages, the W2-B/W2-C sensitivity
statements, the Layer C metrics (C14), and the manuscript's
equivalence-vocabulary discipline. It does not grep every number in the
manuscript; values stated once outside these groups (e.g., the §5.4.5
seed-variance figures) are carried directly from the locked source tables
rather than re-derived by the suite.

## Requirements

- Quick test and claim tests: Python ≥ 3.10, standard library only.
- Full pipeline (synthesis, training, evaluation): Python 3.12, NumPy,
  SciPy, pandas, PyTorch (CUDA optional), PyKrige. Versions as of the final
  verification build: PyTorch 2.11, NumPy 2.4, SciPy 1.17, PyKrige 1.7.3.

## Data availability

The airborne survey data, the AGS_CH reference grid and the in-situ
measurements were provided by the Paul Scherrer Institute within the AGC25
exercise framework; their redistribution is subject to the permission of
the Swiss data owner (NEOC). The trained checkpoint and any artifact
containing survey measurement coordinates (`measurements.csv`,
`flight_lines.csv`, `layerc_results.csv`, `synthetic_m1.npz`,
`unet_m1_best.pt`) are therefore excluded from this repository and are
available from the corresponding author (bjkim@changwon.ac.kr). The
training and evaluation scripts are complete as published; they require
those files to re-run the full pipeline, while the quick test and the claim
suite run on the included canonical tables alone.
