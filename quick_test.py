#!/usr/bin/env python3
"""Quick test for the thun-validation-geometry repository.

Recomputes the paper's headline aggregates directly from the canonical
result tables included in this repository and checks them against the
published values. Requires only the Python standard library.

Usage:
    python quick_test.py

Expected output: 10 checks, all OK, exit code 0.
"""
import csv
import statistics
import sys

OK = True


def check(name, got, want, tol):
    global OK
    good = abs(got - want) <= tol
    print(f"[{'OK' if good else 'FAIL'}] {name}: got {got:.5f}, expected {want} (tol {tol})")
    if not good:
        OK = False


# ---- Block-CV aggregates (Table 5.4.B of the paper) ----
rows = list(csv.DictReader(open("summary_block_cv_v12.csv", encoding="utf-8")))
by_buf = {}
for r in rows:
    by_buf.setdefault(int(r["buffer_m"]), []).append(r)

mean = {}
for buf, rs in sorted(by_buf.items()):
    mean[buf] = tuple(statistics.mean(float(r[k]) for r in rs)
                      for k in ("rmse_unet", "rmse_idw_matched", "rmse_kriging_matched"))

check("buffer=0 mean RMSE, U-Net", mean[0][0], 0.00324, 5e-5)
check("buffer=0 mean RMSE, IDW", mean[0][1], 0.00434, 5e-5)
check("buffer=0 mean RMSE, Kriging", mean[0][2], 0.00360, 5e-5)

rel_125 = 100 * (mean[125][0] - mean[125][1]) / mean[125][1]
check("relative U-Net advantage vs IDW at 125 m (%, peak)", rel_125, -31.9, 0.1)

sens_u = 100 * (mean[250][0] - mean[0][0]) / mean[0][0]
check("U-Net buffer sensitivity 0->250 m (%)", sens_u, 66.3, 0.1)

# fold-paired sign count: Kriging outperforms U-Net in 2 of 5 folds at every buffer
counts = {buf: sum(1 for r in rs if float(r["rmse_unet"]) > float(r["rmse_kriging_matched"]))
          for buf, rs in by_buf.items()}
good = all(c == 2 for c in counts.values())
print(f"[{'OK' if good else 'FAIL'}] Kriging outperforms U-Net in 2/5 folds at every buffer: {counts}")
OK = OK and good

# ---- LOO-FL aggregates (Table 5.4.C of the paper) ----
loo = list(csv.DictReader(open("summary_loo_fl_v12.csv", encoding="utf-8")))
mu = statistics.mean(float(r["rmse_unet"]) for r in loo)
mk = statistics.mean(float(r["rmse_kriging_matched"]) for r in loo)
check("LOO-FL line-macro mean, U-Net", mu, 0.00294, 5e-5)
check("LOO-FL line-macro mean, Kriging", mk, 0.00291, 5e-5)

kw = sum(1 for r in loo if float(r["rmse_unet"]) > float(r["rmse_kriging_matched"]))
good = kw == 5
print(f"[{'OK' if good else 'FAIL'}] Kriging outperforms U-Net in 5/8 LOO-FL lines: {kw}/8")
OK = OK and good

# point-pooled estimand reversal (+5.0% U-Net advantage)
sum_u = sum(float(r["rmse_unet"]) ** 2 * int(r["n_test"]) for r in loo)
sum_k = sum(float(r["rmse_kriging_matched"]) ** 2 * int(r["n_test"]) for r in loo)
n = sum(int(r["n_test"]) for r in loo)
pu, pk = (sum_u / n) ** 0.5, (sum_k / n) ** 0.5
check("point-pooled U-Net advantage vs Kriging (%)", 100 * (pk - pu) / pk, 5.0, 0.2)

print()
if OK:
    print("ALL CHECKS PASSED — the included tables reproduce the paper's headline numbers.")
    sys.exit(0)
print("SOME CHECKS FAILED")
sys.exit(1)
