"""Benchmark sensor_placement_qr's factorization paths at one problem size.

Runs in a fresh process so peak RSS is attributable. Compares the full
LAPACK pivoted QR against the truncated k-step greedy (identical
selections), plus reconstruction evaluation.

Usage:
    python benchmarks/bench_placement.py <n_timesteps> <n_locations> <k>

Prints CSV: n_t,n_loc,k,t_full,t_truncated,match,t_eval,peak_rss_gib
"""
import resource
import sys
import time

import numpy as np

from hydrosensenet import sensor_placement_qr
from hydrosensenet.core.metrics import reconstruction_evaluation


def main():
    n_t, n_loc, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    rng = np.random.default_rng(0)
    rank = 32
    U = rng.uniform(0.5, 1.5, (n_t, rank))
    V = rng.uniform(0.5, 2.0, (rank, n_loc))
    X = U @ V + 0.01 * np.abs(rng.standard_normal((n_t, n_loc)))

    t0 = time.perf_counter()
    sel_full = sensor_placement_qr(X, k, method="full")
    t_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    sel_trunc = sensor_placement_qr(X, k, method="truncated")
    t_trunc = time.perf_counter() - t0

    match = "exact" if list(sel_full) == list(sel_trunc) else (
        "sameset" if set(sel_full) == set(sel_trunc) else "DIFF"
    )

    tr = int(0.7 * n_t)
    t0 = time.perf_counter()
    reconstruction_evaluation(X[:tr], X[tr:], sel_trunc, k)
    t_eval = time.perf_counter() - t0

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(f"{n_t},{n_loc},{k},{t_full:.2f},{t_trunc:.2f},{match},{t_eval:.2f},{peak:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
