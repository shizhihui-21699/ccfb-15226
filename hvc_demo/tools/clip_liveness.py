#!/usr/bin/env python3
"""Clip-level liveness: group frames into clips and compute windowed medians
to verify that replay and live voices separate after temporal accumulation."""
import csv
import sys
from collections import defaultdict


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main(path):
    clips = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            clips[(r["label"], r["mic"])].append(float(r["liveness"]))
    by_label = defaultdict(list)
    for (lab, mic), xs in clips.items():
        # rolling median over ~0.7s windows (64 frames), report the minimum
        # sustained value an attacker could hide in and the clip median
        w = 64
        win_med = [median(xs[i:i + w]) for i in range(0, len(xs) - w, 8)]
        if win_med:
            by_label[lab].append((median(xs), min(win_med), max(win_med)))
    print(f"{'class':<20}{'clip median range':>26}{'min windowed med':>20}")
    for lab, vals in by_label.items():
        med = sorted(v[0] for v in vals)
        wmin = sorted(v[1] for v in vals)
        print(f"{lab:<20}  [{med[0]:.3f} .. {med[-1]:.3f}]      "
              f"p10={wmin[len(wmin)//10]:.3f} p50={wmin[len(wmin)//2]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/scores.csv")
