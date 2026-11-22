#!/usr/bin/env python3
"""Per-clip aggregation of detector features to set clip-level thresholds."""
import csv
import sys
from collections import defaultdict

# (label, mic, clip identity) needs frame ordering only; clips are contiguous
# runs of frames per label/mic. We approximate per-clip stats by label/mic
# medians of frame medians plus tail percentiles of the gate features.
GATES = ["flatness", "modSyll", "modSlow", "liveness", "hfRoll",
         "speechRatio", "midGap"]


def pct(xs, q):
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (i - lo)) + xs[hi] * (i - lo)


def main(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["label"]].append(r)
    print(f"{'class':<20}{'modSyll/slow p10':>18}{'liveness p90':>15}"
          f"{'liveness p50':>15}{'hfRoll p90':>13}{'spRatio p50':>14}")
    print("-" * 95)
    for lab, rs in rows.items():
        ms = [float(x["modSyll"]) for x in rs]
        mw = [float(x["modSlow"]) for x in rs]
        ratio = [a / (b + 1e-9) for a, b in zip(ms, mw)]
        liv = [float(x["liveness"]) for x in rs]
        hfr = [float(x["hfRoll"]) for x in rs]
        spr = [float(x["speechRatio"]) for x in rs]
        print(f"{lab:<20}{pct(ratio,0.10):>18.3f}{pct(liv,0.90):>15.3f}"
              f"{pct(liv,0.50):>15.3f}{pct(hfr,0.90):>13.3f}"
              f"{pct(spr,0.50):>14.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/scores.csv")
