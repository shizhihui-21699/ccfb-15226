#!/usr/bin/env python3
"""Analyze per-frame score distributions dumped by build/eval (stdlib only)."""
import csv
import math
import sys
from collections import defaultdict

FEATS = ["usRatio", "usAbove", "usAm", "usNarrow", "usPeak", "hfRoll", "midGap", "bandContrast", "formant", "liveness", "onset", "harmonic", "flatness",
         "modSyll", "modSlow", "coherence", "modContrast", "modConc", "flux",
         "foreground", "speechRatio", "hfCons"]


def quantile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (i - lo)) + xs[hi] * (i - lo)


def main(path):
    data = defaultdict(lambda: defaultdict(list))
    raw = defaultdict(lambda: [0, 0])
    with open(path) as f:
        for row in csv.DictReader(f):
            lab = row["label"]
            for ft in FEATS:
                data[lab][ft].append(float(row[ft]))
            raw[lab][0] += int(row["aRaw"])
            raw[lab][1] += int(row["bRaw"])
    classes = list(data)
    hdr = f"{'class':<20}" + "".join(f"{ft:>12}" for ft in FEATS)
    print(hdr)
    print("-" * len(hdr))
    for lab in classes:
        # median (p50) and p90 so we see both typical and tail behavior
        p50 = [quantile(data[lab][ft], 0.5) for ft in FEATS]
        p90 = [quantile(data[lab][ft], 0.9) for ft in FEATS]
        print(f"{lab:<20}" + "".join(f"{v:>12.4f}" for v in p50))
        print(f"{'  (p90)':<20}" + "".join(f"{v:>12.4f}" for v in p90))
    print()
    print(f"{'class':<20}{'aRaw frames':>14}{'bRaw frames':>14}")
    for lab in classes:
        print(f"{lab:<20}{raw[lab][0]:>14}{raw[lab][1]:>14}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/scores.csv")
