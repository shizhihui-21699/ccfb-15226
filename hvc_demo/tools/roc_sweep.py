#!/usr/bin/env python3
"""Clip-level ROC sweep for Channel B. Reconstructs clips from frame resets,
aggregates features per clip, and sweeps candidate replay-suspicion scores to
show the honest detection/false-block tradeoff achievable in the front end."""
import csv
import sys
from collections import defaultdict


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def pct(xs, q):
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (i - lo)) + xs[hi] * (i - lo)


def load_clips(path):
    clips = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            cid = r["clip"]
            if cid not in clips:
                clips[cid] = defaultdict(list)
            cur = clips[cid]
            cur["label"].append(r["label"])
            for k in r:
                if k in ("clip", "label", "mic", "frame", "aRaw", "bRaw"):
                    continue
                cur[k].append(float(r[k]))
    return list(clips.values())


def clip_features(c):
    n = len(c["flatness"])
    # windowed medians (64-frame ~ 0.7 s), then take the most suspicious
    # sustained value (10th percentile of window medians)
    def win_med(key, w=64):
        xs = c[key]
        ms = [median(xs[i:i + w]) for i in range(0, max(1, n - w), 8)]
        return pct(ms, 0.10) if ms else median(xs)
    return {
        "label": c["label"][0],
        "flatness": median(c["flatness"]),
        "liveness": win_med("liveness"),
        "hfCons": win_med("hfCons"),
        "modSyll": median(c["modSyll"]),
        "speechRatio": median(c["speechRatio"]),
    }


def main(path):
    clips = [clip_features(c) for c in load_clips(path)]
    attacks = [c for c in clips if c["label"] == "commander_song"]
    benign = [c for c in clips if c["label"] != "commander_song"
              and c["label"] not in ("ultrasonic_attack", "ultrasonic_whine")]
    print(f"clips: {len(attacks)} commander_song, {len(benign)} benign")
    # Replay suspicion score: low liveness + low fricative modulation, but
    # only where a tonal bed and voice-band modulation exist (music arm).
    def score(c):
        arm = 1.0 if (c["flatness"] < 0.20 and c["modSyll"] > 0.07
                      and c["speechRatio"] > 0.08) else 0.0
        return arm * ((0.24 - c["liveness"]) + 0.5 * (0.30 - c["hfCons"]))
    print("\nthr   CS-detect  benign-FB  music-FB  speech/music-FB")
    for thr in [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        def flagged(c):
            return score(c) > thr
        det = sum(flagged(c) for c in attacks)
        fb = sum(flagged(c) for c in benign)
        music = sum(flagged(c) for c in benign if c["label"] == "music_only")
        som = sum(flagged(c) for c in benign
                  if c["label"] == "speech_over_music")
        print(f"{thr:5.2f}  {det:4d}/{len(attacks):3d} "
              f"({100*det/len(attacks):5.1f}%)  {fb:4d}/{len(benign):3d} "
              f"({100*fb/len(benign):5.2f}%)  {music:4d}  {som:4d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/scores.csv")
