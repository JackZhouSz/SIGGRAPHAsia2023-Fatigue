#!/usr/bin/env python3
"""Budget-matching check for the resolution ablation.

Run this BEFORE looking at t50 or kick counts. If cumulative knee exposure is
not matched across rungs, differences in depletion timing are differences in
fatigue rate, not in representation, and the comparison says nothing.

Target: rungs (i)-(iii) previously gave 0.698-0.704.
"""

import os
import numpy as np
import pandas as pd

RUNGS = [
    ('(i)   uniform 3CC-r', 'rc_log_uniform_3CCr.csv'),
    ('(0)   load-dependent', 'rc_log_load_dep.csv'),
    ('(ii)  per-group', 'rc_log_per_group.csv'),
    ('(iii) FGS', 'rc_log_FGS.csv'),
]
TARGET = (0.690, 0.712)


def segments(d):
    """Split each env on frame resets; keep the longest run.

    Episode tagging lives in the base class but the task subclass overrides
    _reset_actors, so the episode column may be all zeros. Frame decreases are
    the reliable segmentation signal.
    """
    out = {}
    for env, g in d.groupby('env'):
        g = g.reset_index(drop=True)
        cuts = [0] + list(np.where(np.diff(g.frame.values) < 0)[0] + 1) + [len(g)]
        a, b = max(zip(cuts[:-1], cuts[1:]), key=lambda ab: ab[1] - ab[0])
        out[env] = g.iloc[a:b].reset_index(drop=True)
    return out


def main():
    print(f"{'rung':22s} {'knee expo':>10s} {'arm expo':>9s} "
          f"{'knee RC min':>12s} {'falls':>7s}  status")
    for lab, f in RUNGS:
        if not os.path.exists(f):
            print(f"{lab:22s} {'-- not found --':>10s}")
            continue
        d = pd.read_csv(f)
        S = segments(d)
        knee = np.mean([1 - g.RC_right_knee.mean() for g in S.values()])
        arm = np.mean([1 - g.RC_right_shoulder_y.mean() for g in S.values()])
        mn = np.mean([g.RC_right_knee.min() for g in S.values()])
        falls = int(d.groupby('env').terminated.max().sum())
        ok = TARGET[0] <= knee <= TARGET[1]
        status = 'MATCHED' if ok else ('EXPOSURE TOO HIGH -> raise loadDepCenter'
                                       if knee > TARGET[1]
                                       else 'EXPOSURE TOO LOW -> lower loadDepCenter')
        print(f"{lab:22s} {knee:10.3f} {arm:9.3f} {mn:12.3f} "
              f"{falls:4d}/{len(S):<2d}  {status}")
    print(f"\ntarget knee exposure {TARGET[0]:.3f}-{TARGET[1]:.3f}")
    print("Arm exposure is EXPECTED to differ: per-group gives the arms a higher")
    print("recovery ratio than the legs by design. Only the knee must match.")


if __name__ == '__main__':
    main()
