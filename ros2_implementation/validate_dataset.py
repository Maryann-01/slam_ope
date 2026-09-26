#!/usr/bin/env python3
"""Validate the logged OPE dataset before scaling up data collection."""

import csv
import glob
import math
import os
import sys

DT = 0.1                 # control period [s]
ROOM_X = (-4.5, 4.5)     # wall positions from the .wbt file
ROOM_Y = (-4.0, 4.0)
MAX_LINEAR = 0.15        # fastest commanded linear speed [m/s]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load(folder):
    eps = []
    for f in sorted(glob.glob(os.path.join(folder, "*_ep*.csv"))):
        rows = list(csv.DictReader(open(f)))
        if rows:
            eps.append((os.path.basename(f), rows))
    return eps


def num(row, key):
    return float(row[key])


def odom_to_world(rows):
    r0 = rows[0]
    sx, sy, syaw = num(r0, "start_x"), num(r0, "start_y"), num(r0, "start_yaw")
    ox, oy, oyaw = num(r0, "odom_x"), num(r0, "odom_y"), num(r0, "odom_yaw")
    out = []
    for r in rows:
        dx, dy = num(r, "odom_x") - ox, num(r, "odom_y") - oy
        c, s = math.cos(-oyaw), math.sin(-oyaw)
        rx, ry = c * dx - s * dy, s * dx + c * dy
        cw, sw = math.cos(syaw), math.sin(syaw)
        out.append((sx + cw * rx - sw * ry, sy + sw * rx + cw * ry))
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def verdict(ok, warn=False):
    return "FAIL" if not ok and not warn else ("WARN" if warn else "PASS")


def main():
    folder = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/ope_data")
    trace_ep = None
    if "--trace" in sys.argv:
        trace_ep = int(sys.argv[sys.argv.index("--trace") + 1])

    eps = load(folder)
    if not eps:
        print(f"No CSVs found in {folder}")
        return
    print(f"Loaded {len(eps)} episodes from {folder}\n")

    print("=" * 78)
    print("CHECK 1  Ground truth obeys physics")
    print("-" * 78)
    worst = 0.0
    bad = 0
    for name, rows in eps:
        for a, b in zip(rows, rows[1:]):
            d = math.hypot(num(b, "true_x") - num(a, "true_x"),
                           num(b, "true_y") - num(a, "true_y"))
            worst = max(worst, d)
            if d > 0.06:
                bad += 1
    lim = 0.06
    print(f"  max step displacement = {worst:.4f} m   (limit {lim:.4f} m)")
    print(f"  violations: {bad}")
    print(f"  -> {verdict(bad == 0)}\n")

    print("=" * 78)
    print("CHECK 2  Ground truth stays inside the room")
    print("-" * 78)
    out = 0
    for name, rows in eps:
        for r in rows:
            x, y = num(r, "true_x"), num(r, "true_y")
            if not (ROOM_X[0] < x < ROOM_X[1] and ROOM_Y[0] < y < ROOM_Y[1]):
                out += 1
    print(f"  rows outside room: {out}")
    print(f"  -> {verdict(out == 0)}\n")

    print("=" * 78)
    print("CHECK 3  Frame maths is correct (odometry control)")
    print("  Valid only for SHORT episodes (<150 steps); dead-reckoning")
    print("  drift dominates over longer runs, so no verdict there.")
    print("-" * 78)
    ok3 = True
    for name, rows in eps:
        ow = odom_to_world(rows)
        oerr = [math.hypot(p[0] - num(r, "true_x"), p[1] - num(r, "true_y"))
                for p, r in zip(ow, rows)]
        serr = [math.hypot(num(r, "est_x") - num(r, "true_x"),
                           num(r, "est_y") - num(r, "true_y")) for r in rows]
        flag = "" if max(oerr) < 0.30 or len(rows) < 150 else "  <-- suspicious"
        if max(oerr) >= 0.30 and len(rows) < 150:
            ok3 = False
        print(f"  {name}: odom_err max={max(oerr):.3f}  slam_err max={max(serr):.3f}{flag}")
    print(f"  -> {verdict(ok3)}\n")

    print("=" * 78)
    print("CHECK 4  Actions consistent with observed motion")
    print("-" * 78)
    ratios = []
    for name, rows in eps:
        for a, b in zip(rows, rows[1:]):
            cmd = num(a, "action_linear")
            if cmd > 0.02:
                d = math.hypot(num(b, "true_x") - num(a, "true_x"),
                               num(b, "true_y") - num(a, "true_y"))
                ratios.append((d / DT) / cmd)
    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        print(f"  actual/commanded speed ratio: median={med:.2f} "
              f"p05={ratios[len(ratios)//20]:.2f} p95={ratios[-max(1,len(ratios)//20)]:.2f}")
        print("  (1.00 = perfect tracking; <1 means the robot under-delivers)")
        print(f"  -> {verdict(0.5 < med < 1.5, warn=not (0.75 < med < 1.25))}\n")
    else:
        print("  no moving steps found\n")

    print("=" * 78)
    print("CHECK 5  Behaviour-policy density usable for importance sampling")
    print("-" * 78)
    worst5 = 0.0
    for name, rows in eps:
        for r in rows:
            a, m, s = (num(r, "action_angular"), num(r, "action_angular_mean"),
                       num(r, "sigma"))
            expect = -0.5 * ((a - m) / s) ** 2 - math.log(s * math.sqrt(2 * math.pi))
            worst5 = max(worst5, abs(expect - num(r, "log_prob")))
    print(f"  max |recomputed - logged| log_prob = {worst5:.2e}")
    print(f"  -> {verdict(worst5 < 1e-6)}\n")

    print("=" * 78)
    print("CHECK 6  Does reported covariance predict actual error?")
    print("  Core research measurement, not pass/fail.")
    print("-" * 78)
    pooled_cov, pooled_err = [], []
    ep_cov, ep_err = [], []
    for name, rows in eps:
        errs, covs = [], []
        for r in rows:
            e = math.hypot(num(r, "est_x") - num(r, "true_x"),
                           num(r, "est_y") - num(r, "true_y"))
            c = num(r, "cov_xx")
            errs.append(e); covs.append(c)
            pooled_err.append(e); pooled_cov.append(c)
        ep_err.append(max(errs)); ep_cov.append(max(covs))
        print(f"  {name}: err mean={sum(errs)/len(errs):.3f} max={max(errs):.3f} | "
              f"cov mean={sum(covs)/len(covs):.3f} max={max(covs):.3f}")
    r_step = pearson(pooled_cov, pooled_err)
    r_ep = pearson(ep_cov, ep_err)
    print(f"\n  per-step correlation  r = {r_step:+.3f}  (n={len(pooled_err)})")
    print(f"  per-episode (peaks)   r = {r_ep:+.3f}  (n={len(ep_err)})")
    print("  r near 0 -> covariance carries little info about true error.")
    print("  REPORT THIS NUMBER whichever way it comes out.\n")

    print("=" * 78)
    print("CHECK 7  Dataset is OPE-ready")
    print("-" * 78)
    n_term = n_trunc = n_goal = n_coll = 0
    returns = []
    for name, rows in eps:
        last = rows[-1]
        n_term += int(num(last, "terminated"))
        n_trunc += int(num(last, "truncated"))
        reason = last["term_reason"]
        n_goal += (reason == "goal")
        n_coll += (reason == "collision")
        returns.append(sum(num(r, "reward") for r in rows))
    flags_ok = all(
        not (int(num(r[-1], "terminated")) and int(num(r[-1], "truncated")))
        for _, r in eps)
    print(f"  episodes={len(eps)}  terminated={n_term}  truncated={n_trunc}")
    print(f"  outcomes: goal={n_goal}  collision={n_coll}  timeout={n_trunc}")
    print(f"  flags mutually exclusive: {flags_ok}")
    print(f"  returns: min={min(returns):+.2f} max={max(returns):+.2f} "
          f"mean={sum(returns)/len(returns):+.2f}")
    spread = max(returns) - min(returns)
    print(f"  return spread = {spread:.2f}")
    print(f"  -> {verdict(flags_ok and spread > 1.0)}\n")

    if trace_ep is not None:
        match = [(n, r) for n, r in eps if f"ep{trace_ep:04d}" in n]
        if match:
            name, rows = match[0]
            ow = odom_to_world(rows)
            print("=" * 78)
            print(f"PER-STEP TRACE: {name}")
            print("-" * 78)
            print(f"{'t':>4} {'true_x':>8} {'true_y':>8} | {'est_x':>8} {'est_y':>8} "
                  f"{'err':>6} | {'odm_x':>8} {'odm_y':>8} {'oerr':>6} | "
                  f"{'cov_xx':>7} {'front':>6} {'rew':>7}")
            step = max(1, len(rows) // 20)
            for i in range(0, len(rows), step):
                r = rows[i]
                e = math.hypot(num(r, "est_x") - num(r, "true_x"),
                               num(r, "est_y") - num(r, "true_y"))
                oe = math.hypot(ow[i][0] - num(r, "true_x"),
                                ow[i][1] - num(r, "true_y"))
                print(f"{i:>4} {num(r,'true_x'):8.3f} {num(r,'true_y'):8.3f} | "
                      f"{num(r,'est_x'):8.3f} {num(r,'est_y'):8.3f} {e:6.3f} | "
                      f"{ow[i][0]:8.3f} {ow[i][1]:8.3f} {oe:6.3f} | "
                      f"{num(r,'cov_xx'):7.3f} {num(r,'front_dist'):6.2f} "
                      f"{num(r,'reward'):7.3f}")


if __name__ == "__main__":
    main()
