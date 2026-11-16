"""Aggregate statistics of the INPUT data (tracking + Smart Tagging) used to design my_model.py.

Reads no ground truth. Prints medians, percentiles and counts only -- never rows, coordinates,
ids or per-event values -- because the terms forbid sharing extracts of the match data.

    python analysis/data_stats.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import my_model as mm  # noqa: E402
import load_smart_tagging  # noqa: E402


def pct(values, qs=(10, 25, 50, 75, 90)):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return "n=0"
    return "n={:<6} ".format(len(values)) + "  ".join(
        "p{}={:.2f}".format(q, np.percentile(values, q)) for q in qs)


def main():
    track = mm.load_tracking()
    n = track["n_frames"]
    graded = mm.graded_mask(n)
    ball = track["ball"]
    seen = ~np.isnan(ball[:, 0])

    # Ball speed over 0.2 s (5 frames): a single-frame difference is dominated by jitter.
    lag = 5
    disp = np.full(n, np.nan)
    both = seen[lag:] & seen[:-lag]
    step = np.linalg.norm(ball[lag:] - ball[:-lag], axis=1)
    disp[:-lag][both] = step[both]
    ball_speed = disp / (lag / mm.FPS)

    out_x = (ball[:, 0] < 0) | (ball[:, 0] > mm.PITCH_LENGTH)
    out_y = (ball[:, 1] < 0) | (ball[:, 1] > mm.PITCH_WIDTH)
    out = seen & (out_x | out_y)

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_speed = np.nanmean(track["player_speed"], axis=1)
    n_players = np.sum(~np.isnan(track["player_speed"]), axis=1)

    print("== graded frames:", int(graded.sum()))
    print("ball visible            {:.1%}".format(seen[graded].mean()))
    print("ball outside the lines  {:.2%} of graded frames, {:.2%} of visible graded".format(
        out[graded].mean(), out[graded].sum() / max(1, seen[graded].sum())))
    print("ball speed m/s (0.2 s)  ", pct(ball_speed[graded]))
    print("mean player speed m/s   ", pct(mean_speed[graded]))
    print("tracked players         ", pct(n_players[graded]))

    # Ball-missing gaps inside the graded halves.
    gaps = []
    for _, first, last in mm.half_video_blocks(mm.GAME):
        miss = ~seen[first:last + 1]
        edges = np.diff(np.concatenate(([0], miss.astype(int), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        gaps.extend((ends - starts).tolist())
    gaps = np.array(gaps) / mm.FPS
    print("\n== ball-missing gaps (s): count {}, total {:.0f} s".format(len(gaps), gaps.sum()))
    for lo, hi in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 15), (15, 30), (30, 1e9)):
        sel = (gaps > lo) & (gaps <= hi)
        print("  ({:>4}, {:>4}] s   {:>5} gaps   {:>6.0f} s total".format(
            lo, hi if hi < 1e9 else "inf", int(sel.sum()), gaps[sel].sum()))

    # Smart Tagging set pieces.
    events = load_smart_tagging.load()
    pieces = [e for e in events if (e["code"] or "").strip().lower() == "set pieces"]
    print("\n== SET PIECES: {} of {} instances".format(len(pieces), len(events)))
    types = sorted({(e["tags"].get("Type") or "?") for e in pieces})
    window = 30 * mm.FPS
    offsets = np.arange(-window, 10 * mm.FPS + 1, 5 * mm.FPS)
    for kind in types + ["ALL"]:
        sel = [e for e in pieces if kind == "ALL" or (e["tags"].get("Type") or "?") == kind]
        dur = [(e["end_frame"] - e["start_frame"]) / mm.FPS for e in sel]
        vis, oob, onset_rel, onset_found = [], [], [], 0
        back_out = []
        profile = {o: [] for o in offsets}
        for e in sel:
            s, t = e["start_frame"], e["end_frame"]
            vis.append(seen[s:t].mean())
            oob.append(out[s:t].mean())
            fast = np.flatnonzero(ball_speed[s:t] > 4.0)
            onset = s + int(fast[0]) if len(fast) else None
            if onset is not None:
                onset_found += 1
                onset_rel.append((onset - s) / mm.FPS)
            anchor = onset if onset is not None else (s + t) // 2
            prev_out = np.flatnonzero(out[max(0, anchor - window):anchor])
            back_out.append((window - prev_out[-1]) / mm.FPS if len(prev_out) else np.nan)
            for o in offsets:
                lo, hi = anchor + o - mm.FPS // 2, anchor + o + mm.FPS // 2
                if 0 <= lo and hi < n:
                    profile[o].append(np.nanmedian(mean_speed[lo:hi]))
        print("\n  {} (n={})".format(kind, len(sel)))
        print("    window length s        ", pct(dur, (10, 50, 90)))
        print("    ball visible in window ", pct(vis, (10, 50, 90)))
        print("    ball out in window     ", pct(oob, (10, 50, 90)))
        print("    ball-onset >4 m/s found in {}/{} windows; s after window start {}".format(
            onset_found, len(sel), pct(onset_rel, (10, 50, 90))))
        print("    s since ball last seen outside lines (<=30 s)", pct(back_out, (10, 50, 90)))
        print("    median player speed (m/s) at offset from restart (s):")
        print("      " + "  ".join("{:+d}:{:.2f}".format(int(o / mm.FPS), np.median(profile[o]))
                                   for o in offsets if profile[o]))

    # Kickoffs relative to the half starts (aggregate: one number per half, seconds).
    print("\n== kick-off (start) tags vs graded half start")
    for _, first, last in mm.half_video_blocks(mm.GAME):
        ko = [e for e in pieces if (e["tags"].get("Type") or "").lower() == "kick off (start)"
              and first - 10 * mm.FPS <= e["start_frame"] <= last]
        print("  half starting at graded frame: {} kickoff tag(s); tag start minus half start "
              "{}".format(len(ko), ["{:+.1f} s".format((e["start_frame"] - first) / mm.FPS)
                                    for e in ko]))


if __name__ == "__main__":
    main()
