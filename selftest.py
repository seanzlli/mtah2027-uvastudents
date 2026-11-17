"""One-command self-test for the UvAStudents submission.

    python selftest.py            # ~20 s; exit code 0 = no FAIL

Checks, in order:
  1. baseline port      my_model.py --steps "" equals baseline_smooth50 on every graded frame
  2. submission valid   the full model passes the official validator (same checks as the scorer)
  3. invariants         labels only H/A/D/X; X everywhere outside the halves; D before each
                        half's kick-off strike; no unexpected abstentions
  4. plausibility       dead-ball share and possession count in a football-realistic range (WARN)
  5. reproducible       re-parsing the raw tracking gives identical labels
  6. on-disk file       evaluation/results/UvAStudents.csv is what the current code produces
  7. public score       the published baseline numbers are reproduced (pipeline alignment), and
                        the submission's public scores are printed for information only --
                        17 possessions, noise band ~0.33, NOT a tuning target
  8. repo hygiene       no match data, cache or submission CSV tracked by git; notebooks with
                        saved outputs flagged before the repository goes public

DATA SAFETY: prints PASS/WARN/FAIL with counts and scorer metrics only -- never rows,
coordinates, ids or frame numbers.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import my_model as mm  # noqa: E402
import possession_metrics as pm  # noqa: E402

BASELINE_CSV = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SUBMISSION = os.path.join(mm.RESULTS_DIR, mm.TEAM_NAME + ".csv")
# Published in README.md / EVALUATION.md for baseline_smooth50 on the public window.
PUBLISHED_BASELINE_PUBLIC = {"macro_f1": 0.6207, "spell_f1_mean": 0.3256}

# Football-realistic ranges, for WARN only. Effective playing time in top-flight football is
# ~55-65 min of ~95-100, so the ball is dead for very roughly a third to a half of graded time.
# Possession counts: a few hundred team spells per match is normal; >600 means flicker.
DEAD_SHARE_RANGE = (0.25, 0.55)
POSSESSIONS_RANGE = (100, 600)
MEAN_POSSESSION_S_RANGE = (5.0, 30.0)
MAX_GRADED_X_SHARE = 0.02

results = []


def report(status, name, detail=""):
    results.append(status)
    print("{:<5} {:<20} {}".format(status, name, detail))


def quiet_predict(track, steps):
    with contextlib.redirect_stdout(io.StringIO()):
        return mm.predict(track, steps=steps)


def codes_of(labels):
    return pm.label_codes(np.asarray(labels, dtype=object))


def check_baseline(track, graded):
    ref = pm.label_codes(pm.read_label_csv(BASELINE_CSV))
    mine = codes_of(quiet_predict(track, ""))
    diff = int(np.sum(mine[graded] != ref[graded]))
    report("PASS" if diff == 0 else "FAIL", "baseline port",
           "{} of {} graded frames differ from baseline_smooth50".format(diff, int(graded.sum())))
    return mine


def check_valid(labels):
    with tempfile.TemporaryDirectory() as tmp:
        path = mm.baseline.write_submission(os.path.join(tmp, "s.csv"), labels)
        reason = pm.validate_submission(pm.read_label_csv(path),
                                        mm.source_frames("tactical"))
    report("FAIL" if reason else "PASS", "submission valid", reason or "official checks pass")


def check_invariants(track, labels, graded):
    bad = sorted(set(np.unique(labels)) - set("HADX"))
    report("FAIL" if bad else "PASS", "label alphabet",
           "unexpected labels present" if bad else "only H/A/D/X")

    outside = int(np.sum(labels[~graded] != "X"))
    report("PASS" if outside == 0 else "FAIL", "ungraded = X",
           "{} ungraded frames are not X".format(outside))

    if "f" in mm.DEFAULT_STEPS:
        # Step f must be exactly the scorer's own spellF1 fill, applied to the model without it.
        n_x = int(np.sum(labels[graded] == "X"))
        unfilled = quiet_predict(track, mm.DEFAULT_STEPS.replace("f", ""))
        halves = [(first, last + 1) for _, first, last in mm.half_video_blocks(mm.GAME)]
        diff = int(np.sum(pm.fill_abstentions(codes_of(unfilled), halves) != codes_of(labels)))
        report("PASS" if n_x == 0 and diff == 0 else "FAIL", "graded abstentions",
               "{} graded frames are X; {} frames differ from the scorer's spellF1 fill".format(
                   n_x, diff))
    else:
        x_share = float(np.mean(labels[graded] == "X"))
        report("PASS" if x_share <= MAX_GRADED_X_SHARE else "WARN", "graded abstentions",
               "{:.2%} of graded frames are X (limit {:.0%})".format(x_share, MAX_GRADED_X_SHARE))

    # Before each half's kick-off strike the ball is dead (found from tags + tracking; for the
    # second half the kick-off published in games.py when step d is on).
    speed = mm.ball_speed(track)
    kickoff_2nd = mm.second_half_kickoff() if "d" in mm.DEFAULT_STEPS else None
    not_dead, halves_checked = 0, 0
    for kind, start, end in mm.load_restarts():
        if kind != "kick off (start)":
            continue
        r, _ = mm.restart_frame(speed, start, end)
        half_start = next((first for _, first, last in mm.half_video_blocks(mm.GAME)
                           if r <= last), None)
        if half_start is None:
            continue
        if kickoff_2nd is not None and half_start == kickoff_2nd[0]:
            r = kickoff_2nd[1]
        halves_checked += 1
        not_dead += int(np.sum(labels[half_start:r] != "D"))
    report("PASS" if halves_checked == 2 and not_dead == 0 else "FAIL", "pre-kick-off = D",
           "{} halves checked, {} graded frames before the strike are not D".format(
               halves_checked, not_dead))


def check_plausible(labels, graded):
    codes = codes_of(labels)
    dead_share = float(np.mean(labels[graded] == "D"))
    lo, hi = DEAD_SHARE_RANGE
    report("PASS" if lo <= dead_share <= hi else "WARN", "dead-ball share",
           "{:.1%} of graded frames (football-plausible {:.0%}-{:.0%})".format(
               dead_share, lo, hi))

    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    n_alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
    mean_s = n_alive / float(mm.FPS) / len(spells) if len(spells) else 0.0
    ok = (POSSESSIONS_RANGE[0] <= len(spells) <= POSSESSIONS_RANGE[1]
          and MEAN_POSSESSION_S_RANGE[0] <= mean_s <= MEAN_POSSESSION_S_RANGE[1])
    report("PASS" if ok else "WARN", "possession count",
           "{} possessions of mean {:.1f} s (plausible {}-{}, {:.0f}-{:.0f} s)".format(
               len(spells), mean_s, POSSESSIONS_RANGE[0], POSSESSIONS_RANGE[1],
               *MEAN_POSSESSION_S_RANGE))


def check_reproducible(labels):
    with contextlib.redirect_stdout(io.StringIO()):
        fresh = mm.load_tracking()                   # a second, independent parse
    again = quiet_predict(fresh, mm.DEFAULT_STEPS)
    diff = int(np.sum(again != labels))
    report("PASS" if diff == 0 else "FAIL", "reproducible",
           "{} frames differ after re-parsing the raw tracking".format(diff))


def check_on_disk(labels):
    if not os.path.isfile(SUBMISSION):
        report("WARN", "on-disk file", "missing -- run: python my_model.py")
        return
    disk = pm.label_codes(pm.read_label_csv(SUBMISSION))
    diff = int(np.sum(disk != codes_of(labels)))
    report("PASS" if diff == 0 else "FAIL", "on-disk file",
           "{} frames differ from the current code{}".format(
               diff, "" if diff == 0 else " -- re-run: python my_model.py"))


def check_public(base_labels, labels):
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    base = pm.evaluate(gt, base_labels)
    off = max(abs(base[k] - v) for k, v in PUBLISHED_BASELINE_PUBLIC.items())
    report("PASS" if off < 5e-5 else "FAIL", "public alignment",
           "baseline port scores macroF1 {:.4f} spellF1 {:.4f} (published {:.4f} / {:.4f})".format(
               base["macro_f1"], base["spell_f1_mean"],
               PUBLISHED_BASELINE_PUBLIC["macro_f1"], PUBLISHED_BASELINE_PUBLIC["spell_f1_mean"]))
    mine = pm.evaluate(gt, codes_of(labels))
    report("INFO", "public score",
           "macroF1 {:.4f} spellF1 {:.4f}, {} possessions (truth {}) -- 17 possessions, "
           "noise ~0.33: not a tuning target".format(
               mine["macro_f1"], mine["spell_f1_mean"], mine["n_pred_spells"],
               mine["n_gt_spells"]))


def check_hygiene():
    try:
        tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT,
                                          universal_newlines=True).split("\n")
    except (OSError, subprocess.CalledProcessError):
        report("WARN", "repo hygiene", "git not available, not checked")
        return
    allowed_csv = {"evaluation/gt/BAR-ATH_possession_gt_public.csv"} | {
        "evaluation/results/baseline_{}.csv".format(v)
        for v in ("raw", "smooth25", "smooth50", "smooth125")}
    data_suffixes = (".npz", ".mp4", "_atd.txt", "_atd.xml", "_smart_tagging.xml")
    leaks = [p for p in tracked if p and (
        (p.startswith("data/") and p != "data/README.md") or p.startswith("cache/")
        or p.endswith(data_suffixes)
        or (p.endswith(".csv") and p not in allowed_csv))]
    # File paths are repository names, not match data, so they are safe to print.
    report("PASS" if not leaks else "FAIL", "repo hygiene",
           "{} tracked files look like match data, cache or submissions{}".format(
               len(leaks), (": " + ", ".join(leaks)) if leaks else ""))

    check_notebooks([p for p in tracked if p.endswith(".ipynb")])


def cells_with_outputs(text):
    """How many cells of a notebook (JSON text) carry saved outputs. Counts only."""
    try:
        return sum(1 for cell in json.loads(text).get("cells", []) if cell.get("outputs"))
    except ValueError:
        return 0


def check_notebooks(indexed):
    """Saved outputs in the version git would commit -- the index (staged, else last commit).

    Outputs computed from the match data must not reach the public repository. A working copy
    with outputs that are NOT staged is fine, and is only mentioned so it is not added by
    accident with `git add -A` / `git commit -a`.
    """
    staged, local_only = [], []
    for path in indexed:
        try:
            committed = subprocess.check_output(["git", "show", ":" + path], cwd=ROOT,
                                                universal_newlines=True,
                                                stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            continue
        if cells_with_outputs(committed):
            staged.append(path)
            continue
        local = os.path.join(ROOT, path)
        if os.path.isfile(local):
            with open(local) as handle:
                if cells_with_outputs(handle.read()):
                    local_only.append(path)
    if staged:
        report("WARN", "notebook outputs",
               "git would commit saved outputs in {} -- clear them or unstage "
               "(git restore --staged <file>)".format(", ".join(staged)))
    else:
        report("PASS", "notebook outputs",
               "none in what git would commit{}".format(
                   "; unstaged outputs in your working copy of {} -- do not git add it, and "
                   "avoid git add -A / git commit -a".format(", ".join(local_only))
                   if local_only else ""))


def main():
    track = mm.load_tracking()
    graded = mm.graded_mask(track["n_frames"])
    labels = quiet_predict(track, mm.DEFAULT_STEPS)

    base_codes = check_baseline(track, graded)
    check_valid(labels)
    check_invariants(track, labels, graded)
    check_plausible(labels, graded)
    check_reproducible(labels)
    check_on_disk(labels)
    check_public(base_codes, labels)
    check_hygiene()

    fails, warns = results.count("FAIL"), results.count("WARN")
    print("\n{} checks: {} FAIL, {} WARN".format(len(results), fails, warns))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
