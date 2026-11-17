"""Score two variants of my_model.py side by side on the public five minutes.

    python analysis/compare.py abc abcd      # old steps, new steps

Builds both variants from one parse of the tracking, writes them to a temporary folder, runs the
official scorer (evaluation/run_evaluation.py) on that folder only, and prints the scoreboard
numbers (spellF1, macroF1, poss, dur) and the per-class F1 (H / A / D) for each. The scorer's
per-submission reports go to the temporary folder as well and are deleted unread: they list
ground-truth transition frames.

A sanity check, not a tuning loop: the public window has 17 possessions and a spellF1 noise
band of ~0.33, and nothing here feeds back into PARAMS.

DATA SAFETY: prints scorer metrics and frame counts only -- never labels, rows or frame numbers.
"""

import contextlib
import csv
import io
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import my_model as mm  # noqa: E402

PUBLIC_GRADED = 7500   # the public ground truth is the first 5 minutes of the first half


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python analysis/compare.py OLD_STEPS NEW_STEPS")
    variants = [("old", sys.argv[1]), ("new", sys.argv[2])]

    track = mm.load_tracking()
    labels = {}
    for tag, steps in variants:
        with contextlib.redirect_stdout(io.StringIO()):
            labels[tag] = mm.predict(track, steps=steps)

    graded = mm.graded_mask(track["n_frames"])
    first = mm.half_video_blocks(mm.GAME)[0][1]
    public = np.zeros(track["n_frames"], dtype=bool)
    public[first:first + PUBLIC_GRADED] = True
    changed = labels["old"] != labels["new"]

    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, "results")
        os.makedirs(results)
        names = {}
        for tag, steps in variants:
            names[tag] = "{}_{}".format(tag, steps or "baseline")
            mm.baseline.write_submission(os.path.join(results, names[tag] + ".csv"), labels[tag])
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
             "--results", results, "--out", tmp, "--no-detail", "--bootstrap", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if out.returncode != 0:
            # The scorer's own messages are reasons, not data; show only its last line.
            sys.exit("scorer failed: " + out.stdout.strip().splitlines()[-1])
        truth = next((line for line in out.stdout.splitlines()
                      if line.startswith("ground truth has")), "")
        with open(os.path.join(tmp, "scoreboard.csv")) as handle:
            board = {row["submission"]: row for row in csv.DictReader(handle)}

    print("public 5 minutes -- sanity check only. {}".format(truth.split("<-")[0].strip()))
    print("{:<14} {:>7} {:>7} {:>5} {:>6}   {:>6} {:>6} {:>6}".format(
        "variant", "spellF1", "macroF1", "poss", "dur", "F1 H", "F1 A", "F1 D"))
    for tag, _ in variants:
        row = board[names[tag]]
        print("{:<14} {:>7.4f} {:>7.4f} {:>5} {:>6.1f}   {:>6.4f} {:>6.4f} {:>6.4f}".format(
            names[tag], float(row["spell_f1_mean"]), float(row["macro_f1"]), row["poss"],
            float(row["dur"]), float(row["f1_H"]), float(row["f1_A"]), float(row["f1_D"])))
    print("graded frames that differ between the two: {} ({} inside the public window)".format(
        int(np.sum(changed & graded)), int(np.sum(changed & public))))


if __name__ == "__main__":
    main()
