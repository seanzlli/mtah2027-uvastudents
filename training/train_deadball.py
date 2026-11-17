"""Fit and export the step (g) dead-ball tree. Run ONCE, offline; not part of the submission.

    test/databallpy_comparison/.venv/bin/python training/train_deadball.py

Needs databallpy and scikit-learn. The submission needs NEITHER: it reads the exported
models/deadball_tree.json and walks it with numpy in deadball.py.

Training data: DFL/Sportec open Bundesliga matches, which publish a per-frame `ball_status`
(alive/dead). No labels from the graded match are used (RULES.md). Features and the per-match
normalisation are imported from deadball.py, so training and inference cannot drift apart --
an earlier version of this work exported a model trained on raw values while inference
normalised, and the mismatch silently cost most of the gain.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "starter"))

import deadball as db                                    # noqa: E402
import baseline                                          # noqa: E402
from databallpy import get_open_game                     # noqa: E402

TRAIN = ["J03WMX", "J03WN1"]
HELDOUT = ["J03WPY", "J03WOH"]
MAX_DEPTH = 4          # 31 nodes; depth 10 scores WORSE on held-out matches (overfits)
MIN_LEAF = 500         # ~20 s: a leaf must describe a sustained situation, not a blip
FPS = 25
OUT = os.path.join(REPO, "models", "deadball_tree.json")


def load_dfl(game_id):
    """DFL match -> (normalised features, dead labels, names), graded frames only."""
    game = get_open_game(provider="dfl", game_id=game_id)
    td = game.tracking_data
    n = len(td)
    hx = sorted(c for c in td.columns if c.startswith("home_") and c.endswith("_x"))
    ax = sorted(c for c in td.columns if c.startswith("away_") and c.endswith("_x"))
    stems = [c[:-2] for c in hx] + [c[:-2] for c in ax]
    players = np.full((n, len(stems), 2), np.nan)
    for s, stem in enumerate(stems):
        players[:, s, 0] = td[stem + "_x"].to_numpy()
        players[:, s, 1] = td[stem + "_y"].to_numpy()
    team = np.array([baseline.HOME] * len(hx) + [baseline.AWAY] * len(ax), dtype=np.int8)

    X, names = db.player_features(players, team, baseline.HOME, FPS)
    graded = np.isin(td["period_id"].to_numpy(), (1, 2))
    # normalise with the same function inference uses, over this match's graded frames
    Xn = db.rank_normalise(X, graded)
    dead = td["ball_status"].to_numpy().astype(object) == "dead"
    ok = graded & ~np.isnan(Xn).any(axis=1)
    return Xn[ok], dead[ok], names


Xs, ys, names = [], [], None
for gid in TRAIN:
    X, y, names = load_dfl(gid)
    print("train {}: {} graded frames, {:.1%} dead".format(gid, len(y), y.mean()))
    Xs.append(X); ys.append(y)
Xtr, ytr = np.vstack(Xs), np.concatenate(ys)

clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=0)
clf.fit(Xtr, ytr)
t = clf.tree_
value = np.asarray(t.value)

art = {
    "_comment": "Dead-ball decision tree for step (g). Fitted on DFL/Sportec open matches "
                "using their per-frame ball_status. No labels from the graded match were "
                "used. Features are ball-free player speed/shape quantities, each converted "
                "to its percentile WITHIN the match being labelled -- see deadball.py.",
    "normalisation": "rank",
    "trained_on": TRAIN, "held_out": HELDOUT,
    "max_depth": MAX_DEPTH, "min_samples_leaf": MIN_LEAF,
    "features": list(names),
    "children_left": t.children_left.astype(int).tolist(),
    "children_right": t.children_right.astype(int).tolist(),
    "feature": t.feature.astype(int).tolist(),
    "threshold": t.threshold.astype(float).tolist(),
    "dead_prob": [float(v[0][1] / v[0].sum()) for v in value],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(art, fh, indent=1)
print("\nwrote {} -- {} nodes, {} features, normalisation={}".format(
    os.path.relpath(OUT, REPO), t.node_count, len(names), art["normalisation"]))

print("\nheld-out per-frame check (2 matches never used for fitting):")
for gid in HELDOUT:
    Xh, yh, _ = load_dfl(gid)
    prob = db.tree_probability(art, lambda nm, Xh=Xh: Xh[:, names.index(nm)])
    m = db.enforce_min_run(prob >= 0.30, 2 * FPS)
    p, r, f, _ = precision_recall_fscore_support(yh, m, average="binary", zero_division=0)
    print("  {}: precision {:.3f}  recall {:.3f}  F1 {:.3f}  macroF1 {:.4f}".format(
        gid, p, r, f, f1_score(yh, m, average="macro")))

print("\n=== the entire model ===")


def show(node=0, depth=0):
    pad = "  " * depth
    if t.children_left[node] == -1:
        v = value[node][0]
        print("{}-> dead p={:.3f}   ({} frames)".format(
            pad, v[1] / v.sum(), int(t.n_node_samples[node])))
        return
    print("{}if {} <= {:.3f} (percentile):".format(
        pad, names[t.feature[node]], t.threshold[node]))
    show(t.children_left[node], depth + 1)
    print("{}else:".format(pad))
    show(t.children_right[node], depth + 1)


show()
