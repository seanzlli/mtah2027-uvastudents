"""Step (g): dead ball from player positions alone, via a small decision tree.

Why this exists. Our rule-based dead-ball detection (steps a, c, d, e) has high precision
but poor recall: on the public window it catches only 37% of dead frames, and the frames it
misses are labelled H or A, which also costs the possession classes. Untagged stoppages --
fouls, injuries, substitutions -- are missed entirely, because nothing marks them.

What this is. A single decision tree of depth 4 (31 nodes), fitted on DFL/Sportec open
Bundesliga matches that publish a per-frame ball_status. NO labels from the graded match are
used, directly or indirectly (RULES.md). The tree is committed as models/deadball_tree.json:
plain thresholds, walked here in numpy. The submission needs neither scikit-learn nor
databallpy -- only this file and that JSON.

The features are deliberately BALL-FREE: the ball is missing on 41% of graded frames, so a
detector that needs it cannot help. They are player speeds and team shape only.

Per-match normalisation is essential, not cosmetic. The tree's splits are positions within
a match's own distribution, not absolute m/s. Tracking providers differ in how fast they read
the same football -- this match's field speed runs ~0.14 m/s slower than the training
matches' -- so an absolute threshold would silently capture a different fraction of frames.
Each feature is therefore converted to its percentile within the match being labelled, using
only that match's inputs and no labels. training/train_deadball.py applies the identical
transform, importing it from here so the two cannot drift apart.
"""

import json
import os
import warnings

import numpy as np

ARTIFACT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "models", "deadball_tree.json")

# Thresholds inside the feature definitions below. These are not tuned: they are the same
# football quantities used elsewhere in the model (walking pace, standing, running).
WALK, STAND, RUN = 1.2, 0.5, 4.0
TRAILING_WINDOWS_S = (1, 2, 5, 10)


def _trailing_mean(x, w):
    """Mean of x over the trailing w frames, ignoring NaN."""
    valid = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(x)) + 1
    lo = np.maximum(0, idx - w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (cs[idx] - cs[lo]) / np.maximum(cn[idx] - cn[lo], 1)


def player_features(players, team_of_slot, home_code, fps):
    """Ball-free per-frame features. players is (n, slots, 2) in metres, corner origin.

    Every feature is an aggregate over the player axis, so the order of the slots within a
    team is irrelevant -- only which team each slot belongs to matters. All of them are
    translation- and origin-free (speeds, spreads, extents, centroid differences), so a
    provider's coordinate convention cannot shift them.
    """
    allX, allY = players[:, :, 0], players[:, :, 1]
    home = team_of_slot == home_code
    hX, hY = allX[:, home], allY[:, home]
    aX, aY = allX[:, ~home], allY[:, ~home]

    dx = np.diff(allX, axis=0, prepend=allX[:1])
    dy = np.diff(allY, axis=0, prepend=allY[:1])
    speed = np.sqrt(dx * dx + dy * dy) * fps

    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        # frames where a team has no tracked player give all-NaN slices; the caller
        # replaces the resulting NaN, so the warnings are noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_sp = np.nanmean(speed, axis=1)
        cols = [
            ("mean_speed", mean_sp),
            ("median_speed", np.nanmedian(speed, axis=1)),
            ("max_speed", np.nanmax(speed, axis=1)),
            ("std_speed", np.nanstd(speed, axis=1)),
            ("frac_walk", np.nanmean(speed < WALK, axis=1)),
            ("frac_stand", np.nanmean(speed < STAND, axis=1)),
            ("frac_run", np.nanmean(speed > RUN, axis=1)),
            ("spread_x", np.nanstd(allX, axis=1)),
            ("spread_y", np.nanstd(allY, axis=1)),
            ("width", np.nanmax(allX, axis=1) - np.nanmin(allX, axis=1)),
            ("height", np.nanmax(allY, axis=1) - np.nanmin(allY, axis=1)),
            ("team_sep_x", np.abs(np.nanmean(hX, axis=1) - np.nanmean(aX, axis=1))),
            ("team_sep_y", np.abs(np.nanmean(hY, axis=1) - np.nanmean(aY, axis=1))),
            ("home_compact", np.nanstd(hX, axis=1) + np.nanstd(hY, axis=1)),
            ("away_compact", np.nanstd(aX, axis=1) + np.nanstd(aY, axis=1)),
        ]
        frac_walk = cols[4][1]
        max_sp = cols[2][1]
        team_sep = cols[11][1]
        spread_x = cols[7][1]
        for w_s in TRAILING_WINDOWS_S:
            w = int(w_s * fps)
            cols += [
                ("mean_speed_t{}s".format(w_s), _trailing_mean(mean_sp, w)),
                ("frac_walk_t{}s".format(w_s), _trailing_mean(frac_walk, w)),
                ("max_speed_t{}s".format(w_s), _trailing_mean(max_sp, w)),
                ("team_sep_t{}s".format(w_s), _trailing_mean(team_sep, w)),
                ("spread_x_t{}s".format(w_s), _trailing_mean(spread_x, w)),
            ]
    names = [n for n, _ in cols]
    return np.column_stack([v for _, v in cols]), names


def rank_normalise(X, reference_rows):
    """Each column replaced by its percentile within this match (0..1).

    reference_rows selects the frames whose distribution defines the percentiles -- the
    graded frames of the match being labelled. No labels are involved: this reads only the
    model's own inputs. See the module docstring for why absolute values do not transfer.
    """
    ref = X[reference_rows]
    ref = ref[~np.isnan(ref).any(axis=1)]
    out = np.empty_like(X)
    n = max(len(ref), 1)
    for j in range(X.shape[1]):
        srt = np.sort(ref[:, j])
        out[:, j] = np.searchsorted(srt, X[:, j], side="right") / float(n)
    return out


def tree_probability(artifact, X_by_name):
    """Walk the exported tree. X_by_name(name) returns that feature's column."""
    left = np.asarray(artifact["children_left"], dtype=np.int64)
    right = np.asarray(artifact["children_right"], dtype=np.int64)
    feat = np.asarray(artifact["feature"], dtype=np.int64)
    thr = np.asarray(artifact["threshold"], dtype=np.float64)
    prob = np.asarray(artifact["dead_prob"], dtype=np.float64)
    X = np.column_stack([X_by_name(nm) for nm in artifact["features"]])
    node = np.zeros(len(X), dtype=np.int64)
    while True:
        internal = left[node] != -1
        if not internal.any():
            break
        idx = np.flatnonzero(internal)
        here = node[idx]
        go_left = X[idx, feat[here]] <= thr[here]
        node[idx] = np.where(go_left, left[here], right[here])
    return prob[node]


def enforce_min_run(mask, frames, exempt=None):
    """Drop runs shorter than `frames`, for both states.

    exempt: optional boolean mask. A dead run overlapping an exempt frame is never dropped,
    however short. Use it for the edges of a graded half: a stoppage clipped by the start of
    the graded window is censored, not brief -- the first half is graded from its own
    kick-off, so only a frame or two of its pre-kick-off dead ball is inside the window.

    Laws 8, 13, 15-17: every restart needs the ball stationary and the taker set, so a real
    stoppage lasts at least 2 s -- and play does not resume for less than that either. This
    is `min_dead` from PARAMS, applied to the detector's output. Without it the detector
    sprinkles short stoppages through live play and splits real possessions, which spellF1
    punishes far more than the frames are worth.
    """
    out = mask.copy()
    for state in (True, False):
        edges = np.diff(np.concatenate(([0], (out == state).astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a < frames:
                if state and exempt is not None and exempt[a:b].any():
                    continue
                out[a:b] = not state
    return out


def load_artifact(path=ARTIFACT):
    with open(path) as handle:
        return json.load(handle)


def dead_mask(track, graded, home_code, fps, dead_threshold, min_run_frames,
              artifact=None):
    """Per-frame dead-ball mask from the tree. graded selects the frames to normalise over."""
    art = artifact if artifact is not None else load_artifact()
    if art.get("normalisation") != "rank":
        raise ValueError("artifact normalisation is {!r}, expected 'rank'; retrain with "
                         "training/train_deadball.py".format(art.get("normalisation")))
    X, names = player_features(track["players"], track["team_of_slot"], home_code, fps)
    Xn = rank_normalise(X, graded)
    index = {nm: i for i, nm in enumerate(names)}
    missing = [nm for nm in art["features"] if nm not in index]
    if missing:
        raise ValueError("artifact wants features this build does not produce: {}".format(
            missing[:5]))
    column = lambda nm: np.nan_to_num(Xn[:, index[nm]], nan=0.0)   # noqa: E731
    prob = tree_probability(art, column)
    return enforce_min_run((prob >= dead_threshold) & graded, min_run_frames) & graded
