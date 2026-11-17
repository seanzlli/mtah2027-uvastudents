"""UvAStudents -- ball possession for More than a Hack 2027, Phase 1.

    python my_model.py                 # writes evaluation/results/UvAStudents.csv

One command, from scratch, deterministic: every run parses the raw tracking and the Smart Tagging
from data/ (a few seconds), keeps all numbers in float64 exactly as written in the files, and
uses no cache, no randomness and no state from earlier runs. numpy + the standard library only;
exact versions are pinned in requirements.txt.

Every parameter lives in PARAMS below with the football reason for its value. None of them was
chosen by scoring against the public five minutes; see NOTES.md.

DATA SAFETY: this script prints summary counts only. It never prints rows, coordinates or ids,
and parse errors report a line number, not the line.
"""

import argparse
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import (GAME, GAMES, anchor_video_frame, half_video_blocks, require,  # noqa: E402
                   source_frames)
import baseline  # noqa: E402  (starter/baseline.py: team table and CSV writer)

TEAM_NAME = "UvAStudents"
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
# The submitted model: every step that passed its sanity check (NOTES.md). Letters as in predict().
DEFAULT_STEPS = "abcde"

FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]
HOME, AWAY, NOBODY = baseline.HOME, baseline.AWAY, baseline.NOBODY

# --------------------------------------------------------------------------------------------
# Parameters. Units are stated; frames are at 25 fps.
# --------------------------------------------------------------------------------------------
PARAMS = {
    # Step 0 -- the published baseline, kept as the starting point: centred majority vote of
    # +/- 2 s over the nearest-player team. Not ours to tune; it is the reference floor.
    "baseline_window": 50,

    # ---- shared signals ----------------------------------------------------------------------
    # Ball speed is measured over 0.2 s, not frame to frame: automatic ball positions jitter by
    # tens of centimetres, which at 25 fps would read as several m/s of fake speed.
    "ball_speed_lag": 5,
    # Faster than the hardest shot in football (~35 m/s) is a tracking jump, not a ball.
    "max_ball_speed": 40.0,
    # The whole ball must cross the line to be out (Law 9). 1 m of slack absorbs tracking error
    # near the lines and a ball in the air projected slightly beyond them.
    "out_margin": 1.0,
    # Field-average player speed, over the trailing 2 s. Walking is ~1-1.4 m/s; in a stoppage
    # most players walk or stand, so the field average sits at or below walking pace. In live
    # play it is around 2 m/s or more because the players near the ball are running.
    "player_speed_window": 50,
    "stop_player_speed": 1.2,

    # ---- step (a): dead ball before Smart Tagging restarts ------------------------------------
    # The restart is the first time the ball is struck: faster than 4 m/s (any restart pass
    # or throw is 5+ m/s; a ball being carried or placed moves under 2 m/s) after at least
    # 1 s without such motion (so a ball already rolling out over the line is not the restart).
    "restart_ball_speed": 4.0,
    "restart_quiet_before": 25,
    # If the tracking never sees the strike, assume it is 3.5 s into the tag: the starter
    # notebook shows the kick-off tags opening ~3.5 s before the real kick-off.
    "restart_fallback_lead": 87,
    # Every restart needs the ball stationary/the taker set first (Laws 8, 13, 15-17), so a
    # stoppage is at least 2 s long.
    "min_dead": 50,
    # A ball is "stationary" below 1 m/s: true speed ~0, plus the ~0.5-1 m/s that position
    # jitter produces over 0.2 s.
    "stationary_ball_speed": 1.0,
    # Walking back from a restart, up to 1 s without any dead-ball evidence is tolerated (a
    # player jogging over to take the throw, a jittery ball reading); longer means live play.
    "evidence_tolerance": 25,
    # Longest plausible stoppage before each restart type, in seconds -- the backward search
    # never goes further. Typical LaLiga times: throw-in 10-20 s, goal kick 15-30 s, corner
    # 20-40 s, free kick 20-60 s (foul, card, wall), kick-off after a goal 60-120 s
    # (celebration, VAR check). The match kick-offs run back to the start of the half.
    "restart_cap_s": {
        "throw-in": 20,
        "goal kick": 30,
        "corner kick": 40,
        "free kick": 60,
        "kick off (after goal)": 120,
        "kick off (start)": None,
    },

    # ---- step (b): hysteresis possession switching -------------------------------------------
    # A team is "on the ball" when its nearest player is within 2 m of it -- playing distance:
    # a dribbler keeps the ball within about a stride (1.5-2 m) -- AND at least 1 m closer than
    # the nearest opponent, i.e. clearly closer rather than in a 50/50 duel. 1 m is well above
    # the tracker's jitter. When neither holds the ball is loose or contested, and possession
    # stays with the last team (CHALLENGE.md: "possession stays with the last team that had it").
    "control_radius": 2.0,
    "control_margin": 1.0,
    # Possession changes only once the other team has been on the ball for 1 s in total with no
    # touch by the holder in between. A touch, deflection or lost aerial duel lasts under 0.3 s
    # and is not control; a controlled reception plus the next action, or an interception
    # followed by a team-mate's touch, easily reaches 1 s. The change is then dated back to the
    # first touch of that run -- that is the moment control was gained, and spellF1 punishes a
    # late boundary heavily (1 s late = 0.65).
    "confirm_frames": 25,

    # ---- step (c): missing-ball handling -----------------------------------------------------
    # Short gaps are a ball in flight or briefly occluded: carry possession. A 50-60 m diagonal
    # is in the air for about 3 s, so up to 3 s is carried unconditionally.
    "carry_max_gap": 75,
    # Longer gaps are a stoppage by default -- in this tracking the ball is mostly missing
    # because play has stopped (input data: field speed median 1.0 m/s inside long gaps vs
    # 2.0 m/s with the ball visible; most restart stoppages lie inside a gap). A gap stretch is
    # treated as live play only if the field moves at live-play tempo -- 2 m/s, the typical
    # field average with the ball in play -- for at least 2 s; there possession is carried.
    "live_tempo_speed": 2.0,
    "min_live_in_gap": 50,

    # ---- step (e): more dead-ball evidence ---------------------------------------------------
    # Ball seen beyond the lines (out_margin above) for at least 0.5 s: it has left the pitch and
    # stays dead until play restarts (Law 9) -- the next strike of the ball inside the pitch --
    # or at most as long as the longest restart that follows the ball going out (corner, 40 s,
    # from restart_cap_s). 0.5 s because a lofted ball's ground projection or a tracking jump
    # overshoots a line for a few tenths of a second, while a ball that has really gone out
    # stays out: rolling away, being fetched, held for the throw. A run while the whole field
    # moves at live-play tempo (live_tempo_speed, as in step c) is not trusted: play is going
    # on, so the "ball" outside is a spare ball by the touchline or a tracking error.
    "out_min_frames": 13,
    # Law 1: 7.32 m between the posts. A goal is the ball over the goal line between them
    # (Law 10); out_margin is added for tracking error, as for the other lines.
    "goal_width": 7.32,
}

# Restarts that follow the ball leaving the pitch (Laws 15-17); their caps bound step (e).
OUT_OF_PLAY_RESTARTS = ("throw-in", "goal kick", "corner kick")


# --------------------------------------------------------------------------------------------
# Loading the tracking
# --------------------------------------------------------------------------------------------

def _parse_raw_atd(raw_path, n_frames, n_slots):
    """Raw ATD -> (ball_xy[n,2], player_xys[n,slots,3]) in normalised units, NaN = missing.

    Line format, per starter/baseline.py: "<frame_id>:<x,y,s>;<x,y,s>;...:<ball group>".
    frame_id counts from 1, so row frame_id-1 is the video frame (DATA.md).
    """
    # float64 throughout: the values are exactly those written in the file, so no threshold
    # comparison can flip because of rounding on another machine or numpy build.
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float64)
    bad_lines = 0
    with open(raw_path) as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split(":")
            try:
                if len(parts) != 3:
                    raise ValueError
                row = int(parts[0]) - 1
                values = [float(v) for v in parts[1].replace(";", ",").split(",")]
                if len(values) != n_slots * 3:
                    raise ValueError
                players[row] = np.asarray(values, dtype=np.float64).reshape(n_slots, 3)
                ball_fields = parts[2].split(",")
                ball[row] = (float(ball_fields[0]), float(ball_fields[1]))
            except (ValueError, IndexError):
                # Deliberately no content in the message: the data must not be echoed.
                bad_lines += 1
                if bad_lines <= 3:
                    print("  warning: could not parse ATD line {}".format(line_no))
    if bad_lines:
        print("  {} unparseable ATD lines left as missing".format(bad_lines))
    return ball, players


def load_tracking():
    """Tracking in metres, parsed fresh from the raw file. Dict of arrays by video frame."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames("tactical")
    team_of_slot = baseline.team_of_each_track(meta_path)
    ball, players = _parse_raw_atd(raw_path, n_frames, len(team_of_slot))

    scale = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float64)
    return {
        "ball": ball * scale,                       # (n, 2) metres
        "players": players[:, :, :2] * scale,       # (n, slots, 2) metres
        "player_speed": players[:, :, 2],           # (n, slots) m/s
        "team_of_slot": team_of_slot,               # (slots,) HOME / AWAY
        "n_frames": n_frames,
    }


# --------------------------------------------------------------------------------------------
# Per-frame evidence
# --------------------------------------------------------------------------------------------

def ball_distances(track):
    """Distance (m) from the ball to the nearest home and nearest away player, plus the team of
    the single nearest player (the baseline's answer). NaN / NOBODY where the ball is missing."""
    diff = track["players"] - track["ball"][:, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))                  # (n, slots), NaN if missing
    team = track["team_of_slot"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows = ball or team missing
        d_home = np.nanmin(np.where(team[None, :] == HOME, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == AWAY, dist, np.nan), axis=1)
    any_player = ~np.all(np.isnan(dist), axis=1)
    nearest = np.full(track["n_frames"], NOBODY, dtype=np.int8)
    slot = np.nanargmin(np.where(any_player[:, None], dist, 0.0), axis=1)
    nearest[any_player] = team[slot[any_player]]
    return d_home, d_away, nearest


def ball_speed(track, params=PARAMS):
    """Ball speed (m/s) over the next `ball_speed_lag` frames; NaN where either end is missing
    or the displacement is a tracking jump."""
    ball, lag, n = track["ball"], params["ball_speed_lag"], track["n_frames"]
    speed = np.full(n, np.nan)
    step = np.linalg.norm(ball[lag:] - ball[:-lag], axis=1) / (lag / float(FPS))
    speed[:-lag] = step
    speed[speed > params["max_ball_speed"]] = np.nan
    return speed


def ball_out_of_play_position(track, params=PARAMS):
    """True where the ball is seen beyond the touch or goal lines."""
    x, y, m = track["ball"][:, 0], track["ball"][:, 1], params["out_margin"]
    with np.errstate(invalid="ignore"):
        return (x < -m) | (x > PITCH_LENGTH + m) | (y < -m) | (y > PITCH_WIDTH + m)


def field_speed(track, params=PARAMS, centred=False):
    """Mean speed of all tracked players, averaged over the trailing window (m/s), or over a
    window centred on the frame."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_frame = np.nanmean(track["player_speed"], axis=1)
    valid = ~np.isnan(per_frame)
    w = params["player_speed_window"]
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, per_frame, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    idx = np.arange(len(per_frame)) + 1
    lo = np.maximum(0, idx - w)
    with np.errstate(invalid="ignore", divide="ignore"):
        trailing = (csum[idx] - csum[lo]) / (ccnt[idx] - ccnt[lo])
    if not centred:
        return trailing
    shift = w // 2
    out = np.full(len(trailing), np.nan)
    out[:-shift] = trailing[shift:]
    return out


# --------------------------------------------------------------------------------------------
# Step (a): dead ball before restarts
# --------------------------------------------------------------------------------------------

def load_restarts():
    """Smart Tagging set pieces as (type, start_frame, end_frame). Case-insensitive."""
    import load_smart_tagging
    out = []
    for event in load_smart_tagging.load():
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        kind = (event["tags"].get("Type") or "").strip().lower()
        out.append((kind, event["start_frame"], event["end_frame"]))
    return out


def find_restart_frame(speed, start, end, params=PARAMS):
    """First strike of the ball inside a tag window, or None."""
    fast = np.nan_to_num(speed, nan=0.0) > params["restart_ball_speed"]
    quiet = params["restart_quiet_before"]
    for t in np.flatnonzero(fast[start:end]) + start:
        if not fast[max(0, t - quiet):t].any():
            return int(t)
    return None


def restart_frame(speed, start, end, params=PARAMS):
    """(restart frame, strike seen?) for one tag: the seen strike, else the fallback lead."""
    r = find_restart_frame(speed, start, end, params)
    if r is not None:
        return r, True
    return min(end - 1, start + params["restart_fallback_lead"]), False


def stoppage_signals(track, params=PARAMS):
    """Per-frame (evidence, struck).

    evidence -- something that only happens with the ball dead: the field at walking pace, the
                ball beyond the lines, or the ball stationary (free kicks, corners, goal kicks
                and kick-offs all require a stationary ball).
    struck   -- the opposite: the ball seen inside the pitch and moving like a pass or a shot.
    """
    speed = ball_speed(track, params)
    out = ball_out_of_play_position(track, params)
    visible = ~np.isnan(track["ball"][:, 0])
    slow = np.nan_to_num(field_speed(track, params), nan=0.0) < params["stop_player_speed"]
    with np.errstate(invalid="ignore"):
        stationary = speed < params["stationary_ball_speed"]
        struck = visible & ~out & (speed > params["restart_ball_speed"])
    return slow | out | stationary, struck, speed


def restart_dead_mask(track, params=PARAMS, kickoff_2nd=None, tag_floor=False):
    """D before every tagged restart, from when the ball went dead up to the strike.

    Every tag is trusted to mark a real restart, so it gets at least min_dead of D. From there
    the stoppage is extended backwards while there is evidence of a dead ball (see
    stoppage_signals), tolerating short evidence gaps, and stopping at the last live strike of
    the ball inside the pitch or at the cap for that restart type.

    kickoff_2nd -- (first graded frame, first live frame) of the second half from games.py
    (step d). When given, the published kick-off replaces the tracking's estimate for the
    second-half kick-off tag.

    tag_floor -- step (e): the stoppage covers at least the tag window up to the strike. Smart
    Tagging opens a set-piece window a few seconds before the restart (input data: strike a
    median 6 s into the window), when the ball is being placed and the taker set (Laws 8, 13,
    15-17). Only live play could precede a quick restart in that stretch, and live play shows
    as the ball struck inside the pitch, so the floor never reaches back past such a strike.
    """
    n = track["n_frames"]
    evidence, struck, speed = stoppage_signals(track, params)
    blocks = half_video_blocks(GAME)
    mask = np.zeros(n, dtype=bool)
    stats = {"tags": 0, "used": 0, "fallback": 0, "unknown_type": 0}
    lengths = {}

    for kind, start, end in load_restarts():
        stats["tags"] += 1
        if kind not in params["restart_cap_s"]:
            stats["unknown_type"] += 1
            continue
        r, seen_strike = restart_frame(speed, start, end, params)
        stats["fallback"] += int(not seen_strike)

        cap = params["restart_cap_s"][kind]
        if cap is None:  # match kick-off: dead from the start of the half
            half_start = next((first for _, first, last in blocks if r <= last), None)
            if kickoff_2nd is not None and half_start == kickoff_2nd[0]:
                r = kickoff_2nd[1]
            if half_start is not None and half_start < r:
                mask[half_start:r] = True
            stats["used"] += 1
            continue

        earliest = max(0, r - int(cap * FPS))
        dead_start, misses, t = r, 0, r - 1
        while t >= earliest and not struck[t]:
            if evidence[t]:
                dead_start, misses = t, 0
            else:
                misses += 1
                if misses > params["evidence_tolerance"]:
                    break
            t -= 1
        dead_start = max(earliest, min(dead_start, r - params["min_dead"]))
        if tag_floor and start < r:
            live = np.flatnonzero(struck[start:r])
            floor = start + int(live[-1]) + 1 if len(live) else start
            if floor < dead_start:
                stats["floor_frames"] = stats.get("floor_frames", 0) + dead_start - max(
                    earliest, floor)
            dead_start = max(earliest, min(dead_start, floor))
        mask[dead_start:r] = True
        stats["used"] += 1
        lengths.setdefault(kind, []).append((r - dead_start) / float(FPS))
    stats["median_s"] = {k: round(float(np.median(v)), 1) for k, v in sorted(lengths.items())}
    return mask, stats


# --------------------------------------------------------------------------------------------
# Step (d): the second-half kick-off from games.py
# --------------------------------------------------------------------------------------------

def second_half_kickoff():
    """(first graded frame, first live frame) of the second half, both from games.py.

    FAQ.md: the second half is graded from its first tracked frame but kicks off ~2 minutes
    later, and every graded frame before the kick-off is D. games.py publishes the kick-off as
    an anchor -- "the ball is struck on this frame" (kickoff_2nd) -- and the graded span as
    half_video_blocks(). The strike frame is the last dead frame; play is live from the next.
    The first half needs nothing: it is graded from its kick-off (FAQ.md).
    """
    first = next(first for half, first, _ in half_video_blocks(GAME) if half == 2)
    live_from = anchor_video_frame(GAMES[GAME]["kickoff_2nd"]) + 1
    return first, live_from


# --------------------------------------------------------------------------------------------
# Step (e): more dead-ball evidence (the set-piece window floor is in restart_dead_mask)
# --------------------------------------------------------------------------------------------

def out_of_play_dead_mask(track, params=PARAMS):
    """D from the ball being seen beyond the lines until play restarts.

    Law 9: once the whole ball has crossed a touch or goal line it is out of play, and it stays
    out until a throw-in, goal kick or corner (Laws 15-17) -- seen here as the next strike of
    the ball inside the pitch. Frames in between with the ball missing stay dead.
    """
    n = track["n_frames"]
    _, struck, _ = stoppage_signals(track, params)
    out = ball_out_of_play_position(track, params)
    tempo = np.nan_to_num(field_speed(track, params, centred=True), nan=0.0)
    cap = int(FPS * max(params["restart_cap_s"][k] for k in OUT_OF_PLAY_RESTARTS))
    mask = np.zeros(n, dtype=bool)
    stats = {"runs": 0, "brief": 0, "live_tempo": 0, "used": 0, "no_restart": 0}
    for lo, hi in runs_of(out):
        stats["runs"] += 1
        if hi - lo < params["out_min_frames"]:
            stats["brief"] += 1
            continue
        if np.median(tempo[lo:hi]) >= params["live_tempo_speed"]:
            stats["live_tempo"] += 1
            continue
        limit = min(n, max(hi, lo + cap))
        restart = np.flatnonzero(struck[hi:limit])
        end = hi + int(restart[0]) if len(restart) else limit
        stats["no_restart"] += int(len(restart) == 0)
        mask[lo:end] = True
        stats["used"] += 1
    return mask, stats


def load_goals():
    """Smart Tagging GOALS intervals as (start_frame, end_frame)."""
    import load_smart_tagging
    return [(event["start_frame"], event["end_frame"]) for event in load_smart_tagging.load()
            if (event["code"] or "").strip().lower() == "goals"]


def goal_dead_mask(track, params=PARAMS):
    """After a goal the ball is dead until the kick-off (Laws 8 and 10).

    A GOALS tag spans the attack that produced the goal and the celebration, and ends at the
    kick-off (input data), so its start is not the goal. The goal is the first sighting of the
    ball over a goal line between the posts inside the tag, after the strike of any other
    restart tagged inside it (a corner that led to the goal, say). From there to the kick-off
    strike is D. If the ball is never seen in the goal, the tag adds nothing and step (a) still
    covers the stoppage before the kick-off.
    """
    n = track["n_frames"]
    speed = ball_speed(track, params)
    x, y, m = track["ball"][:, 0], track["ball"][:, 1], params["out_margin"]
    with np.errstate(invalid="ignore"):
        in_goal = (((x < -m) | (x > PITCH_LENGTH + m))
                   & (np.abs(y - PITCH_WIDTH / 2.0) <= params["goal_width"] / 2.0 + m))
    restarts = [(kind, restart_frame(speed, s, e, params)[0]) for kind, s, e in load_restarts()]
    reach = int(FPS * params["restart_cap_s"]["kick off (after goal)"])
    mask = np.zeros(n, dtype=bool)
    stats = {"goals": 0, "seen": 0}
    for start, end in load_goals():
        stats["goals"] += 1
        kickoffs = [r for kind, r in restarts
                    if kind == "kick off (after goal)" and start <= r <= end + reach]
        if not kickoffs:
            continue
        kickoff = min(kickoffs)
        others = [r + 1 for kind, r in restarts
                  if start <= r < kickoff and not kind.startswith("kick off")]
        search_from = max([start] + others)
        hits = np.flatnonzero(in_goal[search_from:kickoff])
        if len(hits) == 0:
            continue
        mask[search_from + int(hits[0]):kickoff] = True
        stats["seen"] += 1
    return mask, stats


# --------------------------------------------------------------------------------------------
# Step (b): hysteresis possession switching
# --------------------------------------------------------------------------------------------

def control_evidence(d_home, d_away, params=PARAMS):
    """Per frame: HOME / AWAY when that team is clearly on the ball, NOBODY otherwise."""
    radius, margin = params["control_radius"], params["control_margin"]
    far = np.inf
    h = np.nan_to_num(d_home, nan=far)
    a = np.nan_to_num(d_away, nan=far)
    ev = np.full(len(h), NOBODY, dtype=np.int8)
    ev[(h < radius) & (h + margin < a)] = HOME
    ev[(a < radius) & (a + margin < h)] = AWAY
    return ev


def runs_of(mask):
    """[(start, end)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def segment_possession(ev, nearest, params=PARAMS):
    """Team codes for one alive segment (a stretch of live play between stoppages).

    Starts fresh: after a restart there is a clear possessor (CHALLENGE.md), so the first team
    to be confirmed on the ball owns the segment from its start. Then the hysteresis rule above.
    Returns None if nobody is ever confirmed on the ball.
    """
    need = params["confirm_frames"]
    state, streak_team, streak_start, count = NOBODY, NOBODY, 0, 0
    switches = []                                   # (frame offset, team)
    for i in np.flatnonzero(ev != NOBODY):
        team = ev[i]
        if team == state:
            streak_team, count = NOBODY, 0          # the holder touched it: challenge over
            continue
        if team != streak_team:
            streak_team, streak_start, count = team, i, 0
        count += 1
        if count >= need:
            state = team
            switches.append((streak_start, team))
            streak_team, count = NOBODY, 0
    if not switches:
        return None
    out = np.empty(len(ev), dtype=np.int8)
    bounds = [0] + [s for s, _ in switches[1:]] + [len(ev)]
    for (lo, hi), (_, team) in zip(zip(bounds[:-1], bounds[1:]), switches):
        out[lo:hi] = team
    return out


def hysteresis_labels(ev, nearest, alive):
    """H/A on alive frames from the hysteresis machine, D elsewhere.

    A segment where nobody is ever confirmed on the ball falls back to the majority of the
    nearest-player answers in it, or X (no answer) if the ball is never seen at all.
    """
    labels = np.full(len(ev), "D", dtype="<U1")
    stats = {"segments": 0, "fallback": 0, "switches": 0}
    for lo, hi in runs_of(alive):
        stats["segments"] += 1
        teams = segment_possession(ev[lo:hi], nearest[lo:hi])
        if teams is None:
            stats["fallback"] += 1
            seg = nearest[lo:hi]
            home, away = int(np.sum(seg == HOME)), int(np.sum(seg == AWAY))
            labels[lo:hi] = "X" if home + away == 0 else ("H" if home >= away else "A")
            continue
        stats["switches"] += int(np.sum(teams[1:] != teams[:-1]))
        labels[lo:hi] = np.where(teams == HOME, "H", "A")
    return labels, stats


# --------------------------------------------------------------------------------------------
# Step (c): missing-ball handling
# --------------------------------------------------------------------------------------------

def gap_dead_mask(track, params=PARAMS):
    """D inside long ball-missing gaps, except stretches at live-play tempo; carry the rest.

    Also: if the ball was seen beyond the lines right before it vanished, the stoppage starts
    at the first of those frames (Law 9: the ball is out once it has crossed the line).
    """
    n = track["n_frames"]
    visible = ~np.isnan(track["ball"][:, 0])
    out = ball_out_of_play_position(track, params)
    tempo = np.nan_to_num(field_speed(track, params, centred=True), nan=0.0)
    mask = np.zeros(n, dtype=bool)
    stats = {"short": 0, "long": 0, "live_s": 0.0, "dead_s": 0.0}
    for lo, hi in runs_of(~visible):
        if hi - lo <= params["carry_max_gap"]:
            stats["short"] += 1
            continue
        stats["long"] += 1
        dead = np.ones(hi - lo, dtype=bool)
        for a, b in runs_of(tempo[lo:hi] >= params["live_tempo_speed"]):
            if b - a >= params["min_live_in_gap"]:
                dead[a:b] = False
        mask[lo:hi] = dead
        if dead[0]:
            t = lo
            while t - 1 >= 0 and out[t - 1]:
                t -= 1
            mask[t:lo] = True
        stats["live_s"] += float(np.sum(~dead)) / FPS
        stats["dead_s"] += float(np.sum(dead)) / FPS
    return mask, stats


def seen_within(nearest, window):
    """True where the ball is seen within +/- window frames: the baseline's implicit alive rule
    (its majority vote returns D exactly where this is False)."""
    seen = np.concatenate(([0], np.cumsum(nearest != NOBODY)))
    idx = np.arange(len(nearest))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(nearest), idx + window + 1)
    return (seen[hi] - seen[lo]) > 0


# --------------------------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------------------------

def graded_mask(n_frames):
    """True inside the two halves. Published boundaries (DATA.md), not labels."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def predict(track, params=PARAMS, steps=DEFAULT_STEPS):
    """Per-frame labels H/A/D/X for the whole video. `steps` switches improvements on, so each
    one can be scored on its own (ablation for the technical document):

        a  dead ball before Smart Tagging restarts     b  hysteresis possession switching
        c  missing-ball handling                       d  second-half kick-off from games.py
        e  more dead-ball evidence: ball out of play, set-piece windows, goals
    """
    n = track["n_frames"]
    graded = graded_mask(n)
    d_home, d_away, nearest = ball_distances(track)
    kickoff_2nd = second_half_kickoff() if "d" in steps else None

    # Dead ball from the missing ball: step (c), or the baseline's rule (no ball within +/- 2 s).
    if "c" in steps:
        dead, stats = gap_dead_mask(track, params)
        print("(c) ball gaps: {short} short (carried), {long} long; inside long gaps "
              "{dead_s:.0f} s dead, {live_s:.0f} s live at match tempo".format(**stats))
    else:
        dead = ~seen_within(nearest, params["baseline_window"])
    if "a" in steps:
        restart_dead, stats = restart_dead_mask(track, params, kickoff_2nd, "e" in steps)
        print("(a) restarts: {tags} tags, {used} used, {fallback} without a seen strike".format(
            **stats))
        if "e" in steps:
            print("(e) set-piece windows add {} D frames".format(stats.get("floor_frames", 0)))
        print("    median stoppage length (s) by type: {}".format(stats["median_s"]))
        print("    D frames from restarts: {}".format(int(restart_dead.sum())))
        dead |= restart_dead
    if kickoff_2nd is not None:
        first, live_from = kickoff_2nd
        dead[first:live_from] = True
        print("(d) second half: D from its first graded frame to the published kick-off "
              "({} frames)".format(live_from - first))
    if "e" in steps:
        out_dead, stats = out_of_play_dead_mask(track, params)
        print("(e) ball beyond the lines: {runs} runs, {brief} under 0.5 s, {live_tempo} at "
              "live tempo, {used} used ({no_restart} without a restart seen)".format(**stats))
        goal_dead, stats = goal_dead_mask(track, params)
        print("    goals: {goals} tags, ball seen in the goal for {seen}".format(**stats))
        new = (out_dead | goal_dead) & ~dead & graded
        print("    D frames added by out of play / goals: {}".format(int(new.sum())))
        dead |= out_dead | goal_dead

    # Who has the ball while it is alive.
    if "b" in steps:
        ev = control_evidence(d_home, d_away, params)
        print("(b) frames with a team clearly on the ball: {:.1%} of graded".format(
            float(np.mean(ev[graded] != NOBODY))))
        labels, stats = hysteresis_labels(ev, nearest, graded & ~dead)
        print("    {segments} live segments, {switches} possession changes, "
              "{fallback} segments without a confirmed possessor".format(**stats))
    else:
        labels = baseline.smooth(nearest, params["baseline_window"])
        labels = np.where(dead, "D", labels)

    return np.where(graded, labels, "X")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=TEAM_NAME, help="output file stem")
    parser.add_argument("--out", default=RESULTS_DIR, help="output folder")
    parser.add_argument("--steps", default=DEFAULT_STEPS,
                        help="improvements to apply, e.g. 'a' (default: {}); '' = baseline"
                        .format(DEFAULT_STEPS))
    args = parser.parse_args()

    track = load_tracking()
    seen = int(np.sum(~np.isnan(track["ball"][:, 0])))
    print("ball located on {} of {} frames ({:.1f}%)".format(
        seen, track["n_frames"], 100.0 * seen / track["n_frames"]))

    labels = predict(track, steps=args.steps)
    path = baseline.write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    counts = "  ".join("{} {}".format(k, int(np.sum(labels == k))) for k in "HADX")
    print("wrote {}  ({})".format(os.path.relpath(path, ROOT), counts))


if __name__ == "__main__":
    main()
