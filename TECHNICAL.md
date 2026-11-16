# UvAStudents: ball possession from automatic tracking

**More than a Hack 2027, Phase 1. Technical document.**
Code: <https://github.com/seanzlli/mtah2027-uvastudents> · one command: `python my_model.py`

## Summary

We label every frame of the match as `H`, `A` or `D` with a transparent, rule-based model built
on the automatic tracking and the Smart Tagging set pieces. It has no learned components and no
training data. It is the starter baseline (nearest player to the ball), changed in three ways:

1. **(a) Dead ball before restarts.** Smart Tagging set pieces, located exactly with the
   tracking.
2. **(b) Hysteresis possession switching.** Possession changes only after sustained, clear
   control by the other team, and the boundary is dated back to the first touch.
3. **(c) Missing-ball handling.** Long ball gaps are treated as stoppages unless the players move
   at live-play tempo.

Every parameter is a football quantity with a stated reason (§4.4). None was tuned on the
public ground truth. Over the full match the model reports 254 possessions of 14.4 s mean,
against the baseline's 594 flickering spells of 7.2 s.

## 1. Approach, and why

**The metric drives the design.** spellF1 scores whole possessions matched by temporal IoU. Two
things destroy it:

- **Flicker:** one possession split into many.
- **Late boundaries:** perfect but 1 s late scores 0.65.

The baseline's majority vote trades one against the other through a single window size. We
separate them instead:

- **Deciding** that possession changed needs evidence over time (1 s of control).
- **Placing** the change uses the first frame of that evidence.

**Dead ball matters for possessions, not just for the D class.** A missed stoppage between two
spells of the same team merges two real possessions into one, and both are lost. In top-flight
football the ball is out of play for roughly 35–45% of the time; our model puts 41% of graded
frames dead.

**Why rules and not a learned model.** No labels on this match may be used. The public five
minutes (17 possessions) are a pipeline check, too small even to choose between two versions
(§6). Within the time available we had no labelled data from other games. A model whose every
threshold is a physical quantity (metres, m/s, seconds) can be checked by reading it, and it
should transfer to another match unchanged.

**What the input data told us** (aggregates of tracking and tagging only, never the ground
truth; `analysis/data_stats.py`):

- **Ball gaps.** The ball is missing on 40.7% of graded frames, but almost never briefly. Of the
  150 gaps in graded play, 17 last ≤ 1 s and 133 last > 4 s. There are none between 1 and 4 s.
- **Long gaps look like stoppages.**

  | | ball visible | inside long gaps |
  |---|---|---|
  | field-average player speed, median | 2.01 m/s | 1.04 m/s |
  | share of frames at walking pace | 13.6% | 62.2% |

  71.5% of the restart stoppages found by (a) lie inside long gaps. In 80% of long gaps the ball
  is struck (> 4 m/s) within 1 s of reappearing.
- **Set-piece tags.** They are 35 s windows (5 s for kick-offs) that open about 5 s before the
  strike. A tag marks the restart and the play after it, not the stoppage before it.

## 2. Assets: used and dropped

| asset | use |
|---|---|
| Tactical ATD (players, ball, per-slot speed, slot → team from the metadata) | **Used.** The only source of geometry. Positions are converted to metres on the 105.3 × 68.0 m pitch from `games.py`. |
| Tactical Smart Tagging, `SET PIECES` (59 tags) | **Used** for restarts. Each tag's type sets the maximum stoppage length (§4.1). |
| Smart Tagging phase codes (Build Up, Progression, …) and `Team` tags | **Dropped.** Phases are not possessions (DATA.md), every throw-in is `Team = N/A`, and there was no time to validate them as a soft prior. Listed under future work. |
| `games.py` half boundaries | **Used** to write `X` outside the two halves. These are published boundaries, not labels. |
| Tactical homography and video | **Not used by the algorithm.** We work in pitch coordinates. They are useful only for looking at errors. |
| Broadcast video, ATD, homography | **Dropped.** They are not synchronised to the tactical timeline, no mapping is provided, and 34% of frames have no homography. Aligning them was not worth it in the time available. |
| Public 5-minute ground truth | **Pipeline check only:** reproducing the published baseline scores, plus one sanity score per idea. Never training or tuning. |

**Tried and dropped.**

- **Vetoing tags without a visible stoppage.** The first version of (a) dropped any tag where the
  players were not at walking pace and the ball was not seen out. That removed 22 of 59 tags:
  8 of 10 goal kicks and 5 of 12 corners. At those restarts players run back into shape or
  jostle in the box, so this was wrong football, not bad tags. The final version trusts every
  tag and adds "ball stationary" as evidence.
- **A float32 tracking cache.** Replaced by exact float64 parsing on every run (§8).
- **Carrying possession through every ball gap.** Considered and rejected after the gap
  statistics above.

## 3. Handling missing and noisy tracking

- **Ball missing (41% of graded frames).** Gaps of 3 s or less are carried: a pass in flight or
  a brief occlusion. Longer gaps are D by default. The exception is any stretch of ≥ 2 s where
  the field-average speed (centred 2 s mean) is at live-play tempo, ≥ 2 m/s; there possession is
  carried. If the ball was seen beyond the lines just before vanishing, D starts at the first of
  those frames. Over the match, 3,183 s of long gaps are D and 242 s are carried as live play.
- **Ball jitter.** Ball speed is measured over 0.2 s rather than frame to frame. Displacements
  faster than 40 m/s (above the hardest shot) are treated as tracking jumps and ignored. Every
  margin (1 m beyond the lines, 1 m for "clearly closer") is above the jitter level.
- **No player identities, fragmented tracks, identity switches, duplicates.** The model never
  follows a player. Per frame it uses only the nearest home and nearest away distance to the
  ball, and the field-average speed over whichever slots are tracked. A switched or duplicated
  track changes neither.
- **Players missing.** Typically 15–21 players are tracked. The field average uses those
  present, and frames with no players give no control evidence.
- **Malformed lines.** They are left as missing and counted (0 in the delivered file). Error
  messages give a line number, never content.

## 4. Possession rules implemented

### 4.1 Ball alive or dead

The ball is **dead (`D`)** in four cases:

1. **Before every tagged restart.**
   - **Strike.** The restart is the first frame in the tag window where the ball moves faster
     than 4 m/s after at least 1 s without such motion. If the strike is never seen (2 of 59
     tags), we use 3.5 s into the tag.
   - **Stoppage.** At least 2 s of D before the strike. We then walk backwards while there is
     dead-ball evidence: field at walking pace (< 1.2 m/s, trailing 2 s), ball beyond the lines,
     or ball stationary (< 1 m/s). Gaps of up to 1 s without evidence are tolerated.
   - **Stop.** The walk ends at the last live strike of the ball inside the pitch, or at a cap
     per restart type: throw-in 20 s, goal kick 30 s, corner 40 s, free kick 60 s, kick-off
     after a goal 120 s.

   Resulting median stoppages: corner 15 s, free kick 23 s, kick-off after goal 40 s,
   throw-in 5 s, goal kick 2.8 s.
2. **Before each half's kick-off**, from the first graded frame to the strike. This is found from
   the `Kick Off (Start)` tag and the tracking, with no frame constant. It covers the ~2 minutes
   of graded dead ball before the second-half kick-off.
3. **Inside long ball gaps** without live-play tempo (§3).
4. **From the moment the ball is seen beyond the lines** when it then vanishes (Law 9).

We assign no owner to a dead ball. The task asks for `D` there.

### 4.2 Who has the ball while it is alive

- **Control.** A team is *on the ball* in a frame when its nearest player is within 2 m of the
  ball **and** at least 1 m closer than the nearest opponent.
- **Loose balls, passes in flight, missing ball.** Neither team is on the ball, so possession
  stays with the last team. This is CHALLENGE.md's rule for contested or loose phases.
- **Challenges and 50/50 duels.** If the nearest home and nearest away players are within 1 m
  of each other in distance to the ball, the frame gives no evidence. So a duel alone never
  changes possession, and a tackle counts only when the winner's team then shows clear control.
- **Touches, deflections, lost aerial duels.** The other team must collect 1 s (25 frames) of
  clear control, and **any** clear touch by the current holder resets that count. A deflection
  or a single touch (< 0.3 s) cannot switch possession.
- **Interceptions.** Frames with no evidence do not break the count, so "interception, then a
  team-mate's touch" adds up to 1 s across both players. This is our version of the challenge's
  "a team completes consecutive passes" proxy for control.
- **Boundary placement.** Once a switch is confirmed, it is dated back to the first frame of the
  winning run, which is the moment control was gained.

### 4.3 Restarts

After every D stretch the state machine starts fresh. CHALLENGE.md says that after a restart
there is a clear possessor, so the first team confirmed on the ball owns the new live segment
from its first frame. This also handles throw-ins, whose tag carries no team: the tracking
decides.

A live segment where no team is ever confirmed takes the majority of the nearest-player answers.
There are 35 such segments out of 148: median 3.4 s long, 24 of them shorter than 5 s, 166 s in
total. In 14 of them the ball is never seen at all, and those get `X`. Outside the two halves we
always write `X`. Overall, `X` is on 0.89% of graded frames.

### 4.4 Parameters

| parameter | value | football reason |
|---|---|---|
| control radius | 2 m | playing distance: a dribbler keeps the ball within about a stride (1.5–2 m) |
| control margin | 1 m | "clearly closer", not a 50/50; above tracking jitter |
| confirm switch | 1 s | touches and deflections last < 0.3 s; a controlled reception plus the next action reaches 1 s |
| strike speed | 4 m/s | any restart pass or throw is ≥ 5 m/s; a ball carried or placed moves < 2 m/s |
| quiet before strike | 1 s | a ball still rolling out over the line is not the restart |
| stationary ball | 1 m/s | true speed ~0, plus the jitter of 0.2 s of noisy positions |
| walking pace | 1.2 m/s | walking is 1–1.4 m/s; in a stoppage most players walk or stand |
| live-play tempo | 2 m/s | typical field average with the ball in play (~2 m/s); well above walking, because long gaps are stoppages by prior |
| minimum stoppage / live stretch | 2 s | Laws 8, 13, 15–17: ball stationary or taker set before every restart |
| evidence tolerance | 1 s | a player jogging over to take a throw does not end the stoppage |
| carry short gap | 3 s | a 50–60 m diagonal is in the air for about 3 s |
| out margin | 1 m | Law 9: the whole ball must cross the line; slack for tracking error |
| caps per restart | 20 / 30 / 40 / 60 / 120 s | typical top-flight times from ball dead to restart (§4.1) |
| ball-speed lag, jump limit | 0.2 s, 40 m/s | jitter suppression; above the hardest shot |
| strike fallback | 3.5 s into tag | the starter notebook shows kick-off tags opening ~3.5 s early |

## 5. Training data

**None.** No labelled dataset from any game was used, and nothing was learned. In particular,
no hand labels or human-produced data about this match were used for anything. Parameters come
from football knowledge and from aggregate statistics of the *input* data (tracking and
tagging), never from the ground truth. The public five minutes were used only to confirm the
pipeline reproduces the published baseline scores, and for one sanity score per idea.

## 6. Results (public five minutes: a sanity check, not model selection)

| model | macroF1 | spellF1 | possessions, public (truth 17) | possessions, full match |
|---|---|---|---|---|
| baseline (`--steps ""`, = `baseline_smooth50`) | 0.6207 | 0.3256 | 26 | 594 (7.2 s) |
| + (a) restarts | 0.6519 | 0.3545 | 27 | 568 (6.9 s) |
| + (b) hysteresis | 0.6818 | 0.4824 | 17 | 259 (15.2 s) |
| **+ (c) missing ball: submission** | **0.7277** | **0.4941** | 17 | 254 (14.4 s) |

A 1000-draw bootstrap gives a widest spellF1 interval of **~0.33** on this window. None of these
gaps is significant here. The case for each step rests on its football logic and on the
full-match possession structure becoming realistic. Each idea was scored once after it was
written. (a) was scored twice, and the change between versions was decided from the input-data
counts in §2, not from the score.

## 7. Limitations, and what we would do with more time

- **Short goal-kick and throw-in stoppages** (median 2.8 s and 5 s, against 15–30 s and
  10–20 s in real football). The backward search probably stops at the ball being played back
  to the taker, or the strike detector fires on the ball going out. A per-type "ball comes to
  rest" anchor would fix this.
- **Untagged stoppages with the ball visible are missed:** free kicks, fouls, injuries,
  substitutions. The tagging lists only 5 free kicks and 26 throw-ins. A generic detector would
  look for the field at walking pace and a stationary ball for ≥ 2 s.
- **Control evidence exists on only 27% of graded frames.** Long live stretches without a
  visible ball rely on carrying. Team shape could fill in: which side is pushing up, and how
  compact the defending block is.
- **No height information.** Aerial balls project onto the pitch and can create false control
  near a player. A header duel is treated as ground control.
- **Smart Tagging team phases unused.** They could act as a soft prior for the 35 live segments
  with no confirmed possessor, and as evidence in the hysteresis.
- **Timing bias not investigated.** On the public window, changes land about 0.5 s early on
  average. Correcting it on 17 possessions would be tuning.
- **No validation beyond five minutes.** With more time we would label or source possession
  data on *other* matches (for example Metrica's open sample games), check every threshold's
  sensitivity there, and replace the hand-written hysteresis with a hidden Markov model whose
  transition costs are fitted on those games.

## 8. Reproduction

Tested on macOS arm64, Python 3.13.13. A run takes ~3 s and ~0.7 GB of memory.

```bash
git clone https://github.com/seanzlli/mtah2027-uvastudents.git
cd mtah2027-uvastudents
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # numpy 2.5.4, pandas 3.1.0 + pinned dependencies
```

Put the match bundle in `./data`, or set `MTAH_DATA_DIR`. Only three files are read:
`data/tactical/BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_smart_tagging.xml`.

```bash
python my_model.py                  # writes evaluation/results/UvAStudents.csv
shasum -a 256 evaluation/results/UvAStudents.csv      # Linux: sha256sum
python evaluation/validate_submission.py evaluation/results/UvAStudents.csv
python selftest.py                  # 14 checks; exit code 0 = no failures
```

The expected SHA-256 is `56ded60ba2daef692949575dc7dfa4be4aa8f5b9ed2302eee7f02c4a3e8cd0fa`.

The run is deterministic:

- It parses the raw files on every run, with no cache and no randomness.
- All arithmetic is float64 on the values exactly as written in the files.
- Dependencies are pinned, transitive ones included.
- Two runs from scratch give the same hash.

`python my_model.py --steps "" --name base` writes the baseline row of §6; `--steps a` and
`--steps ab` write the next two. `selftest.py` checks reproduction, validity, invariants and
repository hygiene.

## 9. Data handling and development process

The code was written with the help of an AI coding assistant (Claude Code). In line with the
terms of use, no match data, extract of it, or submission file was shared with it:

- Scripts read `data/` locally and printed only summary numbers: counts, percentiles and
  scorer metrics.
- The assistant never opened the ground-truth file or the scorer's per-transition reports.

The repository contains no match data. `data/`, the submission CSV and any derived tracking
are gitignored, and `selftest.py` checks this.
