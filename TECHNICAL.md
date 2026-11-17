# UvAStudents: ball possession from automatic tracking

**More than a Hack 2027, Phase 1. Technical document.**
Code: <https://github.com/seanzlli/mtah2027-uvastudents> · one command: `python my_model.py`

## Summary

We label every frame of the match as `H`, `A` or `D` with a largely rule-based model built on
the automatic tracking and the Smart Tagging set pieces. Seven of its eight steps are
hand-written football rules; the one exception, (g), is a single 31-node decision tree fitted on
**other** matches' public data. **No labels from this match are used anywhere** (§5). It is the starter baseline
(nearest player to the ball), changed in eight ways:

1. **(a) Dead ball before restarts.** Smart Tagging set pieces, located exactly with the
   tracking.
2. **(b) Hysteresis possession switching.** Possession changes only after sustained, clear
   control by the other team, and the boundary is dated back to the first touch.
3. **(c) Missing-ball handling.** Long ball gaps are treated as stoppages unless the players move
   at live-play tempo.
4. **(d) Second-half kick-off.** Dead ball up to the kick-off published in `games.py`.
5. **(e) More dead-ball evidence.** The ball is dead from leaving the pitch until play
   restarts. The stoppage covers each set-piece window up to the strike. After a goal it is
   dead until the kick-off.
6. **(f) No abstentions.** Live segments in which the ball is never seen get the previous label
   in the half. This is the scorer's own spellF1 fill, so no graded frame is `X`.
7. **(g) Dead ball from player positions.** A 31-node decision tree over ball-free player speed
   and shape features, fitted on DFL open matches, catching untagged stoppages the rules miss
   (fouls, injuries, substitutions).
8. **(h) Dead-ball evidence with the ball unseen.** The backward search in (a) was ending early
   because its evidence needs a visible ball. Fixing that brings corner, free-kick and throw-in
   stoppages into their real-football ranges. Output-neutral on this match — (g) already covered
   the frames — so the submission is byte-identical with or without it (§6).

Every parameter in steps (a)–(f) and (h) is a football quantity with a stated reason (§4.4);
step (g) adds two more, set on held-out external matches. None was tuned on the public ground truth.
Over the full match the model reports 237 possessions of 12.2 s mean, against the baseline's
594 flickering spells of 7.2 s.

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
football the ball is out of play for roughly 35–45% of the time. Our model puts **54.7%** of
graded frames dead, which is *above* that range and is the submission's clearest open question
(§7).

**Why mostly rules, and where we made an exception.** No labels on this match may be used, and
the public five minutes (17 possessions) are a pipeline check, too small even to choose between
two versions (§6). A model whose every threshold is a physical quantity (metres, m/s, seconds)
can be checked by reading it, and should transfer to another match unchanged. We held to that
for steps (a)–(f).

We made one exception, step (g), because the rules' dead-ball **recall** was measurably poor —
37% on the public window — and every hand-written generic stoppage detector we tried either did
nothing or fixed recall by fragmenting possessions. The exception is kept small and auditable:
one depth-4 tree, 31 nodes, fitted on other matches, committed as plain thresholds and executed
in numpy. The submission needs no machine-learning library to run.

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
- **Set-piece tags.** They are 35 s windows (5 s for kick-offs). The strike comes a median
  6 s into the window (p10 2.8 s, p90 10.8 s). A tag marks the restart and the play after it,
  not the stoppage before it.
- **Goal tags.** The 4 `GOALS` tags last 94–233 s and end at the kick-off. They span the
  scoring attack and the celebration, so a tag's start is not the goal.

## 2. Assets: used and dropped

| asset | use |
|---|---|
| Tactical ATD (players, ball, per-slot speed, slot → team from the metadata) | **Used.** The only source of geometry. Positions are converted to metres on the 105.3 × 68.0 m pitch from `games.py`. |
| Tactical Smart Tagging, `SET PIECES` (59 tags) | **Used** for restarts. Each tag's type sets the maximum stoppage length, and its window sets a minimum stoppage (§4.1). |
| Tactical Smart Tagging, `GOALS` (4 tags) | **Used** to bound the dead ball after a goal (§4.1). On this match the rule adds nothing, because the ball is never seen in the goal mouth. |
| Smart Tagging phase codes (Build Up, Progression, …) and `Team` tags | **Dropped.** Phases are not possessions (DATA.md), every throw-in is `Team = N/A`, and there was no time to validate them as a soft prior. Listed under future work. |
| `games.py` half boundaries and second-half kick-off anchor | **Used** to write `X` outside the two halves, and `D` before the second-half kick-off (FAQ.md: those 2,934 graded frames are all `D`). Both are published by the organisers. Neither is a label, and we wrote no frame numbers ourselves. |
| Tactical homography and video | **Not used by the algorithm.** We work in pitch coordinates. They are useful only for looking at errors. |
| Broadcast video, ATD, homography | **Dropped.** They are not synchronised to the tactical timeline, no mapping is provided, and 34% of frames have no homography. Aligning them was not worth it in the time available. |
| Public 5-minute ground truth | **Pipeline check only:** reproducing the published baseline scores, plus one sanity score per idea. Never training or tuning. |
| **DFL/Sportec open Bundesliga data** (external, 4 matches) | **Used** to fit step (g): two matches to train the tree, two held out to set its parameters and verify. They publish a per-frame `ball_status`, which this match does not have. Permitted by `RULES.md` as data about other games (§5). |

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

The ball is **dead (`D`)** in seven cases, and step (h) corrects the evidence the first one uses:

1. **Before every tagged restart.**
   - **Strike.** The restart is the first frame in the tag window where the ball moves faster
     than 4 m/s after at least 1 s without such motion. If the strike is never seen (2 of 59
     tags), we use 3.5 s into the tag.
   - **Stoppage.** At least 2 s of D before the strike, and at least the tag window up to the
     strike. The window opens a few seconds before the restart, while the ball is being placed
     or the taker set (Laws 8, 13, 15–17). The window floor never reaches back past a strike
     of the ball inside the pitch, because that is live play before a quick restart.
   - **Walk back.** From there we walk backwards while there is dead-ball evidence: field at
     walking pace (< 1.2 m/s, trailing 2 s), ball beyond the lines, or ball stationary
     (< 1 m/s). Gaps of up to 1 s without evidence are tolerated.
   - **Stop.** The walk ends at the last live strike of the ball inside the pitch, or at a cap
     per restart type: throw-in 20 s, goal kick 30 s, corner 40 s, free kick 60 s, kick-off
     after a goal 120 s.

   Resulting median stoppages, with (h): corner 32 s, free kick 35 s, kick-off after goal
   116 s, throw-in 10 s, goal kick 6.6 s. Without (h) they were 15 s, 23 s, 40 s, 7.5 s and
   5.2 s — so three of the five move from below their real-football ranges into them
   (corner 20–40 s, free kick 20–60 s, throw-in 10–20 s).
2. **Before each half's kick-off**, from the first graded frame to the strike.
   - **First half.** It is graded from its kick-off (FAQ.md), found from the
     `Kick Off (Start)` tag and the tracking.
   - **Second half.** It is graded about 2 minutes before its kick-off. The kick-off is the
     `kickoff_2nd` anchor in `games.py` ("the ball is struck on this frame"), so the anchor
     frame is the last `D` and play is live from the next frame. FAQ.md: all 2,934 graded
     frames before it are `D`. The tracking never sees this strike, and the tag-based estimate
     would have ended the dead stretch 5 frames late.
3. **From the ball leaving the pitch until play restarts** (Law 9). The ball must be seen
   beyond a touch or goal line (1 m margin) for ≥ 0.5 s. A lofted ball's ground projection or
   a tracking jump overshoots a line only for a few tenths of a second.
   - **Stretch.** D runs to the next strike of the ball inside the pitch, which is the throw-in,
     goal kick or corner (Laws 15–17). Frames with the ball missing in between stay dead.
   - **Cap.** At most 40 s, the longest of those restarts' caps.
   - **Exception.** A run while the whole field moves at live-play tempo (≥ 2 m/s) is not
     trusted: play is going on, so the "ball" outside is a spare ball or a tracking error.
   - **Over the match**, 46 of 79 runs beyond the lines qualify.
4. **After a goal until the kick-off** (Laws 8, 10). The goal is the first sighting of the
   ball over a goal line between the posts (7.32 m plus 1 m margin) inside a `GOALS` tag,
   after any other restart tagged in it. From there to the kick-off strike is D. On this match
   the ball is never seen in the goal mouth (the net occludes it), so this rule adds nothing.
   The post-goal stoppages are already 83–91% D from cases 1 and 5.
5. **Inside long ball gaps** without live-play tempo (§3).
6. **From the moment the ball is seen beyond the lines** when it then vanishes (Law 9).
7. **Wherever the decision tree says so** (step (g)). Cases 1–6 are anchored on a tag, on the
   ball leaving the pitch, or on the ball going missing, so they cannot see a stoppage that has
   none of those marks — a foul, an injury, a substitution, with the ball lying in play
   position. Only 5 free kicks are tagged in the whole match. The tree reads **player
   positions only** (speeds over 1/2/5/10 s, team spread, compactness, extents — 35 features,
   no ball, since the ball is missing on 41% of graded frames), each converted to its
   percentile *within this match* so a provider's pace of play cannot shift the thresholds.
   Runs shorter than 2 s are dropped (Laws 8, 13, 15–17: a real stoppage is longer), which also
   stops it sprinkling brief stoppages through live play. On a held-out DFL match the tree
   scores 0.889 F1 against 0.756 for the field-at-walking-pace rule alone, almost entirely by
   recall. Here it adds 17,662 D frames the rules missed — 8.9% of all 199,319 frames.

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
There are 36 such segments out of 150: median 2.8 s long, 25 of them shorter than 5 s, 155 s in
total.

In 12 of them (1,200 frames) the ball is never seen at all, so there is no answer to take a
majority of. **Step (f)** gives those frames the previous label in the same half. A live
segment always starts right after a dead stretch, so that label is D, the stoppage before it.

This is exactly the fill the scorer applies before computing spellF1
(`possession_metrics.fill_abstentions`: forward-fill within each graded half, back-fill at its
head). So the possession structure it scores cannot change, and per-frame metrics stop
charging an abstention as a miss. `selftest.py` checks that the output equals that function's
result. No graded frame is `X`. Outside the two halves we always write `X`.

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
| ball out, minimum | 0.5 s | lofted-ball projections and tracking jumps overshoot a line for a few tenths of a second; a ball that has gone out stays out |
| ball out, cap | 40 s | the longest restart after the ball leaves the pitch (corner) |
| goal mouth | 7.32 m (+ 1 m) | Law 1 goal width; Law 10: a goal is the ball over the line between the posts |
| second-half kick-off | `games.py` anchor | published by the organisers; FAQ.md: every graded frame before it is `D` |
| ball-speed lag, jump limit | 0.2 s, 40 m/s | jitter suppression; above the hardest shot |
| strike fallback | 3.5 s into tag | the starter notebook shows kick-off tags opening ~3.5 s early |
| dead-ball tree threshold | 0.30 | the only fitted number in the model, chosen on two **held-out DFL matches** by dead/alive F1 — never on this match (§5) |
| dead-ball minimum run | 2 s | same Laws 8/13/15–17 minimum stoppage as above, applied to the tree's output |

## 5. Training data

**One learned component, trained on other matches. Nothing from this match.**

**Source.** Step (g) is fitted on the DFL/Sportec open data: M. Bassek, R. Rein, H. Weber and
D. Memmert, "An integrated dataset of spatiotemporal and event data in elite soccer",
*Scientific Data* (2025), <https://doi.org/10.1038/s41597-025-04505-y>. It contains official
position and event data for seven matches from the first and second Bundesliga. The DFL
releases it under the **Creative Commons Attribution 4.0 (CC BY 4.0)** licence, and the
citation above is our attribution. We do not redistribute the data: `databallpy`
(`get_open_game(provider="dfl", ...)`) downloads it from the published source.

**Matches.** We use four of the seven:

| role | DFL match ID |
|---|---|
| training | `DFL-MAT-J03WMX` |
| training | `DFL-MAT-J03WN1` |
| held out | `DFL-MAT-J03WPY` |
| held out | `DFL-MAT-J03WOH` |

The two held-out matches were never used for fitting. They were used to set the two parameters
in §4.4 and to check the result. They are therefore a validation set, not an untouched test
set.

**Labels.** The only label is the per-frame `ball_status` field (alive / dead). The data
provider publishes it with the position data. **It comes from the provider, not from us:** we
created, edited and corrected no labels. `RULES.md` permits labelled data from other games and
forbids labels from the graded match. None were used from the graded match, whether for fitting,
choosing a threshold or selecting features.

**Per-match normalisation, and when the public window was seen.** The first tree was fitted
and applied on raw feature values. On this match it pushed the dead share to **66%**, which is
implausible: the ball is out of play for roughly 35–45% of a top-flight match (§1). That figure
is why we switched to per-match rank normalisation, in which each feature becomes its
percentile within the match being labelled (`deadball.py`, applied identically in training).

The public window was **also checked at that point**: the unnormalised tree lowered spellF1
there (NOTES.md, step (g)). So that score was in view when the switch was decided. The
normalised tree gives 54.7%, still above the football range, and §7 discusses it.

**Artefact and software.** The tree is committed as `models/deadball_tree.json`.
`training/train_deadball.py` reproduces it, and it is the only part of the repository that
needs scikit-learn and `databallpy`. `training/requirements.txt` pins both. The submission
itself needs only `requirements.txt`.

Everything else is unlearned. Steps (a)–(f) and (h) are hand-written rules. Every parameter is
a football quantity (§4.4), taken from football knowledge and from aggregate statistics of the
*input* data (tracking and tagging), never from the ground truth.

The public five minutes were used for two things: to confirm the pipeline reproduces the
published baseline scores, and for one sanity score per idea. No parameter anywhere in the model
was chosen by its score on that window. The one design decision taken with a public-window score
in view is the normalisation switch above.

## 6. Results (public five minutes: a sanity check, not model selection)

| model | macroF1 | spellF1 | possessions, public (truth 17) | possessions, full match | dead share, full match |
|---|---|---|---|---|---|
| baseline (`--steps ""`, = `baseline_smooth50`) | 0.6207 | 0.3256 | 26 | 594 (7.2 s) | |
| + (a) restarts | 0.6519 | 0.3545 | 27 | 568 (6.9 s) | |
| + (b) hysteresis | 0.6818 | 0.4824 | 17 | 259 (15.2 s) | |
| + (c) missing ball | 0.7277 | 0.4941 | 17 | 254 (14.4 s) | 41.4% |
| + (d) second-half kick-off | 0.7277 | 0.4941 | 17 | 254 (14.4 s) | 41.4% |
| + (e) more dead-ball evidence | 0.7277 | 0.4941 | 17 | 257 (13.9 s) | 43.0% |
| + (f) no graded `X` | 0.7277 | 0.4941 | 17 | 257 (13.9 s) | 43.7% |
| + (g) dead-ball tree | 0.8017 | 0.5161 | 14 | 237 (12.2 s) | 54.7% |
| **+ (h) unseen-ball evidence: submission** | **0.8017** | **0.5161** | 14 | 237 (12.2 s) | 54.7% |

Per-class F1 on the public window: for (c) through (f) alike H 0.8587, A 0.8227, D 0.5016;
with (g) the D class improves to 0.7198 and its recall from 0.370 to 0.793, which also lifts
H and A precision (0.806 to 0.908 and 0.782 to 0.838) because they were absorbing dead frames.
Rows (g) and (h) are identical because (h) leaves the output byte-identical once the tree is
present (§4.1, §7). With step (g), the public-window possession count drops from 17 to 14
(truth 17), which we note as a known trade-off of that step.

A 1000-draw bootstrap gives a widest spellF1 interval of **~0.33** on this window. None of these
gaps is significant here. The case for each step rests on its football logic and on the
full-match possession structure becoming realistic.

Each idea was scored once after it was written, except (a), which was scored twice. The change
between (a)'s versions was decided from the input-data counts in §2, not from the score. A step
was kept only if public spellF1 did not drop and the possession count stayed close to 17.

(d), (e) and (f) change 5, 2,536 and 1,200 graded frames respectively, **none of them inside the
public window**. That window is the first five minutes of the first half. It has no ball-out
run of ≥ 0.5 s, no set-piece window that adds frames and no never-seen live segment. Their
identical public scores therefore say nothing either way. Those steps rest on the rules of the
game, the published kick-off and the more realistic stoppage lengths. Step (f) also cannot
change spellF1 by construction.

## 7. Limitations, and what we would do with more time

- **Goal-kick stoppages are still short** (median 6.6 s against 15–30 s in real football).
  We diagnosed and fixed the two general causes in step (h) — the backward search's evidence
  needed a visible ball, and a strike while the whole field walked was treated as live play —
  which brought corners, free kicks and throw-ins into range. Goal kicks did not follow, so
  something specific to them remains: most likely the strike detector firing on the ball
  crossing the goal line rather than on the kick. A per-type "ball comes to rest" anchor is
  still the fix.
- **Step (h) is output-neutral here.** It makes the rule layer correct — the stoppage lengths it
  reports are now defensible — but on this match step (g) had already marked every frame the
  corrected search reaches, so the submission is byte-identical. It is kept because the rules
  should be right independently of the tree, not because it changes the score.
- **Untagged stoppages with the ball in play position are missed:** free kicks, fouls, injuries,
  substitutions. The tagging lists only 5 free kicks. Step (e) catches untagged throw-ins and
  goal kicks when the ball is seen leaving the pitch, but nothing marks a foul. A generic
  detector would look for the field at walking pace and a stationary ball for ≥ 2 s.
- **The goal rule is untested here.** The ball is never seen in the goal mouth on this match,
  so step (e)'s goal case has no effect. A goal seen only as "ball vanishes near the goal" is
  not used.
- **Control evidence exists on only 27% of graded frames.** Long live stretches without a
  visible ball rely on carrying. Team shape could fill in: which side is pushing up, and how
  compact the defending block is.
- **No height information.** Aerial balls project onto the pitch and can create false control
  near a player. A header duel is treated as ground control.
- **Smart Tagging team phases unused.** They could act as a soft prior for the 35 live segments
  with no confirmed possessor, and as evidence in the hysteresis.
- **Timing bias not investigated.** On the public window, changes land about 0.5 s early on
  average. Correcting it on 17 possessions would be tuning.
- **No validation beyond five minutes.** The public window cannot test steps (d) to (f) at
  all (§6). With more time we would source possession labels on *other* matches and check
  every threshold's sensitivity there.
- **The dead share is 54.7% of graded frames, above the 35–45% a whole match of top-flight
  football gives — but measured against the ground truth we have, step (g) made it better, not
  worse.** On the public five minutes the truth is **19.6%** dead; steps (a)–(f) predicted
  **9.3%**, under by more than half, and (a)–(g) predict **23.6%**, over by four points. So the
  full-match 54.7% should not be read as the model having broken a figure that was previously
  right: on the only frames where the answer is known, dead-ball recall went from 0.370 to
  0.793 and the share moved from badly low to slightly high.

  What remains genuinely open is the gap between that 54.7% and the ~46% true share we measure
  on external matches, which we cannot resolve without full-match labels.

  We tested the obvious fix and it does not hold. On the two held-out DFL matches the tree's
  dead/alive F1 **peaks at the shipped threshold** (0.875 at 0.30, falling monotonically to
  0.84 at 0.55 and 0.56 at 0.90), and its predicted dead share there is **46.5% / 48.7%
  against a true 44.3% / 47.5%** — accurate to about two points. So the detector is correctly
  calibrated wherever we can actually measure it, and raising the threshold to force the share
  down destroys it. The 54.7% figure is therefore either genuine for this match or a residual
  provider shift we cannot separate without full-match labels. What we can say is that the
  Premier League benchmark (42–45% dead) is measured on Opta's definition of a delay, which
  need not match what this ground truth calls `D`: on the public five minutes the truth is only
  **19.6%** dead, far below either number.
- **Ideas from a physics-rich pipeline that did not transfer.** We implemented and measured four
  of them — a ball velocity-change proxy, a pass-by veto for aerial balls, touch-anchored switch
  timing, and short-horizon reachability. Each is sound in isolation; none changed our score
  beyond noise, because the 1 s hysteresis in (b) already absorbs the evidence noise they
  remove. Reachability did help on DFL (+0.042 spellF1) but over-segmented here, because DFL's
  possession convention is ~2.4× finer than this match's. That asymmetry is worth stating: any
  idea whose benefit works through *more frequent switching* cannot be validated on DFL for use
  here.
- **Extending step (g) into a single decoded model.** Step (g) learns dead ball only, and its
  output is merged with the rules by a union. The principled version is one H / A / D classifier
  over the same external matches, decoded across the whole match with Viterbi, so that spell
  structure — no flicker, sharp boundaries — comes from transition costs fitted on those games
  rather than from a fixed 1 s confirm time and a 2 s minimum run. Features would add the
  signals (b) already uses: nearest-player distance per team, ball speed, ball beyond the lines,
  and Smart Tagging context. We did not attempt it: it replaces the two steps that carry most of
  our score, and we could not have validated the replacement on five minutes of ground truth.

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

The expected SHA-256 is `a896a8d8bd5140e7306c5ce8f17be654474c145fe279d608383e442523fd0b31`.

The run is deterministic:

- It parses the raw files on every run, with no cache and no randomness.
- All arithmetic is float64 on the values exactly as written in the files.
- Dependencies are pinned, transitive ones included.
- Two runs from scratch give the same hash.

`python my_model.py --steps "" --name base` writes the baseline row of §6; `--steps a`, `ab`,
`abc`, `abcd`, `abcde` and `abcdef` write the next rows. Earlier submissions reproduce exactly:
`--steps abcdef` gives `bff95b03…dc08b`, `--steps abcde` gives `150bad7e…c67c59`, and
`--steps abc` gives `56ded60b…cd0fa`.
`python analysis/compare.py abcdef abcdefg` scores two step sets side by side, printing only the
scoreboard numbers and per-class F1. `selftest.py` checks reproduction, validity, invariants
and repository hygiene.

Step (g) reads the committed `models/deadball_tree.json` and walks it in numpy, so the run above
needs nothing beyond `requirements.txt`. Refitting the tree is a separate, offline step:
`training/train_deadball.py` needs scikit-learn and `databallpy` to download the DFL open
matches, and is **not** required to reproduce the submission.

## 9. Data handling and development process

The code was written with the help of an AI coding assistant (Claude Code). In line with the
terms of use, no match data, extract of it, or submission file was shared with it:

- Scripts read `data/` locally and printed only summary numbers: counts, percentiles and
  scorer metrics.
- The assistant never opened the ground-truth file or the scorer's per-transition reports.
- Step (g)'s training data is the **public** DFL/Sportec open dataset, not this match. Only its
  published per-frame `ball_status` was used as a label, and only aggregate metrics were ever
  printed. The fitted tree is committed in full (`models/deadball_tree.json`, 31 nodes) so the
  one learned part of the model can be read and checked like any other parameter.

The repository contains no match data. `data/`, the submission CSV and any derived tracking
are gitignored, and `selftest.py` checks this.
