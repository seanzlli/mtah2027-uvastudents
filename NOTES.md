# UvAStudents — working notes

Running log for the technical document. One command regenerates the submission:

```bash
pip install -r requirements.txt   # exact pins; tested with Python 3.13.13
python my_model.py          # -> evaluation/results/UvAStudents.csv
python selftest.py          # all correctness checks, ~5 s; exit code 0 = no FAIL
./check.sh                  # validator + scoreboard (public 5 minutes, sanity check only)
python my_model.py --steps "" --name ablation_base   # any subset of improvements, for ablation
```

**Before uploading:** run `python my_model.py`, then `python selftest.py`. The `on-disk file`
check proves the CSV you upload is exactly what the current code produces.

`selftest.py` checks:

- **baseline port:** `--steps ""` equals `baseline_smooth50` on every graded frame.
- **submission valid:** the official validator's checks pass.
- **invariants:** only H/A/D/X; X outside the halves; under 2% X inside them; D before each
  half's kick-off strike.
- **plausibility (WARN only):** dead-ball share 25–55%; 100–600 possessions of mean 5–30 s.
- **reproducible:** a second, independent parse of the raw tracking gives identical labels.
- **on-disk file:** `evaluation/results/UvAStudents.csv` matches the current code.
- **public alignment:** the baseline port reproduces the published public scores
  (0.6207 / 0.3256). This proves frame alignment. The submission's own public score is printed
  as INFO only, not as a target.
- **repo hygiene:** no match data, cache or submission CSV tracked by git.
- **notebook outputs:** WARN if the version git would commit (the staged version, else the
  last commit) holds saved outputs. Unstaged outputs in a working copy are only mentioned.

## Ground rules we follow

- **No match data leaves the machine.** Scripts print summary counts only. `data/`, every CSV
  (submissions, the public ground truth) and `evaluation/reports/` (which lists ground-truth
  transition frames) are never opened by hand or pasted anywhere.
- **Labels come from an algorithm.** There are no hand labels, and we wrote no frame numbers or
  constants tied to this match. Match facts come only from what the organisers publish in
  `games.py`: the half boundaries (`half_video_blocks`) and, from step (d) on, the second-half
  kick-off anchor (`kickoff_2nd`). The other kick-offs and all restarts are found from the Smart
  Tagging plus the tracking.
- **Parameters come from football reasoning**, written down next to each value in `PARAMS`
  (`my_model.py`) and summarised below. The public 5 minutes (17 possessions) are scored once
  per idea as a pipeline sanity check. Nothing is swept against them. Where an idea was revised,
  the reason was input-data diagnostics, and that is recorded here.
- **Input-data diagnostics** (`analysis/data_stats.py`) read only tracking and tagging, never
  the ground truth, and print aggregates (percentiles, counts).

## What the input data looks like (aggregates, `analysis/data_stats.py`)

- Ball visible on 59.3% of graded frames. Ball beyond the lines on 4.2% of visible frames.
- **The ball is never missing for 1–4 s.** In the graded halves the 150 gaps are either
  short (17 gaps ≤ 1 s) or long (133 gaps > 4 s; 37 of 15–30 s, 19 over 30 s). A "carry
  possession for a couple of seconds" rule therefore touches almost nothing. The real question
  is what to do across a 10–30 s gap.
- Field-average player speed: median 1.5 m/s over graded frames; about 0.7–1.1 m/s in the
  15 s before a tagged restart; about 2.3 m/s 5 s after it.
- Smart Tagging `SET PIECES` (59): windows are 35 s long (5 s for kick-offs) and open about
  5 s before the strike, so a tag marks *the restart and the play after it*, not the stoppage.
  The second-half kick-off tag starts about 114 s into the graded second half, consistent with
  the FAQ's "graded from 110,808 but kicks off at 113,742".

## Step 0 — baseline port

`my_model.py --steps ""` reproduces `baseline_smooth50`: nearest tracked player to the ball,
centred majority vote over ±2 s, D where no ball is seen in the window. **0 differing frames out
of 158,857 graded.**

## Step (a) — dead ball before Smart Tagging restarts

**What it does.** For every `SET PIECES` tag:

1. **Strike.** The restart is the first frame in the tag window where the ball moves faster
   than 4 m/s (measured over 0.2 s) after at least 1 s without such motion. If the strike is
   never seen, use 3.5 s into the tag.
2. **Stoppage.** D for at least 2 s before the strike. Then walk backwards while there is
   dead-ball evidence:
   - the field average is at walking pace (< 1.2 m/s over the trailing 2 s), **or**
   - the ball is beyond the lines (1 m margin), **or**
   - the ball is stationary (< 1 m/s).

   Gaps of up to 1 s without evidence are tolerated. The walk stops at a live strike of the
   ball inside the pitch, or at a cap per restart type.
3. **Kick-offs at the start of a half.** D from the start of the graded half up to the strike.
   No frame constant is used.

**Why these values (football reasoning).**

| parameter | value | reason |
|---|---|---|
| strike speed | 4 m/s | any restart pass or throw is ≥ 5 m/s; a ball carried or placed moves < 2 m/s |
| quiet before strike | 1 s | a ball still rolling out over the line is not the restart |
| speed lag | 0.2 s | automatic ball positions jitter by tens of cm; frame-to-frame differences would read as several m/s |
| out margin | 1 m | Law 9: the whole ball must cross the line; slack for tracking error and aerial projection |
| walking pace | 1.2 m/s | walking is 1–1.4 m/s; in a stoppage most players walk or stand, while live play averages ~2 m/s |
| stationary ball | 1 m/s | true speed ~0, plus the jitter that 0.2 s of noisy positions produces |
| minimum stoppage | 2 s | Laws 8, 13, 15–17: ball stationary / taker set before every restart |
| evidence tolerance | 1 s | a player jogging over to take a throw should not end the stoppage |
| caps | throw-in 20 s, goal kick 30 s, corner 40 s, free kick 60 s, kick-off after goal 120 s | typical top-flight times from ball dead to restart (celebrations and VAR checks for goals) |
| fallback lead | 3.5 s | the starter notebook documents kick-off tags opening ~3.5 s early |

**Revision (input-data diagnostics, not the score).** Version 1 vetoed any tag whose 2 s
before the strike were not at walking pace and had no ball beyond the lines. It dropped 22 of
59 tags, mostly goal kicks (8/10) and corners (5/12). That is football, not tagging errors: at
a goal kick the outfield players run back into shape, and at a corner they jostle in the box.
Version 2 trusts every tag and adds "ball stationary" as evidence (those restarts require it).
Both versions were scored. The switch was decided from the dropped-by-type counts.

**Resulting stoppage lengths** (median, input data): corner 15 s, free kick 23 s, kick-off
after goal 40 s, throw-in 5 s, goal kick 2.8 s. The last two are shorter than real football.
The walk-back probably stops at the ball being played back to the taker, or the strike detector
fires on the ball going out. This is a known limitation.

**Sanity check (public 5 min, 17 possessions; differences < ~0.1 spellF1 are noise here).**

| version | macroF1 | spellF1 | poss | dur |
|---|---|---|---|---|
| baseline_smooth50 (step 0) | 0.6207 | 0.3256 | 26 | 10.8 |
| (a) v1, veto tags without evidence | 0.6369 | 0.3349 | 26 | 10.7 |
| **(a) v2, trust tags, stationary ball** | **0.6519** | **0.3545** | 27 | 10.3 |

The public window holds few restarts, so (a) is expected to matter more on the full match.

## Step (b) — hysteresis possession switching

**What it does.** Replaces the majority vote. Live play is cut into segments at every D
stretch, and each segment runs a small state machine:

1. **Evidence per frame.** A team is *on the ball* when its nearest player is within 2 m of the
   ball **and** at least 1 m closer than the nearest opponent. Otherwise the frame is
   loose/contested and carries no evidence.
2. **Start of a segment.** The first team confirmed on the ball (rule 3) owns the segment from
   its first frame. CHALLENGE.md: "whenever play restarts after a stoppage, there is a clear
   possessor."
3. **Switching.** The other team must collect 1 s (25 frames) of evidence with no evidence for
   the current holder in between. Frames with no evidence (ball loose, in flight, missing) do
   not break the run, so "interception, then a team-mate's touch" counts.
4. **Backdating.** When a switch is confirmed, the boundary goes back to the first evidence
   frame of that run, which is the moment control was gained.
5. **No evidence at all in a segment** (35 of 148 segments, mostly very short): majority of
   nearest-player answers, or X if the ball is never seen.

The ball-missing → D rule is unchanged from the baseline in this step, so (b) changes only who
is in possession.

**Why these values.**

| parameter | value | reason |
|---|---|---|
| control radius | 2 m | playing distance: a dribbler keeps the ball within about a stride (1.5–2 m) |
| control margin | 1 m | "clearly closer", not a 50/50 duel; well above automatic tracking jitter |
| confirm | 1 s | a touch, deflection or lost aerial duel lasts < 0.3 s and is not control (CHALLENGE.md); a controlled reception plus the next action reaches 1 s |
| backdate | to first touch | possession changes when control is gained; spellF1 charges a late boundary heavily (1 s late → 0.65) |
| loose ball | keep last team | CHALLENGE.md: "possession stays with the last team that had it" |

**Effect on the full match** (validator counts, no ground truth): 568 → 259 possessions, mean
6.9 s → 15.2 s. Possessions stop flickering. A team clearly on the ball is detected on 26.9%
of graded frames.

## Step (c) — missing-ball handling

**Input-data finding that shaped it** (aggregates only):

| | ball visible | inside long gaps (> 2 s) |
|---|---|---|
| field speed, median | 2.01 m/s | 1.04 m/s |
| frames at walking pace (< 1.2 m/s) | 13.6% | 62.2% |

- 71.5% of the restart stoppages found by step (a) lie inside a long ball gap.
- In 80% of long gaps the ball is struck (> 4 m/s) within 1 s of reappearing.

In this tracking, **a long missing-ball gap is mostly a stoppage**: the ball vanishes when play
stops. So plain "carry possession through the gap" would be wrong. Instead:

1. **Gaps ≤ 3 s:** carry possession (a 50–60 m diagonal is in the air about 3 s).
2. **Longer gaps:** D by default. The exception is any stretch of ≥ 2 s where the field moves
   at live-play tempo (≥ 2 m/s centred 2 s average, the typical field speed with the ball in
   play). There play is going on unseen, so possession is carried.
3. **Ball seen out first:** if the ball was seen beyond the lines immediately before it
   vanished, D starts at the first such frame (Law 9).
4. **Gap edges:** D starts and ends exactly at the gap edges. The baseline's ±2 s window left
   the first and last 2 s of every gap alive.

**Why these values.** 3 s is a long ball's flight time. 2 m/s is live-play tempo, set
deliberately well above walking pace (1.2 m/s): the input data gives a strong prior that long
gaps are stoppages, so "live" needs positive evidence, not merely "not walking". 2 s is the
minimum stoppage from step (a), reused as the minimum live stretch.

**Effect on the full match:** inside long gaps 3,183 s are D and 242 s are carried as live.
Possessions 259 → 254, mean 14.4 s.

## Step (d): the second-half kick-off from `games.py`

**Source.** FAQ.md: "the second half is graded from frame 110,808 but does not kick off until
**113,742**. Those 2,934 frames are graded and all of them are `D`."

**What it does.** Everything from the first graded frame of half 2 up to the kick-off is D.
Both frames are derived from `games.py`, not written in our code:

- the start comes from `half_video_blocks`;
- the kick-off comes from the `kickoff_2nd` anchor via `anchor_video_frame`.

The anchor is "the ball is struck on this frame", so the strike frame is the last dead frame
and play is live from the next one. That matches the FAQ's count of 2,934 D frames, and the
starter loader's `KICKOFF_VIDEO_FRAMES`. The published kick-off also replaces step (a)'s
tracking estimate for that tag. The first half needs nothing, because it is graded from its
kick-off (FAQ.md).

**Why it was needed (input data).** Step (a) already made all 2,934 pre-kick-off frames D. But
the tracking never sees the second-half kick-off strike, so (a) fell back to "3.5 s into the
tag", and that ended the dead stretch 5 frames *after* the real kick-off. Step (d) makes those
5 frames live. It is a correctness fix, not a score lever, and it cannot touch the public
window, which is in the first half.

**Sanity check:** 5 graded frames change, none inside the public window. Public scores are
unchanged (spellF1 0.4941, macroF1 0.7277, 17 possessions; F1 H 0.8587, A 0.8227,
D 0.5016). **Kept.**

## Sanity-check scores (public 5 min — pipeline check, not model selection)

Each idea was scored once after it was written. (a) was scored twice; see its revision note.

| model | macroF1 | spellF1 | poss (public) | dur | poss (full match, validator) |
|---|---|---|---|---|---|
| baseline_smooth50 | 0.6207 | 0.3256 | 26 | 10.8 | 594 (7.2 s) |
| + (a) restarts | 0.6519 | 0.3545 | 27 | 10.3 | 568 (6.9 s) |
| + (b) hysteresis | 0.6818 | 0.4824 | 17 | 16.3 | 259 (15.2 s) |
| **+ (c) missing ball = submission** | **0.7277** | **0.4941** | 17 | 16.0 | 254 (14.4 s) |

Ground truth (public): 17 possessions of mean 14.2 s.

**Noise.** `run_evaluation.py --bootstrap 1000` gives a spellF1 band of **~0.33** on the public
window (17 possessions). None of the gaps above can be called significant there. The case for
each step rests on the football reasoning and on the full-match possession counts becoming
realistic, not on these numbers.

**Reproducibility.**

- **From scratch.** Every run of `python my_model.py` parses the raw tracking and the Smart
  Tagging from `data/`. There is no cache, no randomness and no state from earlier runs.
- **Exact numbers.** All arithmetic is float64 on the values exactly as written in the files.
  An earlier float32 cache shifted 3 of 158,857 graded frames at threshold edges, which is why
  it was removed. The public scores were unchanged.
- **Pinned dependencies.** `requirements.txt` pins numpy 2.5.4, pandas 3.1.0, python-dateutil
  3.0.0, six 1.17.0 and tzdata 2026.5 (transitive dependencies included). The pins match the
  environment the file was built in, and `pip check` reports no conflicts. A fresh virtualenv
  install from these pins has not been tried.
- **Two runs, same file.** Deleting the output and running twice gives the same SHA-256:
  `56ded60ba2daef692949575dc7dfa4be4aa8f5b9ed2302eee7f02c4a3e8cd0fa`. Anyone with the data
  bundle can check that this hash reproduces.
- **Validator:** OK, 199,319 rows. `selftest.py`: 14 checks, 0 FAIL, 0 WARN.

## Limitations / ideas not done

- **Restart stoppages are too short for goal kicks (median 2.8 s) and throw-ins (5 s).** The
  walk-back probably stops at the ball being played back to the taker, or the strike detector
  fires on the ball going out.
- **Untagged stoppages with the ball visible are missed.** For example, a free kick with the
  ball standing still in view. The tagging lists only 5 free kicks and 26 throw-ins. A generic
  detector would be "field at walking pace + stationary ball for ≥ 2 s".
- **Evidence covers 27% of graded frames.** Possession across long live stretches without the
  ball relies on carrying. Team shape (which side is pushing up) could fill in.
- **Smart Tagging `Team` tags on phases are unused.** They could give a prior on the possessor
  for the 35 segments with no confirmed possessor (about 1,400 graded frames written as X).
- **Unused assets:** the broadcast feed and the homography (useful only for looking at errors
  on video).
- On the public window, transitions land about 0.5 s early on average (`lat_f` −12.5).
  This was not investigated, because adjusting for it would be tuning on 17 possessions.

## Files

- `my_model.py`: the model; `PARAMS` holds every value with its reason.
- `selftest.py`: one-command correctness checks (see the top of this file).
- `check.sh`: validator + scoreboard only.
- `analysis/data_stats.py`: the input-data aggregates quoted above.
- `analysis/compare.py OLD NEW`: scores two step sets side by side on the public window.
  It prints the scoreboard numbers and per-class F1 only, and deletes the scorer's reports
  unread.
- `requirements.txt`: exact pins for reproduction.
- `evaluation/results/*.csv` (the submission included) is gitignored.
