# UvAStudents: More than a Hack 2027, Phase 1 submission

Ball possession (`H` / `A` / `D` for every frame) from the automatic tracking and the Smart
Tagging set pieces. Seven of the model's eight steps are hand-written football rules; the one
exception is a 31-node decision tree fitted on **other** matches' public data. No labels from
this match are used anywhere. It is the starter baseline plus eight changes:

- dead ball before restarts;
- hysteresis possession switching;
- missing-ball handling;
- the published second-half kick-off;
- more dead-ball evidence (ball out of play, set-piece windows, goals);
- no `X` on graded frames (the scorer's own spellF1 fill);
- dead ball from a decision tree over player positions, for the stoppages nothing tags;
- dead-ball evidence with the ball unseen (corrects the backward search in step 1).

- **[TECHNICAL.md](TECHNICAL.md):** the technical document. Covers the approach, assets, rules,
  limitations and reproduction.
- **[NOTES.md](NOTES.md):** the development log, with parameter reasoning and sanity-check scores.

## Install

Tested with Python 3.13.13 on macOS arm64.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins exact versions: numpy 2.5.4, pandas 3.1.0 and their dependencies.

## Data

The match data is gated and **not** in this repository. The code reads the organisers' bundle
from `./data`, or from `MTAH_DATA_DIR` if that is set ([DATA.md](DATA.md)). Only these three
tactical files are read:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

## Run

```bash
python my_model.py
```

This writes `evaluation/results/UvAStudents.csv`: 199,319 rows, built from scratch in about
3 s. It is deterministic, with no cache and no randomness. The expected SHA-256 is
`a896a8d8bd5140e7306c5ce8f17be654474c145fe279d608383e442523fd0b31`. Use `shasum -a 256`, or
`sha256sum` on Linux.

## Check

```bash
python evaluation/validate_submission.py evaluation/results/UvAStudents.csv   # official format checks
python selftest.py      # 14 checks: baseline reproduction, validity, reproducibility, repo hygiene
./check.sh              # validator + scoreboard on the public 5 minutes (sanity check only)
```

## What we added

| file | purpose |
|---|---|
| `my_model.py` | the model; `PARAMS` lists every value with its football reason; `--steps` runs any subset of the eight improvements (`abcdefgh`) |
| `deadball.py` | step (g): the features and per-match normalisation, and the numpy tree walk |
| `models/deadball_tree.json` | the fitted dead-ball tree, 31 nodes of plain thresholds |
| `training/train_deadball.py`, `training/requirements.txt` | refits that tree from DFL open data, with its pinned packages; **not** needed to reproduce the submission |
| `selftest.py` | one-command correctness checks |
| `check.sh` | validator plus scoreboard |
| `analysis/compare.py` | scores two step sets side by side: scoreboard numbers and per-class F1 only |
| `analysis/data_stats.py` | aggregate input-data statistics used in the design (no ground truth) |
| `TECHNICAL.md`, `NOTES.md` | documentation |

Everything else is the organisers' starter repository, unchanged apart from `requirements.txt`
(pinned), `.gitignore` and this README. The README replaces the organisers' challenge
description, which is in their repository,
<https://github.com/metrica-sports/more-than-a-hack-2027>. Their other documents are kept here
unchanged: [CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md),
[SUBMISSION.md](SUBMISSION.md), [RULES.md](RULES.md), [FAQ.md](FAQ.md) and
[OVERVIEW.md](OVERVIEW.md).

## Licence

The organisers' starter code and documentation are under the MIT Licence ([LICENSE](LICENSE)).
The match data is not covered by it and is not in this repository. Step (g) is trained on the
DFL/Sportec open data, which is CC BY 4.0. TECHNICAL.md §5 gives the citation.
