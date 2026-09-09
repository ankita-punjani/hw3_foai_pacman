# Ms. Pac-Man DQN — Class 3 Assignment

A Deep Q-Network trained on `ALE/MsPacman-v5` using the class notebook, run locally on Apple Silicon.

**My three choices: exploration 0.10, 600 episodes, learning rate 0.0001.**
**Result: mean 432.0 after training, against an untrained baseline of 492.0 — a change of −60.0.**
The agent did not improve on the graded metric. All five scores are reported below, along with the
evidence and an explanation of what did and did not change.

The notebook and DQN implementation are the class starter from
[pepealonso95/pacman-dqn](https://github.com/pepealonso95/pacman-dqn); the hyperparameter choices, the
executed run, the analysis, and this write-up are mine.

---

## Open and run

```bash
git clone https://github.com/ankita-punjani/homework3_foai_pacman.git
cd homework3_foai_pacman
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab pacman_dqn.ipynb
```

Select a Python 3.11–3.13 kernel and choose **Run All**. CUDA, Apple MPS, and CPU are detected
automatically. The notebook also opens in
[Google Colab](https://colab.research.google.com/github/pepealonso95/pacman-dqn/blob/main/pacman_dqn.ipynb)
(Runtime → Change runtime type → T4 GPU).

[**`pacman_dqn.ipynb`**](pacman_dqn.ipynb) is the executed notebook, saved after the final training and
evaluation run with all cell outputs visible. Its scores, dashboard, and 26 gameplay GIFs render on GitHub
without rerunning anything.

---

## My three hyperparameters

| Setting | Starter | My run |
|---|---|---|
| Exploration | 0.20 | **0.10** |
| Episodes | 100 | **600** |
| Learning rate | 0.0001 | **0.0001** |

**Exploration = 0.10.** The environment already applies **25% sticky actions**, so it injects a lot of
randomness before epsilon adds any. The replay buffer holds only **5,000 transitions ≈ 8 games**, so
training data is nearly on-policy and overwritten quickly — filling it with random moves wastes the only
memory the agent has. Epsilon is also **constant here, with no decay**, so my value applied for all 600
games; since evaluation always runs at 5%, training at 20% would mean optimising a noisier policy than the
one being graded.

**Episodes = 600.** A five-episode setup check took 75 seconds and produced 491 updates — enough to prove
the pipeline, far too few to learn. Episodes govern how much experience the agent sees, so I took the
largest budget that fit comfortably in one sitting (~52 min). This also produced 24 intermediate GIFs and
checkpoints every 25 episodes.

**Learning rate = 0.0001.** The standard Adam DQN value and the notebook's reference point. With a small
buffer and a target network resyncing every 1,000 decisions, the setup already chases a moving target; a
larger rate learns faster per update but risks oscillation, and a diverged run would have wasted the whole
budget.

### Other settings

I changed one setting beyond the three: **`SHOW_POPUPS` from `True` to `False`**, because I executed the
notebook headlessly with `jupyter nbconvert --execute`, which has no desktop for the Tk popup player. This
affects **preview display only** — not the agent's decisions, learning, or scores.

**All learning and evaluation settings are unchanged**: the same five seeds `[101, 202, 303, 404, 505]`,
5% evaluation exploration, 3,000-decision cap, replay capacity 5,000, batch 32, gamma 0.99, target sync
every 1,000 decisions, reward clipping to [−1, 1], demo every 25 episodes.

---

## Actual training budget

| | Value |
|---|---|
| Episodes | **600 requested, 600 completed** |
| Decisions | 365,560 |
| Learning updates | 91,141 |
| Elapsed (training + periodic demos) | 3,100 s ≈ **51.7 min** |
| Hardware | Apple M4, 16 GB RAM, **MPS** backend |
| Software | macOS 15.6.1, Python 3.13.15, PyTorch 2.14.0, Gymnasium 1.3.0, ALE-py 0.11.2 |
| Status | **`completed`** — the run was not interrupted, and learning updates did occur |

---

## What I expected, and what I observed

**Expected.** A modest but visible improvement — perhaps 492 → 700 — with the agent learning to keep eating
pellets instead of dying early, and most of the gain arriving in the first 200 episodes.

**Observed.** The score did not improve. Training scores rose about 15% over the first ~100 episodes
(50-game mean 677 → 784), then plateaued for the remaining 500 episodes, ending at 720. The final
evaluation came in 60 points *below* the untrained baseline, and the 24 progress demos show no upward trend.

Two things I did not expect:

**The untrained baseline is not a weak opponent.** Comparing the two networks directly
([`analysis/action_distribution.py`](analysis/action_distribution.py)), the untrained network picks
**UPLEFT 100% of the time**, regardless of what is on screen — its Q-values are nearly flat, so `argmax`
always returns the same output. In Ms. Pac-Man that is a surprisingly effective degenerate strategy: maze
walls plus 25% sticky actions turn a constant input into a steady patrol along a pellet route. Seed 404
scored 800 that way. The baseline is an untrained network, not a random-action agent, and it sets a real bar.

**The network did learn, even though the score did not improve.**

| | Untrained | Trained (600 episodes) |
|---|---|---|
| Greedy action distribution | UPLEFT 100.0% | DOWNLEFT 42.8%, DOWNRIGHT 41.5%, UP 14.2%, UPRIGHT 1.5% |
| Mean Q-value spread (max − min) | 0.076 | **0.208** |
| Distinct actions used | 1 of 9 | 4 of 9 |
| Mean game length (decisions) | 589.0 | 512.2 |

The policy went from one constant action to genuinely state-dependent, with value estimates about 3× more
discriminating. It just did not convert that into points.

**On loss.** Mean update loss rises 0.024 → 0.114 and rolls over near episode 400. That is Q-value
magnitudes growing as discounted returns propagate, not divergence — and a reminder that **lower training
loss does not mean better play**: loss went *up* over the stretch where the policy structure was improving.

---

## Evaluation: all five scores

Identical settings before and after — seeds `[101, 202, 303, 404, 505]`, 5% exploration, 3,000-decision
cap. The baseline is an untrained network.

| Game | Seed | Before (untrained) | After (trained) | Change |
|---:|---:|---:|---:|---:|
| 1 | 101 | 350 | 450 | +100 |
| 2 | 202 | 500 | 410 | −90 |
| 3 | 303 | 320 | 350 | +30 |
| 4 | 404 | 800 | 480 | −320 |
| 5 | 505 | 490 | 470 | −20 |
| **Mean** | | **492.0** | **432.0** | **−60.0** |

Two of five seeds improved, three declined; neither agent hit the time limit on any game. The paired
differences have a standard deviation of 161.1 (SE 72.0, t = −0.83), so the −60 is within noise rather than
a reliable decline. Full data: [**`comparison.json`**](results/comparison.json).

### Training dashboard

![Training dashboard](results/training_dashboard.png)

Raw score per game with the 25-game running mean (left); mean update loss (centre); training exploration
(right), dropping from the 1.0 warm-up to a flat 0.10 for the rest of the run.

### Gameplay

| Untrained (before training) | Best trained (final model, best of 5 games) |
|---|---|
| ![Untrained](results/demos/episode_0000.gif) | ![Trained](results/demos/final_best.gif) |

Intermediate progress samples (one evaluation game on seed 101 after every 25th episode):

| Episode 25 | Episode 150 | Episode 300 | Episode 450 | Episode 600 |
|---|---|---|---|---|
| ![25](results/demos/episode_0025.gif) | ![150](results/demos/episode_0150.gif) | ![300](results/demos/episode_0300.gif) | ![450](results/demos/episode_0450.gif) | ![600](results/demos/episode_0600.gif) |
| 380 | 670 | 1410 | 540 | 450 |

All 24 progress GIFs are in [`results/demos/`](results/demos), with scores in
[`demo_scores.json`](results/demo_scores.json). GIFs play at 4× speed and show the first 20 seconds of game
time.

---

## What the agent observes, does, and is rewarded for

**Observations — what it sees.** Not the game's memory, and no list of ghost positions. It sees the
**screen**: each frame is turned to greyscale and shrunk to 84×84 pixels, and the **last four frames are
stacked** into one observation. Four rather than one because a single still image cannot show motion — from
one frame you cannot tell whether a ghost is closing in or moving away.

**Actions — what it can do.** Nine **joystick moves**: `NOOP`, `UP`, `RIGHT`, `LEFT`, `DOWN`, and the four
diagonals. One decision covers four emulator frames, and 25% of the time the game repeats the previous
action instead ("sticky actions"), so control is deliberately imprecise.

**Rewards — how it is graded.** Ordinary **game points** — pellets, power pills, fruit, eaten ghosts. During
learning each reward is **clipped to [−1, 1]** so one large payoff cannot swamp the updates; every score in
this README is the raw, unclipped game score.

**The learning rule.** A convolutional network predicts a value for each of the nine actions. Experience is
stored in a replay buffer and sampled in batches of 32, so updates are not dominated by whatever just
happened. A slowly-updated copy of the network (the "target network") supplies the prediction target, and
future reward is discounted by gamma = 0.99. The gap between predicted and target value is closed with
Huber loss and Adam.

---

## One observed limitation

**The agent's performance peaked early and then degraded — it was better at episode 125 than at episode
600.** I evaluated every saved checkpoint under the frozen protocol, then re-tested the key models on 40
fresh paired seeds to escape the five-game noise (details in *Additional analysis* below). Episode 125
averaged 934 and episode 300 averaged 850, while the final episode-600 model dropped to 603. The run's best
policy was discarded in favour of its last one.

This looks like an unstable value estimate rather than too little data. Two things point the same way: mean
update loss climbs steadily to episode 400, and the target network resyncs every 1,000 decisions — only
about 250 gradient updates apart. With a 5,000-transition buffer feeding it, the network chases a target
moving nearly as fast as it is, so Q-values inflate and the greedy policy drifts away from the good one it
had found.

A visible symptom in the final network: **it uses only 4 of its 9 actions**, picking DOWNLEFT or DOWNRIGHT
84% of the time and never choosing plain `LEFT`, `RIGHT`, `DOWN`, or `NOOP` greedily. That repertoire is too
narrow to escape a ghost approaching from a direction those four actions do not address, which is why its
games are *shorter* than the baseline's (512 vs 589 decisions).

## One next experiment

**Change one setting: raise `TARGET_EVERY` from 1,000 decisions to 5,000**, holding exploration at 0.10,
episodes at 600, learning rate at 0.0001, and the replay buffer at 5,000.

I tested two other candidates first and both were wrong. Raising the replay buffer to 50,000 made
performance *worse* at this budget. Raising the episode count would not help either, because the agent had
already peaked by episode 125 and spent the next 475 episodes getting worse — more episodes would buy more
degradation, not more learning. The problem is not how much experience the agent sees; it is that its value
estimates do not hold still.

Slowing the target network is the smallest change that addresses that directly. Resyncing every 5,000
decisions (≈1,250 updates) gives the online network a stationary target five times longer, which is the
standard remedy for the inflating Q-values and late-run drift this run showed — and the original DQN paper
syncs far more slowly than this notebook's default. It costs no extra memory and no extra training time. I
would expect the episode-600 checkpoint to hold near the ~930 that episode 125 reached here instead of
decaying to 603.

---

## Files

- [`pacman_dqn.ipynb`](pacman_dqn.ipynb) — executed notebook with all outputs
- [`results/config.json`](results/config.json) — settings, hardware, exact package versions
- [`results/training.csv`](results/training.csv) — one row per training game (600 rows)
- [`results/training_summary.json`](results/training_summary.json) — status, episodes, decisions, updates, elapsed
- [`results/comparison.json`](results/comparison.json) — all five before/after scores
- [`results/baseline.json`](results/baseline.json) · [`results/demo_scores.json`](results/demo_scores.json)
- [`analysis/`](analysis) — scripts for the checkpoint sweep, plot, action-distribution comparison, and 40-seed verification

**Interrupted runs.** The submitted run **completed all 600 episodes with 91,141 learning updates** — it was
not interrupted and did not lack learning updates. One *additional* experiment described below (the
50,000-buffer ablation) **was interrupted at episode 265 of 600** when my machine's app quit mid-run, so it
has no `trained.pt`, no `comparison.json`, and no `training_summary.json`. It is reported as an ablation
only, never as a submitted result.

**Model checkpoints** (`untrained.pt`, `trained.pt`, and 24 periodic checkpoints, ~6.8 MB each, ~169 MB
total) are **not in this repository**. They are kept locally in the ZIP the notebook produced,
`pacman_runs/20260908_215421_812050.zip` (161 MB), as the assignment allows. Rerunning the notebook
regenerates equivalent checkpoints under a fresh `pacman_runs/` folder.

---

## Additional analysis

These extra checks are beyond the assignment's requirements. They do not change the reported score of
**432.0**, which stands as the notebook's protocol produced it.

**Every checkpoint under the frozen protocol.** I re-ran the same five seeds, 5% exploration, and
3,000-decision cap on all 24 periodic checkpoints
([`analysis/checkpoint_sweep.py`](analysis/checkpoint_sweep.py),
[`results/checkpoint_sweep_run1.json`](results/checkpoint_sweep_run1.json)):

![Checkpoint sweep](results/checkpoint_sweep.png)

They average 650.0 with a standard deviation of 184.6 and a range of 400–982. That spread is wider than the
effect being measured, so any single five-game snapshot is dominated by noise.

**A 40-seed paired re-test.** Because the untrained baseline was itself only a five-game estimate, I
re-evaluated the key models on 40 fresh seeds disjoint from the graded five, paired so every model played
identical games ([`analysis/verify_40_seeds.py`](analysis/verify_40_seeds.py),
[`results/verification_40seeds.json`](results/verification_40seeds.json)):

| Model | Mean over 40 seeds | vs untrained | Paired t | Seeds won |
|---|---|---|---|---|
| Untrained baseline | 522.0 | — | — | — |
| Final model (ep 600) | 602.8 | +80.8 (1.15×) | +1.23 (not significant) | 28/40 |
| Checkpoint ep 300 | 849.5 | +327.5 (1.63×) | **+3.19** | 32/40 |
| Checkpoint ep 125 | **934.0** | +412.0 (1.79×) | **+4.64** | 32/40 |

This is the evidence behind the limitation above: mid-training checkpoints beat the untrained network
significantly, while the final model's edge is real but statistically inconclusive.

**Replay-buffer ablation (interrupted).** I tested whether the 5,000-transition buffer was the bottleneck by
rerunning with `REPLAY_CAPACITY = 50,000` and everything else identical. It performed **worse** at matched
budget — 308.0 vs 504.0 at episode 250 — and across the first ten checkpoints averaged 476.4 against the
5,000 buffer's 641.4. My reading: with only ~365k total decisions, a large buffer keeps warm-up and early
bad-policy experience in the sampling pool far longer, so a scarce update budget is spent rehearsing stale
data. Data: [`results/ablation_buffer50k/`](results/ablation_buffer50k).
