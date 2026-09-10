"""Evaluate checkpoints on a block of seeds, for held-out model selection.

Usage: SEED0=1000 OUT=out.json python analysis/evaluate_models.py label=path [label=path ...]
Uses the notebook's evaluation settings (5% exploration, 3,000-decision cap) but a
selectable block of 40 seeds, so selection and confirmation can use disjoint sets.
"""
import sys, os, json, random, numpy as np, torch, gymnasium as gym, ale_py
from torch import nn
gym.register_envs(ale_py)

SEED0 = int(os.environ.get("SEED0", 1000))
SEEDS = list(range(SEED0, SEED0 + 40))
OUT = os.environ.get("OUT", "/tmp/eval_out.json")

class DQN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, n))
    def forward(self, x): return self.layers(x.float() / 255.0)

def make_env():
    e = gym.make("ALE/MsPacman-v5", frameskip=1, repeat_action_probability=0.25, render_mode="rgb_array")
    e = gym.wrappers.AtariPreprocessing(e, noop_max=30, frame_skip=4, screen_size=84,
                                        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False)
    e = gym.wrappers.FrameStackObservation(e, stack_size=4, padding_type="reset")
    return gym.wrappers.TimeLimit(e, max_episode_steps=3000)

env = make_env(); out = {}
if os.path.exists(OUT):
    out = json.load(open(OUT))
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    if label in out:
        continue
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = DQN(ck["n_actions"]); m.load_state_dict(ck["model"]); m.eval()
    sc = []
    for seed in SEEDS:
        rng = random.Random(seed + 10000); obs, _ = env.reset(seed=seed); tot = 0.0
        for step in range(3000):
            if rng.random() < 0.05:
                a = rng.randrange(ck["n_actions"])
            else:
                with torch.no_grad():
                    a = int(m(torch.as_tensor(np.asarray(obs)).unsqueeze(0)).argmax())
            obs, r, e1, t1, _ = env.step(a); tot += r
            if e1 or t1: break
        sc.append(tot)
    sd = float(np.std(sc, ddof=1))
    out[label] = {"seed_block": SEED0, "scores": sc, "mean": float(np.mean(sc)),
                  "sd": sd, "se": sd / len(sc) ** 0.5}
    print(f"{label:22} n={len(sc)} mean {np.mean(sc):7.1f}  se {sd/len(sc)**.5:5.1f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
env.close(); print("DONE")
