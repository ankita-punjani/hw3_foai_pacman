"""Compare the greedy action distribution and Q-value spread of two checkpoints.

Usage: python analysis/action_distribution.py <run_dir>
Evaluates untrained.pt and trained.pt over the notebook's five frozen evaluation
seeds at 5% exploration, and reports which actions each network actually picks.
"""
import sys, random, numpy as np, torch, gymnasium as gym, ale_py
from torch import nn
from collections import Counter
gym.register_envs(ale_py)
R = sys.argv[1]

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

MEANINGS = make_env().unwrapped.get_action_meanings()
for tag in ("untrained", "trained"):
    ck = torch.load(f"{R}/{tag}.pt", map_location="cpu", weights_only=True)
    m = DQN(ck["n_actions"]); m.load_state_dict(ck["model"]); m.eval()
    counts, qspread = Counter(), []
    env = make_env()
    for seed in [101, 202, 303, 404, 505]:
        rng = random.Random(seed + 10000); obs, _ = env.reset(seed=seed)
        for step in range(3000):
            with torch.no_grad():
                q = m(torch.as_tensor(np.asarray(obs)).unsqueeze(0))[0]
            greedy = int(q.argmax()); qspread.append(float(q.max() - q.min()))
            counts[MEANINGS[greedy]] += 1
            a = rng.randrange(ck["n_actions"]) if rng.random() < 0.05 else greedy
            obs, r, e1, t1, _ = env.step(a)
            if e1 or t1: break
    env.close()
    tot = sum(counts.values())
    print(f"--- {tag}: {tot} greedy decisions | mean Q-range {np.mean(qspread):.3f}")
    print("   ", ", ".join(f"{k} {v/tot:.1%}" for k, v in counts.most_common()))
