"""Rebuild results/checkpoint_sweep.png from the saved evaluation JSONs."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sw = json.load(open("results/checkpoint_sweep.json"))
ho = json.load(open("results/verification_40seeds.json"))
ab = json.load(open("results/ablation_buffer50k/checkpoint_sweep.json"))

ks = sorted(int(k) for k in sw if k not in ("0", "final"))
means = [sw[str(k)]["mean"] for k in ks]
base = sw["0"]["mean"]; final = sw["final"]["mean"]

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))

a = ax[0]
for k in ks: a.scatter([k]*5, sw[str(k)]["scores"], s=14, color="#9ecae1", zorder=2)
a.plot(ks, means, "o-", color="#08519c", lw=2, label="Checkpoint mean (5 games)", zorder=3)
a.axhline(base, color="#e6550d", ls="--", lw=2, label=f"Untrained baseline ({base:.0f})")
a.scatter([450], [final], s=150, color="#31a354", marker="*", zorder=5,
          label=f"Submitted final, ep 450 ({final:.0f})")
a.set_xlabel("Training episode"); a.set_ylabel("Evaluation score")
a.set_title("Submitted run: every checkpoint, frozen protocol"); a.legend(fontsize=8); a.grid(alpha=.3)

b = ax[1]
u = ho["untrained"]["scores"]; f = ho["final_ep450"]["scores"]
b.scatter(np.random.default_rng(0).normal(1, .06, len(u)), u, s=22, color="#e6550d", alpha=.65)
b.scatter(np.random.default_rng(1).normal(2, .06, len(f)), f, s=22, color="#31a354", alpha=.65)
b.hlines([np.mean(u)], .75, 1.25, color="#a63603", lw=3)
b.hlines([np.mean(f)], 1.75, 2.25, color="#006d2c", lw=3)
b.set_xticks([1, 2]); b.set_xticklabels([f"Untrained\n{np.mean(u):.1f}", f"Trained ep450\n{np.mean(f):.1f}"])
b.set_ylabel("Score"); b.set_title("40 held-out seeds (paired): 2.10x, t = +4.64"); b.grid(alpha=.3, axis="y")

c = ax[2]
ak = sorted(int(k) for k in ab if k not in ("0", "final"))
c.plot(ks, means, "o-", color="#08519c", lw=2, label="Buffer 5,000 (submitted)")
c.plot(ak, [ab[str(k)]["mean"] for k in ak], "s-", color="#c51b8a", lw=2,
       label="Buffer 50,000 (interrupted at 265)")
c.axhline(base, color="#e6550d", ls="--", lw=2, label=f"Untrained baseline ({base:.0f})")
c.set_xlabel("Training episode"); c.set_ylabel("Evaluation score (mean of 5)")
c.set_title("Replay-buffer ablation, matched settings"); c.legend(fontsize=8); c.grid(alpha=.3)

plt.tight_layout(); plt.savefig("results/checkpoint_sweep.png", dpi=110); print("saved")
