import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SP="/private/tmp/claude-501/-Users-ankitapunjani/552d5a34-90a1-4753-96db-2eb4bed786a4/scratchpad"
d1=json.load(open(f"{SP}/sweep_run1.json")); d2=json.load(open(f"{SP}/sweep_run2.json"))
def series(d):
    ks=sorted(int(k) for k in d if k not in ("0","final"))
    return ks,[d[str(k)]["mean"] for k in ks]
k1,m1=series(d1); k2,m2=series(d2)
base=d1["0"]["mean"]; final=d1["final"]["mean"]
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
a=ax[0]
for k in k1: a.scatter([k]*5,d1[str(k)]["scores"],s=14,color="#9ecae1",zorder=2)
a.plot(k1,m1,"o-",color="#08519c",lw=2,label="Checkpoint mean (5 games)",zorder=3)
a.axhline(base,color="#e6550d",ls="--",lw=2,label=f"Untrained baseline ({base:.0f})")
a.axhline(np.mean(m1),color="#31a354",ls=":",lw=2,label=f"Mean of 24 checkpoints ({np.mean(m1):.0f})")
a.scatter([600],[final],s=120,color="#d62728",zorder=5,marker="v",label=f"Reported final ({final:.0f})")
a.set_xlabel("Training episode"); a.set_ylabel("Evaluation score"); a.legend(fontsize=8)
a.set_title("Run 1 (buffer 5,000): every checkpoint under the frozen protocol")
a.grid(alpha=.3)
b=ax[1]
b.plot(k1,m1,"o-",color="#08519c",lw=2,label="Run 1 — buffer 5,000")
b.plot(k2,m2,"s-",color="#c51b8a",lw=2,label="Run 2 — buffer 50,000 (interrupted at 265)")
b.axhline(base,color="#e6550d",ls="--",lw=2,label=f"Untrained baseline ({base:.0f})")
b.set_xlabel("Training episode"); b.set_ylabel("Evaluation score (mean of 5 games)")
b.set_title("Replay-buffer ablation, matched settings"); b.legend(fontsize=8); b.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/checkpoint_sweep.png",dpi=110)
print("saved")
