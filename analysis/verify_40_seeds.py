import sys, json, random, numpy as np, torch, gymnasium as gym, ale_py
from torch import nn
gym.register_envs(ale_py)
class DQN(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(4,32,8,stride=4),nn.ReLU(),nn.Conv2d(32,64,4,stride=2),nn.ReLU(),
            nn.Conv2d(64,64,3,stride=1),nn.ReLU(),nn.Flatten(),nn.Linear(3136,512),nn.ReLU(),nn.Linear(512,n))
    def forward(self,x): return self.layers(x.float()/255.0)
def make_env():
    e=gym.make("ALE/MsPacman-v5",frameskip=1,repeat_action_probability=0.25,render_mode="rgb_array")
    e=gym.wrappers.AtariPreprocessing(e,noop_max=30,frame_skip=4,screen_size=84,terminal_on_life_loss=False,grayscale_obs=True,scale_obs=False)
    e=gym.wrappers.FrameStackObservation(e,stack_size=4,padding_type="reset")
    return gym.wrappers.TimeLimit(e,max_episode_steps=3000)
SEEDS=list(range(1000,1040))   # 40 fresh seeds, disjoint from the graded five
env=make_env(); out={}
for label,path in [a.split("=",1) for a in sys.argv[1:]]:
    ck=torch.load(path,map_location="cpu",weights_only=True)
    m=DQN(ck["n_actions"]); m.load_state_dict(ck["model"]); m.eval()
    sc=[]
    for seed in SEEDS:
        rng=random.Random(seed+10000); obs,_=env.reset(seed=seed); tot=0.0
        for step in range(3000):
            if rng.random()<0.05: a=rng.randrange(ck["n_actions"])
            else:
                with torch.no_grad(): a=int(m(torch.as_tensor(np.asarray(obs)).unsqueeze(0)).argmax())
            obs,r,e1,t1,_=env.step(a); tot+=r
            if e1 or t1: break
        sc.append(tot)
    out[label]={"scores":sc,"mean":float(np.mean(sc)),"sd":float(np.std(sc,ddof=1)),
                "se":float(np.std(sc,ddof=1)/len(sc)**0.5)}
    print(f"{label:12} n={len(sc)} mean {np.mean(sc):7.1f}  sd {np.std(sc,ddof=1):6.1f}  se {np.std(sc,ddof=1)/len(sc)**.5:5.1f}", flush=True)
    json.dump(out, open("/private/tmp/claude-501/-Users-ankitapunjani/552d5a34-90a1-4753-96db-2eb4bed786a4/scratchpad/power_rest.json","w"), indent=2)
env.close(); print("DONE")
