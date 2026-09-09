import sys, json, random, glob, os, numpy as np, torch, gymnasium as gym, ale_py
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
def ev(m,n):
    out=[]
    for seed in [101,202,303,404,505]:
        rng=random.Random(seed+10000); obs,_=env.reset(seed=seed); tot=0.0
        for step in range(3000):
            if rng.random()<0.05: a=rng.randrange(n)
            else:
                with torch.no_grad(): a=int(m(torch.as_tensor(np.asarray(obs)).unsqueeze(0)).argmax())
            obs,r,e1,t1,_=env.step(a); tot+=r
            if e1 or t1: break
        out.append(tot)
    return out
env=make_env(); res={}
rundir, outpath = sys.argv[1], sys.argv[2]
paths=[(0,f"{rundir}/untrained.pt")]
paths+= [(int(os.path.basename(p)[8:12]), p) for p in sorted(glob.glob(f"{rundir}/episode_*.pt"))]
if os.path.exists(f"{rundir}/trained.pt"): paths.append(("final",f"{rundir}/trained.pt"))
for ep,p in paths:
    ck=torch.load(p,map_location="cpu",weights_only=True)
    m=DQN(ck["n_actions"]); m.load_state_dict(ck["model"]); m.eval()
    s=ev(m,ck["n_actions"]); res[str(ep)]={"scores":s,"mean":float(np.mean(s))}
    print(ep, s, round(float(np.mean(s)),1), flush=True)
    json.dump(res, open(outpath,"w"), indent=2)
env.close(); print("DONE")
