import numpy as np, mg
def run(path, side=1, step=0.01, f=None):
    c=np.array(path,float); o=np.r_[c[0],c[:-1]]; h=np.maximum(o,c); l=np.minimum(o,c)
    f=np.zeros_like(c) if f is None else f
    return mg.episode(o,h,l,c,f,0,step,side,0.0,0.0,0.0,0.0,0.0,False,0.32)
# up immediately: tp at +1% from base, no costs -> pnl = W0*0.01
r=run([100,100.5,101.2]); print('tp', r[:4], mg.W[0]*0.01)
# down 2 levels then back up
r=run([100,99.5,98.9,97.9,99,101]); print('dca tp', r[:4])
# crash to stop
r=run([100]+list(np.linspace(99,90,30))); print('stop', r[:4])
# short mirror
r=run([100,99.5,98.8],side=-1); print('short tp', r[:4])
# funding long pays
f=np.zeros(5); f[2]=0.001
r=run([100,100,100,100,100],f=f); print('fund',r[4], -0.001*mg.W[0])
