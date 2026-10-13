import sys,json,os; sys.path.insert(0,'/home/user/Generalized_Turing'); sys.path.insert(0,'/home/user/Generalized_Turing/scripts')
import numpy as np
from v4_branch import lattice
from v4_probe import make
eta=float(sys.argv[1]); inner,outer,Cx,N=lattice(10)
net,s0=make(2*np.cos(np.pi/10),Cx,N,eta); net.set_sym("Mx+My+L")
x=np.load(f"runs_v4/n10s_b_relaxed_eta{eta}.npy")
for blk in range(4):
    mo=net.modes(x,[tuple(v) for v in outer[:5]])[:,0]; mi=net.modes(x,[tuple(v) for v in inner[:5]])
    print(f"t+{blk*3e4:.0e}: outer={['%.2e'%a for a in mo]} inner p1={['%.2e'%a[1] for a in mi]}",flush=True)
    x=net.relax(x,3e4)
np.save(f"runs_v4/n10s_drift_eta{eta}.npy",x)
