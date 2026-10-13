import sys,time,json; sys.path.insert(0,'/home/user/Generalized_Turing'); sys.path.insert(0,'/home/user/Generalized_Turing/scripts')
import numpy as np
from v4_probe import *
eta=float(sys.argv[1]); sym=sys.argv[2] if len(sys.argv)>2 else "D4L"
lat=json.load(open("results_v4/network_lattices.json"))["8"]["best"]
inner=[np.array(v) for v in lat["inner"]]; outer=[np.array(v) for v in lat["outer"]]
rho=2*np.cos(np.radians(22.5))
net,s0=make(rho,10,96,eta); net.set_sym(sym)
rng=np.random.default_rng(1)
u=seed_network(net,inner,outer,A_SAT*0.7,0.5*A_SAT,rng)
S=net.S; V=net.v_from_u(u); x=net.to_vec(S.to_real(V[0]),S.to_real(V[1]))
x=net.relax(x,6e4)
r0=np.abs(net.residual(x)).max(); print("after relax residual",r0,flush=True)
t=time.time(); x,r=net.newton(x,f_tol=1e-10,maxiter=15,inner_maxiter=30); print("NK residual",r,"time %.0fs"%(time.time()-t),flush=True)
np.save(f"runs_v4/n8_eta{eta}_{sym}.npy",x)
net.sym=None
t=time.time(); lam,w,V=net.eigenvalues(x,k=10,tol=1e-5); print("full-space Arnoldi λ/σ0:",[round(float(l),4) for l in lam],"time %.0fs"%(time.time()-t),flush=True)
