import sys,time,json; sys.path.insert(0,'/home/user/Generalized_Turing'); sys.path.insert(0,'/home/user/Generalized_Turing/scripts')
import numpy as np
from v4_probe import *
lat=json.load(open("results_v4/network_lattices.json"))["8"]["best"]
inner=[np.array(v) for v in lat["inner"]]; outer=[np.array(v) for v in lat["outer"]]
rho=2*np.cos(np.radians(22.5))
sym=sys.argv[1]
for eta in (0.15,0.12,0.10,0.09):
    net,s0=make(rho,10,96,eta); net.set_sym(sym)
    rng=np.random.default_rng(1)
    u=seed_network(net,inner,outer,A_SAT*0.7,0.5*A_SAT,rng)
    S=net.S; V=net.v_from_u(u); x=net.to_vec(S.to_real(V[0]),S.to_real(V[1]))
    t=time.time()
    for blk in range(3):
        x=net.relax(x,3e4)
        am_o=net.modes(x,[tuple(v) for v in outer[:4]])[:,0]
        am_i=net.modes(x,[tuple(v) for v in inner[:4]])
        print(f"{sym} η={eta} t={(blk+1)*3e4:.0e} outer|F0|(c0..c3)={['%.2e'%a for a in am_o]} inner(p1,p3) I0..I3={['(%.1e,%.1e)'%(a[1],a[3]) for a in am_i]}  [{time.time()-t:.0f}s]",flush=True)
