import sys,time,json; sys.path.insert(0,'/home/user/Generalized_Turing'); sys.path.insert(0,'/home/user/Generalized_Turing/scripts')
import numpy as np
from v3_stageA import DV, A_SAT
from ring_sim.model import Params, peak, solve_kappa
from ring_sim.netstate import Net

def make(case_rho, Cx, N, eta, dt=8.0):
    base=Params(Dv=DV); s0,q0=peak(base,0); k0=q0**.5
    ku,kv=solve_kappa(case_rho,1,base)
    par=Params(Dv=DV,ku=ku,kv=kv,eta=eta,etap=-eta/2)
    return Net(par,N,Cx,k0,s0,dt=dt),s0

def seed_network(net, inner, outer, a_out, a_in, rng, standing=True, phases="zero", sel_out=None, sel_in=None):
    u=np.zeros((4,net.N,net.N))
    n=len(inner)//2
    for j in (sel_out if sel_out is not None else range(n//1)):
        if j>=n//2*2 and False: continue
    outs=sel_out if sel_out is not None else range(n//2)
    ins=sel_in if sel_in is not None else range(n//2)
    for j in outs:
        ph=0.0 if phases=="zero" else rng.uniform(0,2*np.pi)
        u+=net.mode_field(outer[j],a_out,ph,0,0)*0  # placeholder
        arg=2*np.pi*(outer[j][0]*net.X+outer[j][1]*net.Y)/net.L+ph
        u+=a_out*np.cos(arg)[None]
    for m in ins:
        ph=0.0 if phases=="zero" else rng.uniform(0,2*np.pi)
        if standing: u+=net.mode_field(inner[m],a_in,ph,1,1)
        else: u+=net.mode_field(inner[m],a_in,ph,1,0)
    return u

if __name__=="__main__":
    lat=json.load(open("results_v4/network_lattices.json"))["8"]["best"]
    inner=[np.array(v) for v in lat["inner"]]; outer=[np.array(v) for v in lat["outer"]]
    rho=2*np.cos(np.radians(22.5))
    for eta in (0.18,0.15,0.12,0.10):
        net,s0=make(rho,10,96,eta)
        rng=np.random.default_rng(1)
        u=seed_network(net,inner,outer,A_SAT*0.7,0.5*A_SAT,rng)
        S=net.S; V=net.v_from_u(u)
        x=net.to_vec(S.to_real(V[0]),S.to_real(V[1]))
        t=time.time()
        for blk in range(5):
            x=net.relax(x,2e4)
            am_o=net.modes(x,[tuple(v) for v in outer[:4]])[:,0]
            am_i=net.modes(x,[tuple(v) for v in inner[:4]])
            print(f"η={eta} t={(blk+1)*2e4:.0e} outer|F0|={['%.2e'%a for a in am_o]} inner|F1|,|F3|={['(%.1e,%.1e)'%(a[1],a[3]) for a in am_i]}  [{time.time()-t:.0f}s]",flush=True)
