"""연결망 정확 격자 탐색 -> results_v4/network_lattices.json (8, 6, 10회)"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.network import search_network, n_grid
out = {}
for n in (8, 6, 10):
    s = search_network(n, cx_max=120)
    out[n] = dict(best=s[0], N=n_grid(s[0]["Cx"]), all=s)
    print(n, s[0]["Cx"], out[n]["N"])
json.dump(out, open("results_v4/network_lattices.json", "w"), indent=1)
