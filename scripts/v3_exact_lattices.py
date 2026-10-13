"""정확 격자 탐색 -> results_v3/exact_lattices.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ring_sim.exactlattice import search
res = {}
for name, rho in (("oct", 2 * np.cos(np.radians(22.5))), ("ctrlL", 1.82), ("ctrlH", 1.87), ("dodec", 2 * np.cos(np.radians(15)))):
    res[name] = search(rho, cx_max=80)
    print(name, res[name]["Cx"], res[name]["N"])
json.dump(res, open("results_v3/exact_lattices.json", "w"), indent=1)
