import sys, json, math, numpy as np, trimesh
body = max(trimesh.load(sys.argv[1]).split(only_watertight=False), key=lambda p: abs(p.volume)); C = json.load(open(sys.argv[2]))
for nm, key, r in (("T lead", "tubeT", 3.0), ("MU10 lead", "tubeUM", 4.1)):
    P = np.array(C[key]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; bad = []
    for i in range(0, len(P), 2):
        if s[i] > s[-1] - 8.0 or s[i] < 3.0: continue
        a = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]; a /= np.linalg.norm(a); u = np.cross(a, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0]); v = np.cross(a, u)
        ring = [P[i] + (r + 0.6) * (math.cos(t) * u + math.sin(t) * v) for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        f = body.contains(ring).mean()
        if f < 1.0: bad.append((round(float(f), 2), np.round(P[i], 1).tolist()))
    print(f"{nm}: {len(bad)} stations with the 0.6 mm wall ring not fully in material {bad[:6]}")
