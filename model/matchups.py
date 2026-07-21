"""Do lane matchups predict duration beyond additive champion effects?

Permutation test: strip additive per-(role, champion) effects, group the
residuals by lane pair, and compare their weighted mean-square against a null
built by shuffling residuals across games. Positive z means real matchup
structure; z near zero means the "longest matchups" table is noise.

    PYTHONPATH=. python model/matchups.py
"""

import collections, numpy as np
from pathlib import Path
from model import data as D

rows = D.load_matches(Path('data'))
ROLES = D.ROLES

# lane pairs: blue[role] vs red[role], unordered (duration is symmetric)
pairs, mains = [], []
for r in rows:
    b = {p['position']: p['champion'] for p in r['blue_picks']}
    d = {p['position']: p['champion'] for p in r['red_picks']}
    y = r['duration_s']/60
    row = []
    for role in ROLES:
        if role in b and role in d:
            row.append((role, tuple(sorted((b[role], d[role])))))
    pairs.append((row, y))
    mains.append(([ (role, c) for role in ROLES for c in (b.get(role), d.get(role)) if c ], y))

y = np.array([p[1] for p in pairs], dtype=np.float64)
print(f'{len(rows)} games, mean {y.mean():.2f} min, sd {y.std():.2f}')

# --- 1. additive per-(role,champion) effects via ridge, then residuals
keys = sorted({k for m,_ in mains for k in m})
kidx = {k:i for i,k in enumerate(keys)}
X = np.zeros((len(rows), len(keys)), dtype=np.float32)
for i,(m,_) in enumerate(mains):
    for k in m: X[i, kidx[k]] += 1
mu = y.mean()
w = np.linalg.solve(X.T@X + 300*np.eye(len(keys), dtype=np.float32), X.T@(y-mu))
resid = y - (X@w + mu)
print(f'additive model explains: R2 = {1 - resid.var()/y.var():+.4f} (in-sample)')

# --- 2. does lane matchup add structure beyond additive effects?
def group_stats(vals, minn):
    g = collections.defaultdict(list)
    for (row, _), rr in zip(pairs, vals):
        for role, pr in row: g[(role, pr)].append(rr)
    return {k:np.array(v) for k,v in g.items() if len(v) >= minn}

for minn in (20, 40):
    g = group_stats(resid, minn)
    if not g: continue
    means = np.array([v.mean() for v in g.values()])
    ns = np.array([len(v) for v in g.values()])
    obs = np.average(means**2, weights=ns)
    # permutation null: shuffle residuals across games
    rng = np.random.default_rng(0); null = []
    for _ in range(200):
        gp = group_stats(rng.permutation(resid), minn)
        m = np.array([v.mean() for v in gp.values()]); n = np.array([len(v) for v in gp.values()])
        null.append(np.average(m**2, weights=n))
    null = np.array(null)
    z = (obs - null.mean())/null.std()
    print(f'  matchups with n>={minn}: {len(g):5d}   weighted mean sq residual {obs:.4f} '
          f'vs null {null.mean():.4f}+-{null.std():.4f}   z = {z:+.2f}')

# --- 3. top/bottom matchups by raw duration
g = collections.defaultdict(list)
for row, yy in pairs:
    for role, pr in row: g[(role,pr)].append(yy)
g = {k:np.array(v) for k,v in g.items() if len(v) >= 25}
srt = sorted(g.items(), key=lambda kv: -kv[1].mean())
print(f'\nlongest lane matchups (n>=25, of {len(g)} qualifying):')
for (role,pr), v in srt[:8]:
    print(f'  {role:8} {pr[0]:14} vs {pr[1]:14} {v.mean():5.2f} min  n={len(v):4d}  (+{v.mean()-y.mean():.2f})')
print('shortest:')
for (role,pr), v in srt[-5:]:
    print(f'  {role:8} {pr[0]:14} vs {pr[1]:14} {v.mean():5.2f} min  n={len(v):4d}  ({v.mean()-y.mean():+.2f})')
