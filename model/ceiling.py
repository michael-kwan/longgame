"""How much of duration is predictable at draft time, from everything we have?"""
import collections, numpy as np
from pathlib import Path
from model import data as D

rows = D.load_matches(Path('data'))
n = len(rows); y = np.array([r['duration_s']/60 for r in rows])
v = D.build_vocab(rows); enc = D.encode(rows, v); C = len(v)

# champions
Xc = np.zeros((n, C)); rr = np.repeat(np.arange(n), 10)
Xc[rr, enc['picks'].reshape(n,10).ravel()] += 1; Xc[:,0] = 0
# tier one-hot, queue
Xt = np.zeros((n, len(D.TIERS))); Xt[np.arange(n), enc['tier']] = 1
Xq = enc['queue'].reshape(-1,1).astype(float)
# player leave-one-out mean
tot = collections.defaultdict(float); cnt = collections.Counter(); parts=[]
for i, r in enumerate(rows):
    ps=[p['puuid'] for p in r['blue_picks']+r['red_picks'] if p['puuid']]; parts.append(ps)
    for p in ps: tot[p]+=y[i]; cnt[p]+=1
grand=y.mean(); Xp=np.zeros((n,1))
for i in range(n):
    vals=[(tot[p]-y[i])/(cnt[p]-1) for p in parts[i] if cnt[p]-1>=2]
    Xp[i,0]=np.mean(vals) if len(vals)>=3 else grand

blocks = {'champions':Xc, 'tier':Xt, 'queue':Xq, 'player history':Xp}
rng=np.random.default_rng(0); perm=rng.permutation(n); folds=np.array_split(perm,10)

def cv(X, lam):
    r2=[]
    for f in folds:
        tr=np.setdiff1d(perm,f); mu=y[tr].mean()
        w=np.linalg.solve(X[tr].T@X[tr]+lam*np.eye(X.shape[1]), X[tr].T@(y[tr]-mu))
        p=X[f]@w+mu
        r2.append(1-((p-y[f])**2).sum()/((y[f]-y[f].mean())**2).sum())
    return np.array(r2)

print(f'{n} games, duration sd {y.std():.2f} min\n')
print('individually (10-fold CV, best lambda):')
for name, X in blocks.items():
    best=max((cv(X,l).mean(), l, cv(X,l)) for l in (1,10,100,300,1000,3000))
    print(f'  {name:16} R2 = {best[0]:+.4f} +- {best[2].std(ddof=1)/np.sqrt(10):.4f}')
Xall=np.hstack(list(blocks.values()))
best=max((cv(Xall,l).mean(), l, cv(Xall,l)) for l in (10,100,300,1000,3000))
print(f'\n  {"EVERYTHING":16} R2 = {best[0]:+.4f} +- {best[2].std(ddof=1)/np.sqrt(10):.4f}  (lambda={best[1]})')
print(f'\n  => predictable sd: {y.std()*np.sqrt(max(best[0],0)):.2f} min of {y.std():.2f} min total')

# classification framing: is "will this be a long game" easier than E[duration]?
for thresh in (35, 40):
    lab=(y>thresh).astype(float); base=lab.mean()
    r2=[]; aucs=[]
    for f in folds:
        tr=np.setdiff1d(perm,f); mu=lab[tr].mean()
        w=np.linalg.solve(Xall[tr].T@Xall[tr]+300*np.eye(Xall.shape[1]), Xall[tr].T@(lab[tr]-mu))
        p=Xall[f]@w+mu
        pos=p[lab[f]==1]; neg=p[lab[f]==0]
        aucs.append((pos[:,None]>neg[None,:]).mean() + 0.5*(pos[:,None]==neg[None,:]).mean())
    print(f'  P(duration > {thresh} min): base rate {base:.1%}, AUC = {np.mean(aucs):.4f}')
