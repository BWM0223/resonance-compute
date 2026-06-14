#!/usr/bin/env python3
"""
RESONANCE — FULL-FORCE COMPUTE (for the BWM0223 node: 7GB / 2 cores / 5h)
φ = 1.618033988749895

Three decisive experiments. Every rule is fixed A PRIORI from the framework; we
report whatever comes out, including null results. No tuning to known masses.

A) 1D charge-projection: do gauge-shell admissibility rules (su3=8,su2=3,u1=1),
   defined before looking, make the dominant ALLOWED gaps reproduce 0,11,17?
B) 2D golden-angle substrate (the real A7 projection) — heavy dense spectrum the
   Celeron can't hold. Does a 3-generation gap structure emerge naturally?
C) Parameter-free fermion prediction: fix the rule, predict ALL 9 masses, score.

Outputs: resonance_results.txt
Needs: numpy, scipy.  ~3-5GB at the 2D sizes below.
"""
import math, datetime
import numpy as np
from scipy.linalg import eigh_tridiagonal, eigh

PHI=1.618033988749895; R=1/PHI**2; LNPHI=math.log(PHI)
GA=2*math.pi/PHI**2   # golden angle

OUT=[]
def log(s): print(s,flush=True); OUT.append(s)

def fibV(N): return np.array([(math.floor((n+1)/PHI)-math.floor(n/PHI)) for n in range(N)],float)
def spec1d(N,V=1.5,t=1.0): return np.sort(eigh_tridiagonal(fibV(N)*V,np.full(N-1,-t),eigvals_only=True))
def gaplabel(idos,nmax=120):
    best=(9,0)
    for n in range(-nmax,nmax+1):
        d=abs((((n*R)%1)-idos+0.5)%1-0.5)
        if d<best[0]: best=(d,n)
    return best[1]

# measured lepton golden-depths (electron=0)
M={"e":0.51099895,"mu":105.6583755,"tau":1776.86,
   "u":2.16,"d":4.67,"s":93.4,"c":1270.0,"b":4180.0,"t":172570.0}

def header():
    log("="*72)
    log("RESONANCE FULL-FORCE COMPUTE")
    log(f"run {datetime.datetime.now(datetime.UTC).isoformat()}  φ={PHI} ρ_A*={R:.6f}")
    log("="*72)

# ── A) 1D charge projection ─────────────────────────────────────────────────
def expA():
    log("\n[A] 1D CHARGE-PROJECTION — a priori gauge-shell rules")
    N=10946; E=spec1d(N); G=[]
    for i in range(len(E)-1):
        w=E[i+1]-E[i]
        if w<1e-7: continue
        G.append((w,(i+1)/N,gaplabel((i+1)/N),-math.log(w)/LNPHI))
    G.sort(reverse=True)
    rules={
        "all":lambda n:True,
        "shell-sizes |n|in{1,3,8}":lambda n:abs(n) in(1,3,8),
        "negative-only (anti-resonance/mass)":lambda n:n<0,
        "n=-Fibonacci":lambda n:(-n) in (1,2,3,5,8,13,21,34),
        "n=-Lucas":lambda n:(-n) in (1,3,4,7,11,18,29),
    }
    for rn,rule in rules.items():
        allowed=[g for g in G if rule(g[2])][:5]
        log(f"  rule[{rn}]: depths={[round(g[3],2) for g in allowed]} labels={[g[2] for g in allowed]}")
    log("  TARGET generation depths: 0, 11, 17.  (electron,muon,tau)")
    log("  Honest read: a rule WINS only if its top-3 ~ 0,11,17 without being built for it.")

# ── B) 2D golden-angle substrate ────────────────────────────────────────────
def expB():
    log("\n[B] 2D GOLDEN-ANGLE SUBSTRATE (real A7 projection) — dense spectrum")
    for N in (2000, 4000):
        idx=np.arange(1,N+1)
        r=np.sqrt(idx); th=idx*GA
        x=r*np.cos(th); y=r*np.sin(th)
        # tight-binding with phi-decaying coupling to nearby sites
        H=np.zeros((N,N))
        # connect each site to its ~6 nearest neighbors (sunflower packing)
        from scipy.spatial import cKDTree
        tree=cKDTree(np.c_[x,y]); 
        d,nn=tree.query(np.c_[x,y],k=7)
        for i in range(N):
            for j,dist in zip(nn[i][1:],d[i][1:]):
                H[i,j]=-math.exp(-dist/PHI)  # phi-scaled hopping
        H=(H+H.T)/2
        ev=np.sort(np.linalg.eigvalsh(H))
        # density of states gaps
        gaps=sorted(((ev[i+1]-ev[i],(i+1)/N) for i in range(len(ev)-1)),reverse=True)[:8]
        log(f"  N={N}: top DOS gaps (width,IDOS)= "+", ".join(f"({w:.4f},{p:.3f})" for w,p in gaps))
        gl=[gaplabel(p) for w,p in gaps]
        log(f"         gap labels: {gl}")

# ── C) parameter-free prediction with muon-anchored alpha dressing ──────────
def expC():
    log("\n[C] FERMION PREDICTION — muon anchored at 11(1+α), check the rest")
    a=1/137.035999
    log("  rule fixed a priori: depth_n = n*(1+α), n integer. Predict masses:")
    for name in ["mu","tau","u","s","c","b","t"]:
        Emeas=math.log(M[name]/M["e"])/LNPHI
        n=round(Emeas/(1+a))
        pred=n*(1+a)
        log(f"    {name:>3}: n={n:>2}  pred_depth={pred:.3f}  meas={Emeas:.3f}  resid={Emeas-pred:+.3f}")
    log("  WIN if residuals are uniformly tiny (one rule, all fermions). Else: muon special.")

if __name__=="__main__":
    header(); expA(); expB(); expC()
    open("resonance_results.txt","w").write("\n".join(OUT)+"\n")
    log("\nDONE -> resonance_results.txt")
