#!/usr/bin/env python3
"""
RESONANCE — Heavy Compute Job (for the GitHub Actions free runner / Colab)
φ = 1.618033988749895

The honest test the Celeron can't run: instead of FITTING (m,n) gap labels to each
fermion (which density makes trivial), ENUMERATE the actual dominant spectral gaps
of a large Fibonacci Hamiltonian and read off their TRUE labels in order. If the
physically-selected gaps (widest = most resonant/gravitational) carry the generation
structure WITHOUT fitting, that's real. If not, honest negative.

Outputs: resonance_results.txt  (committed back by the workflow)
Needs: numpy, scipy. RAM: ~3GB at N=6765. Time: minutes.
"""
import math, sys, datetime
import numpy as np

PHI = 1.618033988749895
R = 1/PHI**2   # rotation number / balance constant
LNPHI = math.log(PHI)

def fib_potential(N):
    return np.array([(math.floor((n+1)/PHI)-math.floor(n/PHI)) for n in range(N)],
                    dtype=np.float64)

def spectrum(N, V=1.5, t=1.0):
    """Full spectrum of the Fibonacci Hamiltonian via dense eigvalsh."""
    Vn = fib_potential(N)
    # symmetric tridiagonal: diagonal Vn*V, off-diagonal -t
    d = Vn*V
    e = np.full(N-1, -t)
    from scipy.linalg import eigh_tridiagonal
    return np.sort(eigh_tridiagonal(d, e, eigvals_only=True))

def true_gap_label(idos, mmax=40, nmax=40):
    """The gap-labeling integer n with {m+n*R}=idos. Returns (n, residual).
    This is READ from the real gap, not fitted to a target mass."""
    best=(1e9,0,0)
    for n in range(-nmax,nmax+1):
        val=(n*R)%1
        # nearest m is automatic via mod; residual:
        d=abs(((val)-idos+0.5)%1-0.5)
        if d<best[0]: best=(d,n)
    return best[1], best[0]

def main():
    out=[]
    def log(s):
        print(s, flush=True); out.append(s)

    log("="*70)
    log("RESONANCE HEAVY COMPUTE — dominant-gap enumeration (no fitting)")
    log(f"run: {datetime.datetime.utcnow().isoformat()}Z   φ={PHI}  ρ_A*=1/φ²={R:.6f}")
    log("="*70)

    for N in (1597, 6765):
        log(f"\n--- Fibonacci Hamiltonian N={N} ---")
        E = spectrum(N)
        gaps=[]
        for i in range(len(E)-1):
            w=E[i+1]-E[i]
            if w<1e-7: continue
            idos=(i+1)/N
            n,resid=true_gap_label(idos)
            depth=-math.log(w)/LNPHI
            gaps.append((w,idos,n,resid,depth))
        gaps.sort(reverse=True)  # widest first = most resonant/gravitational
        log(f"top 12 DOMINANT gaps (widest), with their TRUE gap labels:")
        log(f"{'width':>9}{'IDOS':>9}{'label n':>8}{'resid':>9}{'depth':>9}")
        for w,idos,n,resid,depth in gaps[:12]:
            log(f"{w:>9.4f}{idos:>9.4f}{n:>8d}{resid:>9.4f}{depth:>9.3f}")
        # The honest question: do the dominant gaps' labels include 0,-1,-5 (the
        # PROOF_FINAL generation labels) AS THE DOMINANT ones, or are 0,-1,-5 just
        # buried among many equally-good labels?
        dom_labels=[g[2] for g in gaps[:12]]
        log(f"dominant-gap labels in width order: {dom_labels}")
        log(f"PROOF_FINAL needs generations at labels n = 0, -1, -5.")
        log(f"  n=0  among top-12 dominant? {0 in dom_labels}")
        log(f"  n=-1 among top-12 dominant? {-1 in dom_labels}")
        log(f"  n=-5 among top-12 dominant? {-5 in dom_labels}")
        log(f"  VERDICT: if -5 is NOT a dominant gap, the tau label was fitted, not selected.")

    # write results
    with open("resonance_results.txt","w") as f:
        f.write("\n".join(out)+"\n")
    log("\nwrote resonance_results.txt")

if __name__=="__main__":
    main()
