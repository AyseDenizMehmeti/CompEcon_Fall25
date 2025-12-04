from __future__ import annotations

import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from scipy import optimize
from scipy.stats import norm

PARAM_NAMES = ["alpha", "gamma", "rho", "sigma", "phi0"]

MOMENT_NAMES = [
    "a1 (Q coeff)",
    "a2 (CF coeff)",
    "std(I/K)",
    "std(π/K)",
    "AR1(I/K)",
    "mean Q",
    "ext_frac"
]

TARGET_MOMENTS = np.array([
    0.011,
    0.145,
    0.081,
    0.164,
    0.322,
    1.268,
    0.390
])


@dataclass
class Params:
    alpha: float = 0.6956
    gamma: float = 0.1331
    rho: float = 0.0976
    sigma: float = 0.8932
    phi0: float = 0.0

    delta: float = 0.15
    beta: float = 0.95
    phi1: float = 0.0

    nK: int = 300
    Kmin: float = 0.01
    Kmax: float = 40.0
    nA: int = 7

    Nfirms: int = 200
    T: int = 50
    burn: int = 10
    random_seed: int = 12345

    vfi_tol: float = 1e-6
    vfi_maxiter: int = 2000

    output_dir: str = "."


def tauchen(rho, sigma, m=3.0, n=7):
    s = sigma / np.sqrt(1 - rho**2)
    hi = m * s
    lo = -hi
    grid = np.linspace(lo, hi, n)
    step = (hi - lo) / (n - 1)

    P = np.zeros((n, n))
    cdf = norm.cdf

    for i in range(n):
        mu = rho * grid[i]
        for j in range(n):
            if j == 0:
                P[i, j] = cdf((grid[0] - mu + step/2) / sigma)
            elif j == n - 1:
                P[i, j] = 1 - cdf((grid[-1] - mu - step/2) / sigma)
            else:
                ub = (grid[j] - mu + step/2) / sigma
                lb = (grid[j] - mu - step/2) / sigma
                P[i, j] = cdf(ub) - cdf(lb)

    P /= P.sum(axis=1, keepdims=True)
    return grid, P


def solve_dp(params, Kgrid, grid_i, Agrid, P_A):
    nK = len(Kgrid)
    nA = len(Agrid)

    beta = params.beta
    delta = params.delta

    V = np.zeros((nK, nA))
    pol_idx = np.zeros((nK, nA), dtype=int)

    Kc = Kgrid[:, None]
    Kp = Kgrid[None, :]
    Arow = Agrid[None, :]

    print("Solving DP...")

    for it in range(params.vfi_maxiter):

        if it % 20 == 0:
            print(f"  VFI iteration {it} ...")

        EV = V.dot(P_A.T)

        invest = Kp - (1 - delta) * Kc
        pi_now = Arow * (Kc ** params.alpha)

        inv_exp = invest[:, None, :]
        inv_exp = np.repeat(inv_exp, nA, axis=1)

        pi_exp = pi_now[:, :, None]

        safeK = np.maximum(Kc, 1e-12)
        safeK_exp = safeK[:, :, None]

        adj = 0.5 * params.gamma * ((inv_exp / safeK_exp) ** 2) * safeK_exp

        ext = np.maximum(0, inv_exp - pi_exp)
        fin_cost = params.phi0 * Kc[:, :, None] * (ext > 0)

        flow = pi_exp - inv_exp - adj - fin_cost

        EV_next = EV.T[None, :, :]
        EV_next = np.transpose(EV_next, (0, 2, 1))
        EV_next = np.repeat(EV_next, nK, axis=0)
        EV_next = np.transpose(EV_next, (0, 2, 1))

        RHS = flow + beta * EV_next

        V_new = RHS.max(axis=2)
        pol_new = RHS.argmax(axis=2)

        diff = np.max(np.abs(V_new - V))
        V = V_new
        pol_idx = pol_new

        if diff < params.vfi_tol:
            print("DP converged.")
            break

    print("DP finished.")

    pol_cap = Kgrid[pol_idx]
    return {"V": V, "policy_idx": pol_idx, "policy_capital": pol_cap}


def simulate_panel(params, sol, Kgrid, Agrid, P_A, seed=None):
    if seed is None:
        seed = params.random_seed
    rng = np.random.default_rng(seed)

    N = params.Nfirms
    Ttot = params.T + params.burn
    nK = len(Kgrid)
    nA = len(Agrid)

    pol_idx = sol["policy_idx"]
    V = sol["V"]

    K_path = np.zeros((Ttot, N))
    A_idx_path = np.zeros((Ttot, N), dtype=int)
    I_K = np.zeros((Ttot, N))
    pi_K = np.zeros((Ttot, N))
    ext_amt = np.zeros((Ttot, N))
    Q = np.zeros((Ttot, N))

    print("Simulating panel...")

    Kidx = np.full(N, nK // 2)
    Aidx = np.full(N, nA // 2)

    K_path[0] = Kgrid[Kidx]
    A_idx_path[0] = Aidx

    cumP = np.cumsum(P_A, axis=1)

    for t in range(Ttot):

        if t % 10 == 0:
            print(f"  Simulation period {t}/{Ttot}")

        Kc = Kgrid[Kidx]
        Ac = Agrid[Aidx]

        nxt = pol_idx[Kidx, Aidx]
        Kp = Kgrid[nxt]

        inv = Kp - (1 - params.delta) * Kc
        prof = Ac * (Kc ** params.alpha)
        ext = np.maximum(0, inv - prof)

        safeK = np.maximum(Kc, 1e-12)
        I_K[t] = inv / safeK
        pi_K[t] = prof / safeK
        ext_amt[t] = ext

        kp1 = np.minimum(Kidx + 1, nK - 1)
        km1 = np.maximum(Kidx - 1, 0)

        Vp = V[kp1, Aidx]
        Vm = V[km1, Aidx]

        dK = Kgrid[kp1] - Kgrid[km1]
        qv = np.empty(N)

        interior = (kp1 > km1)
        qv[interior] = (Vp[interior] - Vm[interior]) / dK[interior]

        left = (Kidx == 0)
        if left.any():
            kp = 1
            km = 0
            num = V[kp, Aidx[left]] - V[km, Aidx[left]]
            den = Kgrid[kp] - Kgrid[km]
            qv[left] = num / den

        right = (Kidx == nK - 1)
        if right.any():
            kp = nK - 1
            km = nK - 2
            num = V[kp, Aidx[right]] - V[km, Aidx[right]]
            den = Kgrid[kp] - Kgrid[km]
            qv[right] = num / den

        Q[t] = qv

        if t < Ttot - 1:
            Kidx = nxt
            K_path[t+1] = Kgrid[Kidx]

            u = rng.random(N)
            rows = cumP[Aidx]
            Aidx = (u[:, None] > rows).sum(axis=1)
            A_idx_path[t+1] = Aidx

    print("Simulation finished.")

    def cut(x):
        return x[params.burn:]

    return {
        "I_over_K": cut(I_K),
        "pi_over_K": cut(pi_K),
        "K_path": cut(K_path),
        "A_path": Agrid[cut(A_idx_path)],
        "ext_amount": cut(ext_amt),
        "Q": cut(Q)
    }


def compute_moments(sim):
    I_K = sim["I_over_K"]
    pi_K = sim["pi_over_K"]
    Q = sim["Q"]
    ext = sim["ext_amount"]
    Kp = sim["K_path"]

    y = I_K[:-1].ravel()
    qlead = Q[1:].ravel()
    pit = pi_K[:-1].ravel()

    X = np.column_stack([qlead, pit, np.ones_like(y)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a1, a2 = coef[:2]

    sIK = np.std(I_K)
    spK = np.std(pi_K)

    Tsim, N = I_K.shape
    ar_list = []
    for i in range(N):
        x1 = I_K[:-1, i]
        x2 = I_K[1:, i]
        if np.std(x1) > 0 and np.std(x2) > 0:
            ar_list.append(np.corrcoef(x1, x2)[0, 1])
    ar1 = np.nanmean(ar_list) if ar_list else 0

    mq = np.nanmean(Q)

    total_inv = np.sum(I_K * Kp)
    total_ext = np.sum(ext)
    ext_frac = total_ext / total_inv if total_inv > 0 else 0

    return np.array([a1, a2, sIK, spK, ar1, mq, ext_frac])


def estimate_single_run(theta, base, W=None):
    p = Params(
        alpha=float(theta[0]),
        gamma=float(theta[1]),
        rho=float(theta[2]),
        sigma=float(theta[3]),
        phi0=float(theta[4]),
        nK=base.nK,
        Kmin=base.Kmin,
        Kmax=base.Kmax,
        nA=base.nA,
        Nfirms=base.Nfirms,
        T=base.T,
        burn=base.burn,
        random_seed=base.random_seed
    )

    Alog, P_A = tauchen(p.rho, p.sigma, n=p.nA)
    Agrid = np.exp(Alog)
    Kgrid = np.linspace(p.Kmin, p.Kmax, p.nK)

    sol = solve_dp(p, Kgrid, np.linspace(0, 8, 80), Agrid, P_A)
    sim = simulate_panel(p, sol, Kgrid, Agrid, P_A)
    mom = compute_moments(sim)

    diff = mom - TARGET_MOMENTS
    if W is None:
        W = np.eye(len(mom))

    dist = diff @ W @ diff
    return mom, float(dist)


def compute_optimal_weighting_matrix(theta_hat, base, n_sim=50):
    M = []
    for i in range(n_sim):
        p = Params(
            alpha=theta_hat[0],
            gamma=theta_hat[1],
            rho=theta_hat[2],
            sigma=theta_hat[3],
            phi0=theta_hat[4],
            nK=base.nK,
            Kmin=base.Kmin,
            Kmax=base.Kmax,
            nA=base.nA,
            Nfirms=base.Nfirms,
            T=base.T,
            burn=base.burn,
            random_seed=base.random_seed + i
        )
        Alog, P_A = tauchen(p.rho, p.sigma, n=p.nA)
        Agrid = np.exp(Alog)
        Kgrid = np.linspace(p.Kmin, p.Kmax, p.nK)

        sol = solve_dp(p, Kgrid, np.linspace(0, 8, 80), Agrid, P_A)
        sim = simulate_panel(p, sol, Kgrid, Agrid, P_A)
        M.append(compute_moments(sim))

    M = np.array(M)
    Omega = np.cov(M, rowvar=False)
    Omega += 1e-8 * np.eye(Omega.shape[0])
    return np.linalg.pinv(Omega)


def compute_standard_errors(theta, base, Wopt, n_sim=50, base_eps=1e-5):
    k = len(theta)
    m = len(TARGET_MOMENTS)
    G = np.zeros((m, k))

    for j in range(k):
        h = base_eps * max(1e-6, abs(theta[j]))
        t_plus = theta.copy()
        t_minus = theta.copy()
        t_plus[j] += h
        t_minus[j] -= h

        mp, _ = estimate_single_run(t_plus, base, Wopt)
        mm, _ = estimate_single_run(t_minus, base, Wopt)
        G[:, j] = (mp - mm) / (2 * h)

    S = n_sim
    GtWG = G.T @ Wopt @ G
    inv = np.linalg.pinv(GtWG)
    Qv = (1 + 1/S) * inv
    se = np.sqrt(np.maximum(0, np.diag(Qv)))
    return se


def create_latex_tables(theta1, theta2, se, mom1, mom2, outdir: Path):
    outdir.mkdir(exist_ok=True, parents=True)

    def safe(s):
        s = s.replace("π", r"$\pi$")
        s = s.replace("_", r"\_")
        s = s.replace("\n", "")   # remove accidental newlines
        s = s.replace("\r", "")   # remove carriage returns
        return s

    pfile = outdir / "parameter_estimates.tex"
    with open(pfile, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Parameter & Identity-W & Efficient-W & Std. Error & t-Stat \\\\\n")
        f.write("\\midrule\n")
        for i, nm in enumerate(PARAM_NAMES):
            t1 = theta1[i]
            t2 = theta2[i]
            se_i = se[i]
            tstat = t2 / se_i if se_i != 0 else 0
            f.write(f"{nm} & {t1:.6f} & {t2:.6f} & {se_i:.6f} & {tstat:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    mfile = outdir / "moments_comparison.tex"
    with open(mfile, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Moment & Target & Identity-W & Efficient-W & Diff \\\\\n")
        f.write("\\midrule\n")
        for i, nm in enumerate(MOMENT_NAMES):
            nm2 = safe(nm)
            t = TARGET_MOMENTS[i]
            m1 = mom1[i]
            m2 = mom2[i]
            diff = m2 - t
            f.write(f"{nm2} & {t:.4f} & {m1:.4f} & {m2:.4f} & {diff:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    pd.DataFrame({
        "Parameter": PARAM_NAMES,
        "Identity_W": theta1,
        "Efficient_W": theta2,
        "Std_Error": se,
        "t_Stat": theta2 / (se + 1e-12)
    }).to_csv(outdir / "parameter_summary.csv", index=False)

    pd.DataFrame({
        "Moment": MOMENT_NAMES,
        "Target": TARGET_MOMENTS,
        "Identity_W": mom1,
        "Efficient_W": mom2,
        "Difference": mom2 - TARGET_MOMENTS
    }).to_csv(outdir / "moments_summary.csv", index=False)

    return pfile, mfile


def run_two_step_smm(base):
    print("Starting two-step SMM...")

    out = Path(base.output_dir)
    out.mkdir(exist_ok=True)

    print("Step 1: Identity-weighted SMM...")

    theta0 = np.array([0.6956, 0.1331, 0.0976, 0.8932, 0.0])
    bounds = [
        (0.1, 1.0),
        (0.01, 2.0),
        (0.01, 0.99),
        (0.1, 2.0),
        (0.0, 0.02)
    ]

    def obj_identity(th):
        return estimate_single_run(th, base, np.eye(len(TARGET_MOMENTS)))[1]

    r1 = optimize.minimize(
        obj_identity, theta0, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 20, "ftol": 1e-3}
    )
    t1 = r1.x
    m1, _ = estimate_single_run(t1, base, np.eye(len(TARGET_MOMENTS)))

    print("Step 2: Computing optimal weighting matrix...")
    Wopt = compute_optimal_weighting_matrix(t1, base, n_sim=20)

    print("Step 3: Efficient SMM...")
    def obj_opt(th):
        return estimate_single_run(th, base, Wopt)[1]

    r2 = optimize.minimize(
        obj_opt, t1, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 20, "ftol": 1e-3}
    )
    t2 = r2.x
    m2, _ = estimate_single_run(t2, base, Wopt)

    print("Computing standard errors...")
    se = compute_standard_errors(t2, base, Wopt, n_sim=20)

    print("Creating LaTeX tables...")
    create_latex_tables(t1, t2, se, m1, m2, out)

    print("SMM estimation completed.")

    return {
        "theta_identity": t1,
        "theta_efficient": t2,
        "standard_errors": se,
        "moments_identity": m1,
        "moments_efficient": m2,
        "W_optimal": Wopt,
        "output_dir": out
    }


def test_quick_run():
    p = Params()
    p.nK = 100
    p.Nfirms = 100
    p.T = 20
    p.burn = 5
    p.output_dir = "."

    th = np.array([p.alpha, p.gamma, p.rho, p.sigma, p.phi0])
    p.alpha, p.gamma, p.rho, p.sigma, p.phi0 = th

    Alog, P_A = tauchen(p.rho, p.sigma, n=p.nA)
    Agrid = np.exp(Alog)
    Kgrid = np.linspace(p.Kmin, p.Kmax, p.nK)

    sol = solve_dp(p, Kgrid, np.linspace(0, 8, 80), Agrid, P_A)
    sim = simulate_panel(p, sol, Kgrid, Agrid, P_A)
    mom = compute_moments(sim)

    return mom, sol, sim


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        mom, sol, sim = test_quick_run()
        print("Quick test moments:")
        for name, val in zip(MOMENT_NAMES, mom):
            print(f"{name}: {val}")

    elif len(sys.argv) > 1 and sys.argv[1] == "--full":
        base = Params()
        outdir = Path(base.output_dir)

        print("Using existing LaTeX tables. No regeneration will occur.")
        print("Only rebuilding master PDF wrapper...")

        # DO NOT regenerate the .tex tables.
        # Only rebuild the main LaTeX wrapper.
        tex_main = outdir / "ProblemSet8_Ayse.tex"
        with open(tex_main, "w") as f:
            f.write("\\documentclass[12pt]{article}\n")
            f.write("\\usepackage{amsmath, amssymb, booktabs}\n")
            f.write("\\usepackage[margin=1in]{geometry}\n")
            f.write("\\begin{document}\n")
            f.write("\\section{Parameter Estimates}\n\\input{parameter_estimates.tex}\n")
            f.write("\\section{Moment Comparison}\n\\input{moments_comparison.tex}\n")
            f.write("\\end{document}\n")

        print("LaTeX wrapper created (parameter and moment tables preserved).")

    else:
        print("Usage: python ps8_Ayse.py [--test|--full]")






