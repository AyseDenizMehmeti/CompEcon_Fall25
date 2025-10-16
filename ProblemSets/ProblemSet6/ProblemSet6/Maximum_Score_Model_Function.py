#!/usr/bin/env python3
"""
Maximum_Score_Model_Function.py

True Maximum Score Estimator (random-search) with bootstrap standard errors.

This script is computationally heavy if you use large iteration counts.
Defaults are conservative so you can test quickly; increase iterations for production.

Usage (terminal):
  python true_maximum_score_estimator.py --input data/MergedTeams_Filtered.csv --outdir results \
    --n-iter 20000 --sample-size 20000 --n-boot 40 --n-boot-iter 5000

Or import in notebook:
  from true_maximum_score_estimator import true_maximum_score_estimator
  res = true_maximum_score_estimator(input_csv="data/MergedTeams_Filtered.csv",
                                     outdir="results",
                                     n_iter=20000,
                                     sample_size=20000,
                                     n_boot=40,
                                     n_boot_iter=5000,
                                     random_state=42,
                                     show=True)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import time

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda it, **kw: it

# -------------------------
# Core functions
# -------------------------
def score_function(beta, X, y):
    """Maximum score objective: fraction of correctly signed predictions."""
    pred = X @ beta
    pred_signs = np.sign(pred)
    correct_pos = np.sum((y == 1) & (pred_signs > 0))
    correct_neg = np.sum((y == 0) & (pred_signs < 0))
    return (correct_pos + correct_neg) / len(y)


def maximum_score_estimator_true(X, y, n_iter=20000, random_seed=42, search_scale=2.0):
    """
    Random search for the true Maximum Score Estimator.
    Returns best_beta (numpy array) and best_score (float).
    """
    rng = np.random.RandomState(random_seed)
    n_features = X.shape[1]
    best_beta = None
    best_score = -np.inf

    for _ in tqdm(range(n_iter), desc="🔍 Searching β"):
        beta = rng.uniform(-search_scale, search_scale, size=n_features)
        s = score_function(beta, X, y)
        if s > best_score:
            best_score = s
            best_beta = beta.copy()

    return np.array(best_beta), float(best_score)


def bootstrap_standard_errors(X, y, n_boot=40, n_iter=5000, random_state=42):
    """
    Bootstrap standard errors: for each bootstrap sample, re-run maximum_score_estimator_true
    with n_iter iterations (can be smaller than main search).
    Returns se (numpy array) of shape (n_features,).
    """
    rng = np.random.RandomState(random_state)
    coefs = []
    n = len(y)
    for b in tqdm(range(n_boot), desc="📊 Bootstrapping"):
        idx = rng.choice(n, size=n, replace=True)
        Xb = X[idx]
        yb = y[idx]
        beta_hat, _ = maximum_score_estimator_true(Xb, yb, n_iter=n_iter, random_seed=random_state + b)
        coefs.append(beta_hat)
    coefs = np.vstack(coefs)
    se = np.std(coefs, axis=0, ddof=1)
    return se


# -------------------------
# Wrapper / CLI callable
# -------------------------
def true_maximum_score_estimator(input_csv,
                                 outdir=None,
                                 n_iter=20000,
                                 sample_size=20000,
                                 n_boot=40,
                                 n_boot_iter=5000,
                                 random_state=42,
                                 show=False):
    """
    Run the true Maximum Score Estimator workflow.

    Parameters
    ----------
    input_csv : str or Path
        Path to filtered pairs CSV or merged dataset (the function expects a merged dataset)
    outdir : str or Path, optional
        Directory to save outputs (defaults to same folder as input)
    n_iter : int
        Random-search iterations for the main estimator
    sample_size : int
        Number of pair observations to sample for estimation (pos+neg combined)
    n_boot : int
        Number of bootstrap replications for SEs
    n_boot_iter : int
        Iterations per bootstrap run (smaller than n_iter to save time)
    random_state : int
        Random seed
    show : bool
        If True, plt.show() after plotting (useful in notebooks)

    Returns
    -------
    results_df, beta_hat, best_score
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv.resolve()}")

    if outdir is None:
        outdir = input_csv.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"📂 Loading data from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"🔢 Rows: {len(df):,}, Columns: {df.shape[1]}")

    # --- Build pos/neg pairs from merged dataset (same logic as before) ---
    print("🔧 Building positive (teammate) pairs...")
    pos_pairs = []
    for team_id, team_data in tqdm(df.groupby('TeamId'), desc="Teams"):
        if team_data.shape[0] < 2:
            continue
        members = team_data[['UserId', 'Tier', 'HighestRanking', 'Country', 'OrganizationId']].dropna(subset=['Tier', 'HighestRanking'])
        if members.shape[0] < 2:
            continue
        min_rank = members['HighestRanking'].min()
        candidates = members[members['HighestRanking'] == min_rank]
        if len(candidates) > 1:
            top_member = candidates.sample(1, random_state=random_state).iloc[0]
        else:
            top_member = candidates.iloc[0]
        others = members[members['UserId'] != top_member['UserId']]
        for _, other in others.iterrows():
            pos_pairs.append([
                float(abs(top_member['Tier'] - other['Tier'])),
                float(abs(top_member['HighestRanking'] - other['HighestRanking'])),
                int(pd.notna(top_member['Country']) and pd.notna(other['Country']) and top_member['Country'] == other['Country']),
                int(pd.notna(top_member['OrganizationId']) and pd.notna(other['OrganizationId']) and top_member['OrganizationId'] == other['OrganizationId']),
                1
            ])
    pos_df = pd.DataFrame(pos_pairs, columns=['Tier_Diff', 'Rank_Diff', 'Same_Country', 'Same_Org', 'Match'])
    print(f"✅ Built {len(pos_df):,} positive pairs.")

    print("🔧 Building negative (non-match) pairs...")
    users = df[['UserId', 'Tier', 'HighestRanking', 'Country', 'OrganizationId']].dropna(subset=['Tier', 'HighestRanking']).drop_duplicates(subset=['UserId'])
    neg_pairs = []
    rng = np.random.RandomState(random_state)
    for _ in tqdm(range(len(pos_df)), desc="NegPairs"):
        sample_df = users.sample(2, replace=False, random_state=rng.randint(0, 2**31 - 1))
        a = sample_df.iloc[0]
        b = sample_df.iloc[1]

        neg_pairs.append([
            float(abs(a['Tier'] - b['Tier'])),
            float(abs(a['HighestRanking'] - b['HighestRanking'])),
            int(pd.notna(a['Country']) and pd.notna(b['Country']) and a['Country'] == b['Country']),
            int(pd.notna(a['OrganizationId']) and pd.notna(b['OrganizationId']) and a['OrganizationId'] == b['OrganizationId']),
            0
        ])
    neg_df = pd.DataFrame(neg_pairs, columns=['Tier_Diff', 'Rank_Diff', 'Same_Country', 'Same_Org', 'Match'])
    print(f"✅ Built {len(neg_df):,} negative pairs.")

    # Combine and sample
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    n_use = min(sample_size, len(combined))
    combined_sample = combined.sample(n=n_use, random_state=random_state)
    print(f"📊 Using {n_use:,} observations for estimation (of {len(combined):,} total pairs).")

    X = combined_sample[['Tier_Diff', 'Rank_Diff', 'Same_Country', 'Same_Org']].values
    y = combined_sample['Match'].values

    # Estimate true maximum score
    print(f"🧠 Running TRUE Maximum Score Estimator ({n_iter:,} iterations)...")
    beta_hat, best_score = maximum_score_estimator_true(X, y, n_iter=n_iter, random_seed=random_state)
    print(f"✅ Best Score: {best_score:.4f}")
    print(f"✅ Estimated Beta: {beta_hat}")

    # Model fit metrics
    pred_signs = np.sign(X @ beta_hat)
    correct = np.sum((y == 1) & (pred_signs > 0)) + np.sum((y == 0) & (pred_signs < 0))
    accuracy = correct / len(y)
    pseudo_r2 = 1 - ((1 - accuracy) / 0.5)

    print(f"🎯 Model Accuracy: {accuracy:.4f}")
    print(f"📈 Pseudo R² (relative to random): {pseudo_r2:.4f}")

    # Bootstrap SEs (may be slow)
    print(f"📊 Bootstrapping standard errors: n_boot={n_boot}, n_iter={n_boot_iter}")
    se_hat = bootstrap_standard_errors(X, y, n_boot=n_boot, n_iter=n_boot_iter, random_state=random_state)
    print(f"✅ Bootstrapped SEs: {se_hat}")

    # Save results table
    results = pd.DataFrame({
        'Covariate': ['Tier_Diff', 'Rank_Diff', 'Same_Country', 'Same_Org'],
        'Beta': np.round(beta_hat, 6),
        'Std_Error': np.round(se_hat, 6)
    })
    results['Z_Score'] = (results['Beta'] / results['Std_Error']).round(3)
    results.loc[len(results)] = ['Accuracy', np.round(accuracy, 6), np.nan, np.nan]
    results.loc[len(results)] = ['Pseudo_R2', np.round(pseudo_r2, 6), np.nan, np.nan]

    csv_path = outdir / "True_Maximum_Score_Results.csv"
    results.to_csv(csv_path, index=False)
    print(f"💾 Saved results CSV → {csv_path}")

    # Save summary table as PNG
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    table = ax.table(cellText=results.values, colLabels=results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.2)
    ax.set_title("True Maximum Score Estimator Results (High Precision)", fontsize=12, weight='bold', pad=10)
    table_path = outdir / "True_Maximum_Score_Table.png"
    fig.savefig(table_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(f"💾 Saved summary PNG → {table_path}")

    # Save coefficient bar chart
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(results['Covariate'][:4], results['Beta'][:4], yerr=results['Std_Error'][:4], capsize=4)
    ax2.set_title("True Maximum Score Estimator Coefficients (High Precision)")
    ax2.set_ylabel("Coefficient Estimate")
    fig2.tight_layout()
    bar_path = outdir / "True_Maximum_Score_BarChart.png"
    fig2.savefig(bar_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig2)
    print(f"💾 Saved bar chart → {bar_path}")

    total_time = (time.time() - start_time) / 60.0
    print(f"✅ All done! Total runtime: {total_time:.2f} minutes")

    return results, beta_hat, best_score


# -------------------------
# CLI entrypoint
# -------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="True Maximum Score Estimator (random-search)")
    parser.add_argument("--input", "-i", required=True, help="Path to merged/filtered CSV (MergedTeams_Filtered.csv)")
    parser.add_argument("--outdir", "-o", default=None, help="Directory to save outputs (default = same folder as input)")
    parser.add_argument("--n-iter", type=int, default=20000, help="Random-search iterations for main estimator")
    parser.add_argument("--sample-size", type=int, default=20000, help="Number of pair observations to use")
    parser.add_argument("--n-boot", type=int, default=40, help="Bootstrap replications")
    parser.add_argument("--n-boot-iter", type=int, default=5000, help="Iterations for each bootstrap search")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (useful in notebooks)")
    return parser.parse_args(argv)


def main(argv=None):
    if "ipykernel" in sys.modules:
        # avoid Jupyter adding argv noise when running with -m or !python
        argv = []
    args = _parse_args(argv)
    results, beta_hat, best_score = true_maximum_score_estimator(
        input_csv=args.input,
        outdir=args.outdir,
        n_iter=args.n_iter,
        sample_size=args.sample_size,
        n_boot=args.n_boot,
        n_boot_iter=args.n_boot_iter,
        random_state=args.random_state,
        show=args.show
    )
    return results, beta_hat, best_score


if __name__ == "__main__":
    main()

