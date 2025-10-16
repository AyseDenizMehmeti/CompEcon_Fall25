#!/usr/bin/env python3
"""
Get_Summary_Stats_Filtered_Function.py

Generates descriptive statistics and visualizations for a filtered dataset.

Outputs (saved to --outdir or same folder as input):
  - Descriptive stats table PNG
  - Boxplot of Points & HighestRanking PNG
  - Team size bar chart PNG
  - Tier, Country, and Organization completeness tables PNG

Usage (terminal):
  python generate_summary_stats.py --input data/MergedTeams_Filtered.csv --outdir results --label Filtered --show

Or import in a notebook:
  from generate_summary_stats import generate_summary_stats
  df = pd.read_csv("data/MergedTeams_Filtered.csv")
  paths = generate_summary_stats(df, outdir="results", label_suffix="Filtered", show=True)
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def generate_summary_stats(df, outdir=None, label_suffix="Filtered", show=False):
    """
    Generate and save descriptive tables and plots for the filtered dataset.

    Parameters
    ----------
    df : pandas.DataFrame
    outdir : str or Path, optional
        Directory where output PNGs will be saved. If None, uses the folder of the input DataFrame (if available)
        or the current working directory.
    label_suffix : str
        Suffix used in filenames and titles (e.g., "Filtered")
    show : bool
        If True, call plt.show() after creating each plot (useful in notebooks)

    Returns
    -------
    dict
        Paths to saved files (keys: desc_table, boxplot, team_chart, tier_table, country_table, org_table)
    """
    # determine outdir
    if outdir is None:
        outdir = Path.cwd()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved = {
        "desc_table": None,
        "boxplot": None,
        "team_chart": None,
        "tier_table": None,
        "country_table": None,
        "org_table": None
    }

    # make a local copy to avoid mutating caller DF
    df = df.copy()

    # --- 1️⃣ DESCRIPTIVE STATS: Points & HighestRanking ---
    numeric_cols = [c for c in ("Points", "HighestRanking") if c in df.columns]
    if numeric_cols:
        desc = df[numeric_cols].describe().rename(index={
            "count": "Count",
            "mean": "Mean",
            "std": "Standard Deviation",
            "min": "Minimum",
            "25%": "25th Percentile",
            "50%": "Median (50th %)",
            "75%": "75th Percentile",
            "max": "Maximum"
        })

        desc_formatted = desc.applymap(lambda x: f"{x:,.2f}")

        # Save descriptive stats table as PNG
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.axis("off")
        table = ax.table(
            cellText=desc_formatted.values,
            colLabels=desc_formatted.columns,
            rowLabels=desc_formatted.index,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.2)
        ax.set_title(f"Descriptive Statistics ({label_suffix} Dataset)", fontsize=12, weight="bold", pad=10)

        desc_path = outdir / f"Descriptive_Stats_Table_{label_suffix}.png"
        fig.savefig(desc_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        saved["desc_table"] = str(desc_path)

        # Boxplot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        df[numeric_cols].boxplot(ax=ax2)
        ax2.set_title(f"Distribution of {', '.join(numeric_cols)} ({label_suffix})")
        ax2.set_ylabel("Value")
        ax2.grid(True, linestyle="--", alpha=0.6)
        fig2.tight_layout()

        boxplot_path = outdir / f"Descriptive_Stats_Boxplot_{label_suffix}.png"
        fig2.savefig(boxplot_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig2)
        saved["boxplot"] = str(boxplot_path)
    else:
        print("⚠️ Columns 'Points' or 'HighestRanking' not found in dataset; skipping numeric summaries.")

    # --- 2️⃣ TEAM SIZE ANALYSIS ---
    if "TeamId" in df.columns and "UserId" in df.columns:
        team_sizes = df.groupby("TeamId")["UserId"].nunique().reset_index(name="TeamSize")
        merged = df.merge(team_sizes, on="TeamId", how="left")

        single_member = merged[merged["TeamSize"] == 1]["UserId"].nunique()
        multi_member = merged[merged["TeamSize"] >= 2]["UserId"].nunique()

        team_stats = pd.DataFrame({
            "Team Type": ["Solo Teams (1 member)", "Multi-member Teams (2+)"],
            "User Count": [single_member, multi_member]
        })

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.bar(team_stats["Team Type"], team_stats["User Count"])
        ax3.set_title(f"Users by Team Size Category ({label_suffix})")
        ax3.set_ylabel("Number of Unique Users")
        ax3.set_xticklabels(team_stats["Team Type"], rotation=15)
        fig3.tight_layout()

        team_chart_path = outdir / f"Team_Size_Chart_{label_suffix}.png"
        fig3.savefig(team_chart_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig3)
        saved["team_chart"] = str(team_chart_path)
    else:
        print("⚠️ 'TeamId' or 'UserId' not found in dataset; skipping team-size analysis.")

    # --- 3️⃣ TIER, COUNTRY, AND ORGANIZATION SUMMARIES ---
    if all(col in df.columns for col in ("Tier", "Country", "OrganizationId")):
        # Tier summary (all tiers)
        tier_summary = df["Tier"].value_counts(dropna=False).reset_index()
        tier_summary.columns = ["Tier", "User Count"]

        fig4, ax4 = plt.subplots(figsize=(4, 2.5))
        ax4.axis("off")
        t = ax4.table(cellText=tier_summary.values, colLabels=tier_summary.columns, loc="center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(9)
        t.scale(1.1, 1.2)
        ax4.set_title("User Count by Tier", fontsize=12, weight="bold", pad=10)

        tier_table_path = outdir / f"Tier_Summary_Table_{label_suffix}.png"
        fig4.savefig(tier_table_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig4)
        saved["tier_table"] = str(tier_table_path)

        # Country summary (top 4)
        country_summary = df["Country"].value_counts().nlargest(4).reset_index()
        country_summary.columns = ["Country", "User Count"]

        fig5, ax5 = plt.subplots(figsize=(5, 2.5))
        ax5.axis("off")
        t2 = ax5.table(cellText=country_summary.values, colLabels=country_summary.columns, loc="center", cellLoc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(9)
        t2.scale(1.1, 1.2)
        ax5.set_title("Top 4 Countries by User Count", fontsize=12, weight="bold", pad=10)

        country_table_path = outdir / f"Country_Summary_Table_{label_suffix}.png"
        fig5.savefig(country_table_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig5)
        saved["country_table"] = str(country_table_path)

        # Organization completeness
        org_present = int(df["OrganizationId"].notna().sum())
        org_missing = int(df["OrganizationId"].isna().sum())
        total_obs = len(df)
        org_summary = pd.DataFrame({
            "Category": ["With OrganizationId", "Missing OrganizationId", "Total Observations"],
            "Count": [org_present, org_missing, total_obs]
        })

        fig6, ax6 = plt.subplots(figsize=(5, 2.5))
        ax6.axis("off")
        t3 = ax6.table(cellText=org_summary.values, colLabels=org_summary.columns, loc="center", cellLoc="center")
        t3.auto_set_font_size(False)
        t3.set_fontsize(9)
        t3.scale(1.1, 1.2)
        ax6.set_title("OrganizationId Completeness Summary", fontsize=12, weight="bold", pad=10)

        org_table_path = outdir / f"Organization_Summary_Table_{label_suffix}.png"
        fig6.savefig(org_table_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig6)
        saved["org_table"] = str(org_table_path)
    else:
        print("⚠️ Columns 'Tier', 'Country', or 'OrganizationId' not found; skipping those summaries.")

    return saved


def _load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")
    return pd.read_csv(path, low_memory=False)


def main():
    parser = argparse.ArgumentParser(description="Generate summary stats & plots for filtered MergedTeams data.")
    parser.add_argument("--input", "-i", default="data/MergedTeams_Filtered.csv", help="Path to filtered CSV")
    parser.add_argument("--outdir", "-o", default=None, help="Directory to save outputs (default = same folder as input)")
    parser.add_argument("--label", "-l", default="Filtered", help="Label suffix used in titles/filenames")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (useful in notebooks)")
    args = parser.parse_args()

    input_path = Path(args.input)
    df = _load_csv(input_path)
    outdir = Path(args.outdir) if args.outdir else input_path.parent

    paths = generate_summary_stats(df, outdir=outdir, label_suffix=args.label, show=args.show)
    print("\nSaved files:")
    for k, v in paths.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
