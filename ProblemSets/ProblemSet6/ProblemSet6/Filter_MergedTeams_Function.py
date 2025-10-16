# Filter_MergedTeams_Function.py
#!/usr/bin/env python3
"""
Filter_MergedTeams_Function.py

Filters merged Kaggle teams to keep only teams with:
 - at least one member who has Points >= 2  OR  HighestRanking <= 681
 - and caps all HighestRanking values at 681

Usage:
    python filter_merged_teams.py --input data/MergedTeams.csv --output data/MergedTeams_Filtered.csv
"""

import pandas as pd
from pathlib import Path
import argparse
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['filter_merged_teams.py']


def filter_merged_teams(df):
    """
    Filters teams in the merged dataset based on points/ranking criteria.
    Returns a filtered DataFrame.
    """

    required_cols = {'TeamId', 'UserId', 'Points', 'HighestRanking'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 1️⃣ Identify qualifying teams
    qualifying_teams = df.loc[
        (df['Points'] >= 2) | (df['HighestRanking'] <= 681),
        'TeamId'
    ].unique()

    print(f"✅ Found {len(qualifying_teams):,} qualifying teams.")

    # 2️⃣ Filter dataset to those teams
    filtered_df = df[df['TeamId'].isin(qualifying_teams)].copy()
    print(f"Filtered dataset shape: {filtered_df.shape}")

    # 3️⃣ Cap HighestRanking at 681
    filtered_df['HighestRanking'] = filtered_df['HighestRanking'].clip(upper=681)
    print("🔧 HighestRanking values capped at 681 for all remaining observations.")

    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter MergedTeams.csv by competition ranking and points criteria."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/MergedTeams.csv",
        help="Path to input CSV (default: data/MergedTeams.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/MergedTeams_Filtered.csv",
        help="Path to save filtered CSV (default: data/MergedTeams_Filtered.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    print(f"📂 Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"✅ Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Apply filter
    filtered_df = filter_merged_teams(df)

    # Save filtered dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    print(f"💾 Filtered dataset saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
