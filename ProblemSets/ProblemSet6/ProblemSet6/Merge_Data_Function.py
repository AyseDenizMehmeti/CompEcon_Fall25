#!/usr/bin/env python3
"""
merge_data.py

Portable version of Merge_Data_Function.py

Features:
 - No hard-coded path; specify --dest (default ./data)
 - Option to skip Kaggle download and use existing CSVs (--no-download)
 - CLI options for dataset id, desired base names, output filename, join-date column
 - Cross-platform (uses pathlib)
 - Clear error messages if files are missing or Kaggle not configured
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
import sys
import pandas as pd

# try to import KaggleApi but allow running without it
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    _KAGGLE_AVAILABLE = True
except Exception:
    KaggleApi = None
    _KAGGLE_AVAILABLE = False


DEFAULT_DATASET_ID = "kaggle/meta-kaggle"
DEFAULT_DESIRED_BASE_NAMES = [
    "Teams",
    "TeamMemberships",
    "Users",
    "UserOrganizations",
    "UserAchievements",
]
DEFAULT_OUTPUT = "MergedTeams.csv"
DEFAULT_JOIN_DATE_COL = "JoinDate"


def download_meta_kaggle_files(dataset=DEFAULT_DATASET_ID, dest_dir=Path("data"),
                               desired_names=DEFAULT_DESIRED_BASE_NAMES, unzip=True):
    """
    Download the Kaggle dataset and copy matching CSVs to dest_dir with normalized names.
    Returns dict: { 'Teams.csv': fullpath, ... } for the files found.
    Raises RuntimeError if Kaggle API isn't available or authentication is missing.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not _KAGGLE_AVAILABLE:
        raise RuntimeError("kaggle package not installed or KaggleApi not importable. "
                           "Install with `pip install kaggle` and ensure kaggle.json is configured.")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError("Kaggle authentication failed. Make sure kaggle.json is placed in "
                           "~/.kaggle/ or %USERPROFILE%\\.kaggle\\ and permissions are correct. "
                           f"Underlying error: {e}")

    found = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(f"Downloading dataset '{dataset}' to temporary folder...")
        api.dataset_download_files(dataset, path=str(tmp), unzip=unzip, quiet=False)
        print("Download complete — searching for desired CSVs...")

        # index CSVs in tmp (case insensitive)
        lower_to_path = {}
        for root, _, files in os.walk(tmp):
            for fname in files:
                if fname.lower().endswith(".csv"):
                    key = Path(fname).stem.lower()
                    lower_to_path[key] = Path(root) / fname

        for base in desired_names:
            base_l = base.lower()
            matched_path = None
            if base_l in lower_to_path:
                matched_path = lower_to_path[base_l]
            else:
                # fallback: find any filename containing the base word
                for k, p in lower_to_path.items():
                    if base_l in k:
                        matched_path = p
                        break

            if matched_path:
                dest_fname = f"{base}.csv"
                dest_path = dest_dir / dest_fname
                shutil.copyfile(matched_path, dest_path)
                found[dest_fname] = str(dest_path)
                print(f"  → Found '{matched_path.name}' -> saved as '{dest_fname}'")
            else:
                print(f"  ! WARNING: could not find CSV matching '{base}' in the dataset.")

    return found


def find_local_csvs(dest_dir: Path, desired_names):
    """
    Look for files like Teams.csv or case-insensitive matches in dest_dir.
    Returns dict mapping dest filename (e.g. 'Teams.csv') to full path (str) when found.
    """
    found = {}
    lower_to_path = {}
    for p in dest_dir.glob("**/*.csv"):
        name = p.name
        lower_to_path[p.stem.lower()] = str(p)

    for base in desired_names:
        base_l = base.lower()
        matched = None
        if base_l in lower_to_path:
            matched = lower_to_path[base_l]
        else:
            for k, v in lower_to_path.items():
                if base_l in k:
                    matched = v
                    break
        if matched:
            found[f"{base}.csv"] = matched
    return found


def merge_and_dedupe_files(path="data",
                           teams_fname="Teams.csv",
                           memberships_fname="TeamMemberships.csv",
                           users_fname="Users.csv",
                           orgs_fname="UserOrganizations.csv",
                           ach_fname="UserAchievements.csv",
                           output_fname=DEFAULT_OUTPUT,
                           join_date_col=DEFAULT_JOIN_DATE_COL):
    """
    Merge the five CSVs located in `path` and deduplicate the merged output.
    Saves merged, deduped file to path/output_fname and returns the full path.
    """
    path = Path(path)
    teams_fp = path / teams_fname
    mem_fp = path / memberships_fname
    users_fp = path / users_fname
    org_fp = path / orgs_fname
    ach_fp = path / ach_fname

    missing = [str(fp) for fp in (teams_fp, mem_fp, users_fp, org_fp, ach_fp) if not fp.is_file()]
    if missing:
        raise FileNotFoundError(f"Required file(s) not found in {path}:\n" + "\n".join(missing))

    print("Reading CSVs...")
    teams = pd.read_csv(teams_fp, low_memory=False)
    team_memberships = pd.read_csv(mem_fp, low_memory=False)
    users = pd.read_csv(users_fp, low_memory=False)
    user_orgs = pd.read_csv(org_fp, low_memory=False)
    user_achievements = pd.read_csv(ach_fp, low_memory=False)

    print("Filtering achievement rows to AchievementType == 'Competitions' (if column exists)...")
    if "AchievementType" in user_achievements.columns:
        user_achievements = user_achievements[user_achievements["AchievementType"] == "Competitions"]

    print("Merging dataframes...")
    merged = (
        team_memberships
        .merge(teams, left_on="TeamId", right_on="Id", how="left", suffixes=("", "_team"))
        .merge(users, left_on="UserId", right_on="Id", how="left", suffixes=("", "_user"))
        .merge(user_orgs, left_on="UserId", right_on="UserId", how="left", suffixes=("", "_org"))
        .merge(user_achievements, left_on="UserId", right_on="UserId", how="left", suffixes=("", "_ach"))
    )

    print(f"Merged shape before dedupe: {merged.shape}")
    rows_before = len(merged)

    # 1) Drop exact duplicate rows
    merged_nodup = merged.drop_duplicates()
    exact_dups_removed = rows_before - len(merged_nodup)
    print(f"Exact duplicate rows removed: {exact_dups_removed:,}")

    # 2) Drop duplicate (TeamId, UserId) pairs
    before_pair = len(merged_nodup)
    if ("TeamId" in merged_nodup.columns) and ("UserId" in merged_nodup.columns):
        merged_nodup = merged_nodup.drop_duplicates(subset=["TeamId", "UserId"], keep="first")
    pair_dups_removed = before_pair - len(merged_nodup)
    print(f"Duplicate (TeamId, UserId) pairs removed: {pair_dups_removed:,}")

    # 3) If UserId appears in multiple rows, keep earliest JoinDate row per UserId (if JoinDate parseable)
    before_user_dedup = len(merged_nodup)
    if "UserId" in merged_nodup.columns:
        if join_date_col in merged_nodup.columns:
            merged_nodup["_parsed_join"] = pd.to_datetime(merged_nodup[join_date_col], errors="coerce")
            merged_nodup = merged_nodup.sort_values(by=["UserId", "_parsed_join"], na_position="last")
            merged_nodup = merged_nodup.drop_duplicates(subset=["UserId"], keep="first")
            merged_nodup = merged_nodup.drop(columns=["_parsed_join"])
            print("Used JoinDate to keep earliest entry per UserId.")
        else:
            merged_nodup = merged_nodup.drop_duplicates(subset=["UserId"], keep="first")
            print("No JoinDate column found — kept first occurrence per UserId.")
    after_user_dedup = len(merged_nodup)
    kept_by_join = before_user_dedup - after_user_dedup
    print(f"Rows removed by keeping earliest JoinDate per UserId (if applicable): {kept_by_join:,}")

    # Final info & save
    rows_after = len(merged_nodup)
    print("Merged shape after dedupe:", merged_nodup.shape)
    print(f"Total rows removed by deduplication: {rows_before - rows_after:,}")

    out_path = path / output_fname
    merged_nodup.to_csv(out_path, index=False)
    print("Saved merged & deduped file to:", out_path)

    return str(out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download (optional) and merge Meta-Kaggle CSVs into one deduped CSV.")
    parser.add_argument("--dest", "-d", default="data", help="Destination folder to save/read CSVs (default ./data)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID, help=f"Kaggle dataset id (default {DEFAULT_DATASET_ID})")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt to download from Kaggle; use CSVs already in --dest")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output merged filename (default {DEFAULT_OUTPUT})")
    parser.add_argument("--join-date-col", default=DEFAULT_JOIN_DATE_COL, help=f"Column name to use as JoinDate (default '{DEFAULT_JOIN_DATE_COL}')")
    parser.add_argument("--list-found", action="store_true", help="List CSVs found in destination and exit")
    parser.add_argument("--filenames", nargs="*", default=DEFAULT_DESIRED_BASE_NAMES,
                        help="Base names to look for (default common set). Example: Teams Users TeamMemberships")
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # If not downloading, just find local CSVs
    if args.no_download:
        print("Skipping Kaggle download — searching local CSVs in", dest)
        found = find_local_csvs(dest, args.filenames)
    else:
        # attempt to download; if Kaggle isn't available, give helpful message
        try:
            found = download_meta_kaggle_files(dataset=args.dataset, dest_dir=dest, desired_names=args.filenames)
        except Exception as e:
            print("Error while trying to download dataset from Kaggle:", e, file=sys.stderr)
            print("Falling back to searching for existing CSV files in", dest)
            found = find_local_csvs(dest, args.filenames)

    if args.list_found:
        if found:
            print("Found files:")
            for k, v in found.items():
                print(f"  {k} -> {v}")
        else:
            print("No matching CSVs found in", dest)
        return 0

    # If some expected files are missing, inform user but attempt merge (will raise if required files absent)
    missing_bases = [b for b in args.filenames if f"{b}.csv" not in found]
    if missing_bases:
        print("WARNING: Some requested files not found:", missing_bases)

    try:
        merged_path = merge_and_dedupe_files(
            path=dest,
            teams_fname="Teams.csv",
            memberships_fname="TeamMemberships.csv",
            users_fname="Users.csv",
            orgs_fname="UserOrganizations.csv",
            ach_fname="UserAchievements.csv",
            output_fname=args.output,
            join_date_col=args.join_date_col
        )
        print("\nFinal merged file:", merged_path)
    except FileNotFoundError as fe:
        print("Merge failed because required files were missing:", fe, file=sys.stderr)
        # Show what files we did find to help debugging
        if found:
            print("\nFiles I did find in destination (matching patterns):")
            for k, v in found.items():
                print(f"  {k} -> {v}")
        return 2
    except Exception as e:
        print("An unexpected error occurred during merge:", e, file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
