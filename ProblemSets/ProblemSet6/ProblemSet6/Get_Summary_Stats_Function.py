import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def make_descriptive_table(csv_path="data/MergedTeams.csv", output_name="Descriptive_Stats_Table.png"):
    """
    Loads MergedTeams.csv, computes descriptive stats for 'Points' and 'HighestRanking',
    creates a formatted summary table, and saves it as a PNG.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path.resolve()}")

    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Compute Descriptive Stats ---
    cols = [col for col in ["Points", "HighestRanking"] if col in df.columns]
    if not cols:
        raise ValueError("None of ['Points', 'HighestRanking'] columns found in dataset.")

    desc = df[cols].describe().rename(index={
        "count": "Count",
        "mean": "Mean",
        "std": "Standard Deviation",
        "min": "Minimum",
        "25%": "25th Percentile",
        "50%": "Median (50th %)",
        "75%": "75th Percentile",
        "max": "Maximum",
    })

    # Format numbers
    desc_formatted = desc.applymap(lambda x: f"{x:,.2f}")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=desc_formatted.values,
        colLabels=desc_formatted.columns,
        rowLabels=desc_formatted.index,
        loc="center",
        cellLoc="center"
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.2)
    ax.set_title("Descriptive Statistics: Points and HighestRanking", fontsize=12, weight="bold", pad=10)

    # --- Save to same directory as CSV ---
    output_path = csv_path.parent / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Clean summary table saved to:\n{output_path.resolve()}")
    return output_path

# --- Run standalone ---
if __name__ == "__main__":
    make_descriptive_table()
