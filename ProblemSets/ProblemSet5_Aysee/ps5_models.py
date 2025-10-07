import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from pathlib import Path

def clean_data(filepath="states_all_extended.csv"):
    """
    Load and clean the dataset.

    Steps:
    - Read CSV from `filepath` (relative path).
    - Keep relevant columns and drop rows with missing values.
    - Create 'spend_per_student' and its log 'log_spending'.
    - Uppercase state names and replace spaces with underscores.
    - Set a panel index (STATE, YEAR) and sort.
    Returns:
        pd.DataFrame with panel index (STATE, YEAR).
    """
    df = pd.read_csv(filepath)
    cols = ['STATE', 'YEAR', 'ENROLL', 'INSTRUCTION_EXPENDITURE', 'G08_A_A_MATHEMATICS']
    df = df[cols].dropna()
    df['spend_per_student'] = df['INSTRUCTION_EXPENDITURE'] / df['ENROLL']
    df = df[df['spend_per_student'] > 0]
    df['log_spending'] = np.log(df['spend_per_student'])
    # Standardize state names for indexing
    df['STATE'] = df['STATE'].str.upper().str.replace(" ", "_", regex=False)
    df = df.set_index(['STATE', 'YEAR']).sort_index()
    return df

def create_visuals(df, output_dir="images"):
    """
    Generate visuals and save them to `output_dir`:
      - fig1_math_trend.png: national average math score by year
      - fig2_top_spending_states.png: top 10 states by spending (latest year)
      - fig3_spending_vs_score.png: scatterplot of log spending vs math score

    Returns:
        None (saves files to disk). Uses portable Path operations.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Fig 1: Math score over time
    avg_by_year = df.groupby('YEAR')['G08_A_A_MATHEMATICS'].mean()
    plt.figure(figsize=(8,5))
    avg_by_year.plot(marker='o')
    plt.title("National Avg 8th Grade Math Score (NAEP, 2000–2015)")
    plt.xlabel("Year")
    plt.ylabel("Math Score")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "fig1_math_trend.png", dpi=200)
    plt.close()

    # Fig 2: Top 10 states by spending (latest year)
    latest_year = int(df.reset_index()['YEAR'].max())
    top_states = df.reset_index()
    top_states = top_states[top_states['YEAR'] == latest_year]
    top_states = top_states.sort_values('spend_per_student', ascending=False).head(10)
    plt.figure(figsize=(10,5))
    # Use hue to silence future warning; remove legend after plotting
    sns.barplot(
        data=top_states,
        x='STATE', y='spend_per_student',
        palette='Blues_d', hue='STATE', dodge=False
    )
    # remove legend if present
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.title(f"Top 10 States by Instruction Spending per Student ({latest_year})")
    plt.ylabel("Spending per Student ($)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "fig2_top_spending_states.png", dpi=200)
    plt.close()

    # Fig 3: Scatter plot spending vs score
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df.reset_index(), x='log_spending', y='G08_A_A_MATHEMATICS', alpha=0.7)
    sns.regplot(data=df.reset_index(), x='log_spending', y='G08_A_A_MATHEMATICS', scatter=False)
    plt.title("Instruction Spending vs 8th Grade Math Score (State-Year Panel)")
    plt.xlabel("Log(Instruction Spending per Student)")
    plt.ylabel("8th Grade Math Score")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "fig3_spending_vs_score.png", dpi=200)
    plt.close()

def estimate_models(df, output_tex="results.tex"):
    """
    Estimate three models and save a summary LaTeX table.

    Models:
      1) Baseline OLS:            G08_A_A_MATHEMATICS ~ log_spending (HC1 robust SE)
      2) State fixed effects:     PanelOLS with EntityEffects
      3) Two-way fixed effects:   PanelOLS with EntityEffects + TimeEffects

    Writes a human-friendly LaTeX table to `output_tex`. This manual writer avoids pandas' Styler/Jinja2 dependency.
    """
    results = {}

    # Baseline OLS (statsmodels)
    y = df['G08_A_A_MATHEMATICS']
    X = sm.add_constant(df['log_spending'])
    ols_model = sm.OLS(y, X).fit(cov_type='HC1')
    results['Baseline OLS'] = ols_model

    # State fixed effects (linearmodels)
    fe_model = PanelOLS.from_formula("G08_A_A_MATHEMATICS ~ log_spending + EntityEffects", data=df)
    fe_res = fe_model.fit(cov_type='clustered', cluster_entity=True)
    results['State FE'] = fe_res

    # Two-way FE (state + year)
    fe2_model = PanelOLS.from_formula(
        "G08_A_A_MATHEMATICS ~ log_spending + EntityEffects + TimeEffects", data=df
    )
    fe2_res = fe2_model.fit(cov_type='clustered', cluster_entity=True)
    results['Two-Way FE'] = fe2_res

    # Collect numeric results into a dict of rows
    table = {}
    for name, model in results.items():
        coef = np.nan
        se = np.nan
        r2 = np.nan

        # coefficient extraction (works for statsmodels and linearmodels)
        if hasattr(model, "params"):
            try:
                coef = float(model.params.get('log_spending', model.params.get('log\\_spending', np.nan)))
            except Exception:
                # fallback to first param
                try:
                    coef = float(model.params.iloc[0])
                except Exception:
                    coef = np.nan

        # standard errors
        if hasattr(model, "bse"):
            try:
                se = float(model.bse.get('log_spending', model.bse.get('log\\_spending', np.nan)))
            except Exception:
                try:
                    se = float(model.bse.iloc[0])
                except Exception:
                    se = np.nan
        elif hasattr(model, "std_errors"):
            try:
                se = float(model.std_errors.get('log_spending', model.std_errors.get('log\\_spending', np.nan)))
            except Exception:
                try:
                    se = float(model.std_errors.iloc[0])
                except Exception:
                    se = np.nan
        else:
            se = np.nan

        # R-squared: try common attributes / fallbacks
        r2_try = getattr(model, "rsquared", None)
        if r2_try is None:
            r2_try = getattr(model, "rsquared_within", None) or getattr(model, "rsquared_between", None)
        try:
            r2 = float(r2_try) if r2_try is not None else np.nan
        except Exception:
            r2 = np.nan

        table[name] = {
            "Coefficient (log_spending)": coef,
            "Std. Error": se,
            "R-squared": r2
        }

    # Convert to DataFrame and format for output
    df_table = pd.DataFrame(table).T
    df_table.index.name = "Model"
    # Round numeric values for display (keeps numeric types)
    df_table = df_table.round(2)

    # Ensure output path exists
    outpath = Path(output_tex)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # safe formatter for LaTeX output
    def safe_fmt(x, ndigits=2):
        try:
            if pd.isna(x):
                return "--"
            val = float(x)
            return f"{val:.{ndigits}f}"
        except Exception:
            return str(x)

    # Write human-friendly LaTeX table manually (avoids pandas Styler / jinja2)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("% Auto-generated regression summary\n")
        f.write("\\textbf{Dep. Variable: 8th Grade Math Score (NAEP)}\n\n")
        f.write("% Models included:\n")
        f.write("Baseline OLS\\\\\nState FE\\\\\nTwo-Way FE\\\\\n\n")

        # Manual LaTeX table (booktabs)
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Model & Coefficient (log\\_spending) & Std. Error & R-squared \\\\\n")
        f.write("\\midrule\n")

        for model, row in df_table.iterrows():
            model_tex = str(model).replace("_", "\\_")
            coef = safe_fmt(row.get("Coefficient (log_spending)", np.nan))
            se = safe_fmt(row.get("Std. Error", np.nan))
            r2 = safe_fmt(row.get("R-squared", np.nan))
            f.write(f"{model_tex} & {coef} & {se} & {r2} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Regression results (coefficients on log spending).}\n\\end{table}\n")

if __name__ == "__main__":
    # Run full pipeline when script is executed directly
    df_clean = clean_data()
    create_visuals(df_clean)
    estimate_models(df_clean)
    print("✅ Analysis complete — visuals saved to /images and regression output saved to results.tex")
