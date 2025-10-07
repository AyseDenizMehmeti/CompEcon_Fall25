# tests/test_ps5.py
import os
import pandas as pd
import pytest
from pathlib import Path

from ps5_models import clean_data, create_visuals, estimate_models

@pytest.fixture
def df():
    """Fixture to load and clean data once for all tests."""
    return clean_data("states_all_extended.csv")

def test_clean_data(df):
    """Check that clean_data returns a DataFrame with no missing values and correct columns."""
    assert isinstance(df, pd.DataFrame)
    expected_cols = {"ENROLL", "INSTRUCTION_EXPENDITURE", "G08_A_A_MATHEMATICS", "spend_per_student", "log_spending"}
    assert expected_cols.issubset(df.columns)
    assert df.notna().all().all(), "DataFrame contains NaNs"

def test_create_visuals(df, tmp_path):
    """Check that create_visuals produces PNG files in the given directory."""
    outdir = tmp_path / "images"
    create_visuals(df, output_dir=outdir)
    figs = ["fig1_math_trend.png", "fig2_top_spending_states.png", "fig3_spending_vs_score.png"]
    for fig in figs:
        assert (outdir / fig).exists(), f"{fig} was not created"

def test_estimate_models(df, tmp_path):
    """Check that estimate_models creates a .tex file with regression results."""
    outpath = tmp_path / "results.tex"
    estimate_models(df, output_tex=outpath)

    # File should exist
    assert outpath.exists(), "results.tex was not created"

    # Read contents
    content = outpath.read_text()

    # Check that regression results are there (more robust than checking a single variable name)
    assert "Dep. Variable" in content, "Regression summary missing"
    assert "Baseline OLS" in content, "OLS section missing"
    assert "State FE" in content, "State FE section missing"
    assert "Two-Way FE" in content, "Two-Way FE section missing"
