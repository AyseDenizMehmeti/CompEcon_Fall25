import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

# Negative log-likelihood
def neg_loglike(params, y, X):
    beta = params[:-1]
    sigma = params[-1]
    resid = y - X @ beta
    n = len(y)
    ll = -0.5*n*np.log(2*np.pi*sigma**2) - 0.5*np.sum((resid/sigma)**2)
    return -ll

def test_mle_matches_ols():
    """
    Unit test: checks that MLE and OLS give nearly identical betas.
    """

    # ------------------------------------------------------------------
    # Load and clean data
    # ------------------------------------------------------------------
    df = pd.read_stata("C:/Users/Deniz Bilen/Downloads/PSID_data.dta")

    # Create hourly wage and log wage
    df["hr_wage"] = df["hlabinc"] / df["hannhrs"]
    df["ln_wage"] = np.log(df["hr_wage"])

    # Apply filters: male, age 25–60, wage > 7
    df = df[
        (df["hsex"] == 1) &
        (df["age"].between(25, 60)) &
        (df["hr_wage"] > 7)
    ].copy()

    # Replace inf with NaN and drop missing
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["ln_wage", "hyrsed", "age", "hrace"])

    # Create regressors
    df["educ"] = df["hyrsed"]
    df["age2"] = df["age"]**2
    df["black"] = (df["hrace"] == 2).astype(int)
    df["hispanic"] = (df["hrace"] == 5).astype(int)
    df["other"] = (~df["hrace"].isin([1, 2, 5])).astype(int)

    reg_vars = ["educ", "age", "age2", "black", "hispanic", "other"]
    df = df.dropna(subset=reg_vars)

    X = df[reg_vars]
    X = sm.add_constant(X)
    y = df["ln_wage"]

    # ------------------------------------------------------------------
    # OLS
    # ------------------------------------------------------------------
    ols_model = sm.OLS(y, X).fit()

    # ------------------------------------------------------------------
    # MLE
    # ------------------------------------------------------------------
    init_params = np.append(ols_model.params.values, ols_model.resid.std())
    bounds = [(None, None)]*X.shape[1] + [(1e-6, None)]
    result = minimize(neg_loglike, init_params, args=(y, X),
                      method="L-BFGS-B", bounds=bounds)
    mle_params = result.x[:-1]

    # ------------------------------------------------------------------
    # Test: MLE and OLS should match
    # ------------------------------------------------------------------
    np.testing.assert_allclose(mle_params, ols_model.params.values, rtol=1e-3, atol=1e-3)