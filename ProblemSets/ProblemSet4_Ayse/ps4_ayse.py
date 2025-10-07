#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Problem Set 4
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

# Load data
df = pd.read_stata("C:/Users/Deniz Bilen/Downloads/PSID_data.dta")

# Create hourly wage = income / hours
df["hr_wage"] = df["hlabinc"] / df["hannhrs"]

# Filter conditions
df = df[
    (df["hsex"] == 1) &               # male
    (df["age"].between(25, 60)) &     # age 25–60
    (df["hr_wage"] > 7)               # hourly wage > $7
].copy()

# Add log wage
df["ln_wage"] = np.log(df["hr_wage"])

print(df.head())




# In[8]:


# Define variables
df["educ"] = df["hyrsed"]
df["age2"] = df["age"]**2
df["black"] = (df["hrace"] == 2).astype(int)
df["hispanic"] = (df["hrace"] == 5).astype(int)
df["other"] = (~df["hrace"].isin([1, 2, 5])).astype(int)

# Drop missing rows
reg_vars = ["ln_wage", "educ", "age", "age2", "black", "hispanic", "other"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=reg_vars)

# Build X and y
X = df[["educ", "age", "age2", "black", "hispanic", "other"]]
X = sm.add_constant(X)
y = df["ln_wage"]

# OLS
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

# MLE
init_params = np.append(ols_model.params.values, ols_model.resid.std())
bounds = [(None, None)]*X.shape[1] + [(1e-6, None)]

result = minimize(neg_loglike, init_params, args=(y, X), # type: ignore
                  method="L-BFGS-B", bounds=bounds)

mle_params = result.x
print("MLE coefficients:", mle_params[:-1])
print("MLE sigma:", mle_params[-1])





# In[5]:


from scipy.optimize import minimize
import numpy as np

def neg_loglike(params, y, X):
    beta = params[:-1]
    sigma = params[-1]
    resid = y - X @ beta
    n = len(y)
    ll = -0.5*n*np.log(2*np.pi*sigma**2) - 0.5*np.sum((resid/sigma)**2)
    return -ll  # minimize negative log-likelihood

# Initial guess: all betas=0, sigma=1
init_params = np.append(np.zeros(X.shape[1]), 1.0)

# Bounds: sigma > 0
bounds = [(None, None)]*X.shape[1] + [(1e-6, None)]

result = minimize(neg_loglike, init_params, args=(y, X),
                  method="L-BFGS-B", bounds=bounds)

mle_params = result.x
print("MLE coefficients:", mle_params[:-1])
print("MLE sigma:", mle_params[-1])
print(X.head())


# In[6]:


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

# Years required by assignment
years = [1971, 1980, 1990, 2000]

results = []

for year in years:
    # Filter data for the given year
    df_year = df[df["year"] == year].copy()

    # Recreate variables
    df_year["educ"] = df_year["hyrsed"]
    df_year["age2"] = df_year["age"]**2
    df_year["black"] = (df_year["hrace"] == 2).astype(int)
    df_year["hispanic"] = (df_year["hrace"] == 5).astype(int)
    df_year["other"] = (~df_year["hrace"].isin([1, 2, 5])).astype(int)

    # Drop missing
    reg_vars = ["educ", "age", "age2", "black", "hispanic", "other"]
    df_year = df_year.replace([np.inf, -np.inf], np.nan)
    df_year = df_year.dropna(subset=["ln_wage"] + reg_vars)

    X = df_year[reg_vars]
    X = sm.add_constant(X)
    y = df_year["ln_wage"]

    # OLS
    ols_model = sm.OLS(y, X).fit()
    beta1_ols = ols_model.params["educ"]

    # MLE
    init_params = np.append(ols_model.params.values, ols_model.resid.std())
    bounds = [(None, None)]*X.shape[1] + [(1e-6, None)]
    result = minimize(neg_loglike, init_params, args=(y, X),
                      method="L-BFGS-B", bounds=bounds)
    beta1_mle = result.x[X.columns.get_loc("educ")]  # educ coefficient

    results.append({
        "year": year,
        "OLS_beta1": beta1_ols,
        "MLE_beta1": beta1_mle
    })

# Convert to table
results_df = pd.DataFrame(results)
print(results_df)




# ### Interpretation of β₁ (Education)
# 
# The coefficient on education (β₁) reflects the percentage increase in wages associated with an additional year of schooling, showing that returns were about 6.7% in 1971 and remained stable at roughly 6.6% in 1980, but rose significantly to around 9.6% in 1990 and further to 11.0% in 2000; overall, this indicates that while returns to education were relatively flat in the 1970s and 1980s, they increased substantially in the 1990s and 2000s, suggesting that education has become increasingly rewarded in the labor market over time.
# 
# 

# In[ ]:




