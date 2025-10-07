import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("states_all_extended.csv")

# Create spending variables like in ps5_models.py
df['spend_per_student'] = df['INSTRUCTION_EXPENDITURE'] / df['ENROLL']
df = df[df['spend_per_student'] > 0]
df['log_spending'] = np.log(df['spend_per_student'])

# Print basic info
print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- First few rows ---")
print(df.head())

print("\n--- Summary statistics ---")
print(df.describe().T)
