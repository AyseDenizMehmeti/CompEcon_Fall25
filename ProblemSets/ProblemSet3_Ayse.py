import pandas as pd
import matplotlib.pyplot as plt

# === Load data ===
f = r"C:\Users\Deniz Bilen\Desktop\VAPE\FLORIDA DATA\SUSPENSION BY SCHOOL\sesir2021a-h.csv"

# Load CSV (skip junk rows at top)
df = pd.read_csv(f, encoding="latin1", skiprows=6, low_memory=False)

# Clean column names
df.columns = df.columns.map(str).str.strip().str.lower()

# Keep only the columns we need
df = df[["district name", "school name", "type of incident", "total incidents", "vaping related"]]

# === Visualization 1: Top 10 Schools ===
top_schools = df.groupby("school name")["vaping related"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_schools.plot(kind="barh")
plt.title("Top 10 Schools for Vaping Incidents (2021)")
plt.xlabel("Number of Vaping Incidents")
plt.tight_layout()
plt.savefig("top_schools_vaping.png")   # save figure
plt.close()

# === Visualization 2: Top 10 Districts ===
top_districts = df.groupby("district name")["vaping related"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_districts.plot(kind="barh")
plt.title("Top 10 Districts for Vaping Incidents (2021)")
plt.xlabel("Number of Vaping Incidents")
plt.tight_layout()
plt.savefig("top_districts_vaping.png")  # save figure
plt.close()

# === Visualization 3: Vaping vs Total Incidents ===
compare = df[["vaping related","total incidents"]].sum()

plt.figure(figsize=(6,4))
compare.plot(kind="bar")
plt.title("Vaping vs Total Incidents (2021)")
plt.ylabel("Number of Incidents")
plt.tight_layout()
plt.savefig("vaping_vs_total.png")   # save figure
plt.close()

print("✅ Done! Figures saved as top_schools_vaping.png, top_districts_vaping.png, vaping_vs_total.png")
