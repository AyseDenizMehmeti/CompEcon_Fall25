import pandas as pd
import os
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

print("="*60)
print("OLD NAVY PRICING ANALYSIS")
print("="*60)

# Create figures directory
os.makedirs('figures', exist_ok=True)

# ============================================================================
# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
data_file = "oldnavy12052025daily.zip"

try:
    # What's actually in the .zip file
    import zipfile
    with zipfile.ZipFile(data_file, 'r') as zf:
        file_list = zf.namelist()
        print(f"   Found zip file. Contains: {file_list}")
    
    # Read the data 
    df = pd.read_csv(data_file)
    print(f"   ✓ Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"   Using file: {data_file}")
    
except FileNotFoundError:
    print(f"   ✗ ERROR: File '{data_file}' not found!")
    print(f"   Current files in {os.getcwd()}:")
    for f in os.listdir('.'):
        print(f"     - {f}")
    exit(1)
except Exception as e:
    print(f"   ✗ ERROR reading {data_file}: {type(e).__name__}: {e}")
    exit(1)

# ============================================================================
# 2. CLEAN DATA
# ============================================================================
print("\n2. Cleaning data...")
df['scrape_date'] = pd.to_datetime(df['scrape_date'])
df['date'] = df['scrape_date'].dt.date

product_cols = ['gender','size','styleId','styleName','ccId','ccName','ccShortDescription']
df['product_id'] = df[product_cols].astype(str).agg('_'.join, axis=1)
print(f"   ✓ Unique products: {df['product_id'].nunique():,}")

# ============================================================================
# 3. CREATE FIGURES 1-5
# ============================================================================
print("\n3. Creating descriptive figures...")

# Create daily dataset
daily = df.copy().rename(columns={
    'effectivePrice':'price',
    'inventoryCount':'inventory',
    'percentageOff':'discount_percent'
})
daily['discount'] = daily['discount_percent'] / 100
daily['promo_type'] = 'none'
daily.loc[daily['discount'] > 0.05, 'promo_type'] = 'sale'
daily.loc[daily['discount'] > 0.30, 'promo_type'] = 'clearance'

# Figure 1: Discount distribution
plt.figure(figsize=(12,5))
sns.histplot(daily[daily['discount'] > 0.02]['discount'], bins=40, kde=True, color="#4C72B0", alpha=0.6)
plt.title("Distribution of Discounts (Excluding Near-Zero Values)")
plt.xlabel("Discount"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/discount_distribution_zoom.png", dpi=300)
plt.close()
print("   ✓ Figure 1: discount_distribution_zoom.png")

# Figure 2: Gender promotions
plt.figure(figsize=(10,6))
sns.countplot(data=daily, x='gender', hue='promo_type', palette=['#DD8452', '#4C72B0', '#55A868'])
plt.title("Promotion Type Frequency by Gender")
plt.xlabel("Gender"); plt.ylabel("Number of Product-Days")
plt.tight_layout()
plt.savefig("figures/gender_promo.png", dpi=300)
plt.close()
print("   ✓ Figure 2: gender_promo.png")

# Figure 3: Clearance by category
clear_cat = (daily[daily['promo_type']=="clearance"]
             .groupby('ccName').size().reset_index(name='count')
             .sort_values('count', ascending=False))
plt.figure(figsize=(12,6))
sns.barplot(data=clear_cat, y='ccName', x='count', hue='count', palette='Reds', legend=False)
plt.title("Clearance Frequency by Category")
plt.xlabel("Number of Clearance Product-Days"); plt.ylabel("Category")
plt.tight_layout()
plt.savefig("figures/clearance_by_category.png", dpi=300)
plt.close()
print("   ✓ Figure 3: clearance_by_category.png")

# Figure 4: Example product
example_id = daily['product_id'].value_counts().index[0]
example = daily[daily['product_id'] == example_id].sort_values('date')
fig, ax1 = plt.subplots(figsize=(14,5))
ax1.plot(example['date'], example['price'], marker='o', color='blue')
ax1.set_ylabel("Price", color='blue'); ax1.tick_params(axis='y', labelcolor='blue')
color_map = {'none':'black', 'sale':'orange', 'clearance':'red'}
colors = example['promo_type'].map(color_map)
ax1.scatter(example['date'], example['price'], c=colors, s=80)
ax2 = ax1.twinx()
ax2.plot(example['date'], example['inventory'], marker='s', color='green')
ax2.set_ylabel("Inventory", color='green'); ax2.tick_params(axis='y', labelcolor='green')
plt.title(f"Price & Inventory Dynamics for Product {example_id}")
plt.tight_layout()
plt.savefig("figures/price_inventory_path_example.png", dpi=300)
plt.close()
print("   ✓ Figure 4: price_inventory_path_example.png")

# Figure 5: Lifecycle analysis
first_dates = daily.groupby('product_id')['date'].min().rename('first_date')
daily = daily.merge(first_dates, on='product_id', how='left')
daily['age'] = (pd.to_datetime(daily['date']) - pd.to_datetime(daily['first_date'])).dt.days
agg = daily.groupby('age').agg(
    median_price=('price','median'),
    p10_price=('price', lambda x: x.quantile(0.10)),
    p90_price=('price', lambda x: x.quantile(0.90)),
    median_inv=('inventory','median'),
    p10_inv=('inventory', lambda x: x.quantile(0.10)),
    p90_inv=('inventory', lambda x: x.quantile(0.90)),
    n_products=('product_id','nunique')
).reset_index()
fig, ax1 = plt.subplots(figsize=(14,5))
ax1.plot(agg['age'], agg['median_price'], color='blue', linewidth=2)
ax1.fill_between(agg['age'], agg['p10_price'], agg['p90_price'], color='blue', alpha=0.15)
ax1.set_ylabel("Price", color='blue'); ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(agg['age'], agg['median_inv'], color='green', linewidth=2)
ax2.fill_between(agg['age'], agg['p10_inv'], agg['p90_inv'], color='green', alpha=0.10)
ax2.set_ylabel("Inventory", color='green'); ax2.tick_params(axis='y', labelcolor='green')
plt.title(f"Median Price & Inventory vs Age (Products used: {agg['n_products'].max()})")
plt.tight_layout()
plt.savefig("figures/median_product_lifecycle.png", dpi=300)
plt.close()
print("   ✓ Figure 5: median_product_lifecycle.png")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n4. Feature engineering...")

df = daily.copy()
df = df.sort_values(['product_id', 'date'])
df['first_date'] = df.groupby('product_id')['date'].transform('min')
df['duration'] = (pd.to_datetime(df['date']) - pd.to_datetime(df['first_date'])).dt.days

# One-hot encoding
top_cats = df['ccName'].value_counts().nlargest(12).index
df['cat_reduced'] = df['ccName'].where(df['ccName'].isin(top_cats), 'other')
df = pd.get_dummies(df, columns=['cat_reduced'], prefix='cat', drop_first=True)
df = pd.get_dummies(df, columns=['gender'], prefix='gender', drop_first=True)

df['date'] = pd.to_datetime(df['date'])
df['dow'] = df['date'].dt.weekday
df = pd.get_dummies(df, columns=['dow'], prefix='dow', drop_first=True)

# Inventory features
df['max_inv'] = df.groupby('product_id')['inventory'].transform('max').replace(0, np.nan)
df['rel_inventory'] = (df['inventory'] / df['max_inv']).fillna(0)

# Promotion features
df = df.sort_values(['product_id','date'])
df['promo_flag'] = (df['promo_type'] != 'none').astype(int)
df['promo_7d'] = df.groupby('product_id')['promo_flag'].rolling(7, min_periods=1).mean().reset_index(0,drop=True)

df['promo_sale'] = (df['promo_type'] == 'sale').astype(int)
df['promo_clear'] = (df['promo_type'] == 'clearance').astype(int)

# Demand proxy (simulated since we don't have actual sales)
df['demand_proxy'] = (
    0.1
    + 0.5 * df['promo_sale']
    + 1.0 * df['promo_clear']
    + 0.01 * df['duration']
    + 0.25 * df['promo_7d']
)

# ============================================================================
# 5. DEMAND MODEL
# ============================================================================
print("\n5. Training demand model...")

demand_features = [
    'duration',
    'promo_sale', 'promo_clear',
    'promo_7d',
] + [c for c in df.columns if c.startswith('cat_')] \
  + [c for c in df.columns if c.startswith('gender_')] \
  + [c for c in df.columns if c.startswith('dow_')]

X = df[demand_features].fillna(0)
y = df['demand_proxy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_demand = RandomForestRegressor(
    n_estimators=100,  # Reduced for speed
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_demand.fit(X_train, y_train)
pred_test = rf_demand.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
print(f"   ✓ Demand Model RMSE: {rmse:.4f}")

df['state_demand_hat'] = rf_demand.predict(df[demand_features].fillna(0))
df['state_inventory'] = df['inventory']

# ============================================================================
# 6. POLICY MODEL
# ============================================================================
print("\n6. Training policy model...")

# Create policy dataset
df = df.sort_values(['product_id','date'])
df['action_next'] = df.groupby('product_id')['promo_type'].shift(-1)
df_policy = df.dropna(subset=['action_next']).copy()

action_map = {'none':0, 'sale':1, 'clearance':2}
df_policy['action_label'] = df_policy['action_next'].map(action_map)

policy_features = [
    'state_inventory',
    'state_demand_hat',
    'duration',
    'promo_7d',
] + [c for c in df_policy.columns if c.startswith('cat_')] \
  + [c for c in df_policy.columns if c.startswith('gender_')] \
  + [c for c in df_policy.columns if c.startswith('dow_')]

# Time-based split
dates = sorted(df_policy['date'].unique())
split_date = dates[int(0.8 * len(dates))]
train_mask = df_policy['date'] <= split_date
test_mask  = df_policy['date'] >  split_date

X_train = df_policy.loc[train_mask, policy_features]
X_test  = df_policy.loc[test_mask,  policy_features]
y_train = df_policy.loc[train_mask, 'action_label']
y_test  = df_policy.loc[test_mask, 'action_label']

print(f"   ✓ Training set: {X_train.shape}")
print(f"   ✓ Test set: {X_test.shape}")

# Train classifier
clf = RandomForestClassifier(
    n_estimators=100,  # Reduced for speed
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n   === Classification Report ===")
print(classification_report(y_test, y_pred, labels=[0,1,2], target_names=['none','sale','clearance']))

accuracy = clf.score(X_test, y_test)
print(f"\n   ✓ Accuracy: {accuracy:.4f}")

# Baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_acc = dummy.score(X_test, y_test)
print(f"   ✓ Baseline accuracy: {baseline_acc:.4f}")
print(f"   ✓ Improvement: {(accuracy - baseline_acc)*100:.1f}%")

# Balanced metrics
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"   ✓ Balanced accuracy: {balanced_acc:.4f}")

# ============================================================================
# 7. FINAL FIGURES (6 & 7)
# ============================================================================
print("\n7. Creating final figures...")

# Figure 6: Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['none','sale','clearance'],
            yticklabels=['none','sale','clearance'],
            cmap='Blues')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.close()
print("   ✓ Figure 6: confusion_matrix.png")

# Figure 7: Feature importance
fi = pd.DataFrame({
    'feature': policy_features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

fi_top5 = fi.head(5).copy()
pretty_names = {
    'state_demand_hat': 'Expected Demand',
    'promo_7d': 'Promotion Intensity (7-Day)',
    'duration': 'Product Age (Days)',
    'state_inventory': 'Inventory Level',
    'gender_Women': 'Women\'s Category'
}
fi_top5['pretty_feature'] = fi_top5['feature'].map(pretty_names)

plt.figure(figsize=(10,6))
sns.set(style="whitegrid")
sns.barplot(data=fi_top5, y="pretty_feature", x="importance", palette="Blues", edgecolor="black")
plt.title("Top 5 Policy Feature Importances", fontsize=16, weight='bold')
plt.xlabel("Importance", fontsize=14); plt.ylabel("Feature", fontsize=14)
plt.tight_layout()
plt.savefig("figures/policy_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✓ Figure 7: policy_feature_importance.png")

# ============================================================================
# DONE!
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"Created 7 figures in 'figures/' folder:")

figures = os.listdir('figures')
for i, fig in enumerate(sorted(figures), 1):
    print(f"  {i}. {fig}")

print("\nModel Performance Summary:")
print(f"  • Policy Model Accuracy: {accuracy:.1%}")
print(f"  • Baseline Accuracy: {baseline_acc:.1%}")
print(f"  • Improvement: {(accuracy - baseline_acc)*100:.1f}%")
print(f"  • Balanced Accuracy: {balanced_acc:.1%}")

print("\nNext steps:")
print("  1. pdflatex FinalProject_Ayse.tex")
print("  2. pdflatex FinalProject_Ayse.tex  (run twice for references)")
print("="*60)