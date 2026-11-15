"""Chennai House Price Prediction - Data Science Project (Part A)
Run: python chennai_house_price.py   (CSV in this folder or in data/)
Goal set before modelling: test R2 >= 0.85 and MAPE <= 15 %, beating an average-price baseline."""
import difflib
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42                                   # fixed seed -> repeatable results
np.random.seed(SEED)
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")
GOAL_R2, GOAL_MAPE = 0.85, 15.0
rs = lambda v: f"Rs {v:,.0f}"               # format a number as rupees


def save(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=120)
    plt.close()


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ================================================================ 1. DATA OVERVIEW
header("1. DATA OVERVIEW")
files = sorted(glob.glob("*.csv") + glob.glob("data/*.csv"))
if not files:
    raise SystemExit("Put the Chennai house price CSV in this folder (or in data/) and run again.")
df = pd.read_csv(files[0])
df.columns = df.columns.str.strip().str.upper()
TARGET = "SALES_PRICE"
rows_raw = len(df)
print(f"File: {files[0]}")
print(f"Size: {rows_raw:,} rows x {df.shape[1]} columns | target to predict: {TARGET}")
missing = df.isna().sum()
missing = missing[missing > 0]
print("Columns with missing values (filled later, inside the model pipeline):")
print("\n".join(f"  {c}: {n} missing ({n / rows_raw:.2%})" for c, n in missing.items()) or "  none")

# ================================================================ 2. DATA CLEANING
header("2. DATA CLEANING")

# Duplicates: a repeated house would be counted twice (and could sit in both train and test).
df = df.drop_duplicates().drop_duplicates(subset="PRT_ID")
print(f"Duplicate rows/IDs removed: {rows_raw - len(df)}")


# Messy labels ('Ann Nagar' vs 'Anna Nagar'): merge rare labels into the most similar common label.
def fix_labels(col):
    original = col.astype(str).str.strip().where(col.notna())
    key = original.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    counts = key.value_counts()
    common = counts[counts >= 0.02 * len(col)].index.tolist()
    to_common = {k: (difflib.get_close_matches(k, common, n=1, cutoff=0.75) or [k])[0] for k in counts.index}
    spelling = original.groupby(key).agg(lambda x: x.mode()[0])
    return key.map(to_common).map(spelling)


print("Text labels checked (unique labels before -> after):")
text_cols = [c for c in df.columns
             if c not in ("PRT_ID", "DATE_SALE", "DATE_BUILD") and not pd.api.types.is_numeric_dtype(df[c])]
for col in text_cols:
    before = df[col].nunique()
    df[col] = fix_labels(df[col])
    after = df[col].nunique()
    print(f"  {col}: {before} -> {after}" + ("  (typos merged)" if after < before else "  (already consistent)"))

# Dates are text -> convert; a house cannot be built after it was sold.
df["DATE_SALE"] = pd.to_datetime(df["DATE_SALE"], dayfirst=True, errors="coerce")
df["DATE_BUILD"] = pd.to_datetime(df["DATE_BUILD"], dayfirst=True, errors="coerce")
impossible = df["DATE_BUILD"] > df["DATE_SALE"]
df = df[~impossible]
print(f"Dates converted. Sales range: {df['DATE_SALE'].min().date()} to {df['DATE_SALE'].max().date()} (historical data)")
print(f"Impossible rows removed (built after being sold): {int(impossible.sum())}")

# Leakage: columns that are worked out from the price (fees, 'inflated' price, formula columns) would give the
# model the answer. A real predictor would not have them, so they are dropped.
KNOWN_LEAK = ["REG_FEE", "COMMIS", "INF_PRICE", "POWER(Z,Y)"]
corr_price = df.select_dtypes(include="number").corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
leak = sorted(set([c for c in KNOWN_LEAK if c in df.columns] + corr_price[corr_price > 0.95].index.tolist()))
print("Correlation with price (strongest 5 numeric columns):")
print("\n".join(f"  {c}: {v:.3f}" + ("   <- dropped as leakage" if c in leak else "") for c, v in corr_price.head(5).items()))
print(f"Columns dropped as leakage: {leak if leak else 'none'}")
df = df.drop(columns=leak)

# Outliers: large houses are real and are kept; only impossible price-per-sq-ft rows are removed.
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
sns.boxplot(y=df[TARGET], ax=ax[0])
sns.boxplot(y=df["INT_SQFT"], ax=ax[1])
save("01_outliers")
ppsf = np.log(df[TARGET] / df["INT_SQFT"])
q1, q3 = ppsf.quantile([0.25, 0.75])
extreme = (ppsf < q1 - 3 * (q3 - q1)) | (ppsf > q3 + 3 * (q3 - q1))
df = df[~extreme]
print(f"Implausible price-per-sq-ft rows removed (outliers): {int(extreme.sum())}  (large houses are kept)")
print(f"Rows after cleaning: {rows_raw:,} -> {len(df):,}")

# ================================================================ 3. FEATURE ENGINEERING + EDA
header("3. FEATURE ENGINEERING AND EDA")
df["SALE_YEAR"] = df["DATE_SALE"].dt.year                                   # prices change over time
df["PROPERTY_AGE"] = df["DATE_SALE"].dt.year - df["DATE_BUILD"].dt.year      # older buildings sell for less
df["SQFT_PER_ROOM"] = (df["INT_SQFT"] / df["N_ROOM"]).replace([np.inf, -np.inf], np.nan)   # spaciousness
print("New features created: SALE_YEAR, PROPERTY_AGE, SQFT_PER_ROOM")
price = df[TARGET]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(price, bins=40, kde=True, ax=ax[0])
sns.histplot(np.log1p(price), bins=40, kde=True, ax=ax[1])
save("02_price_distribution")
sk_raw, sk_log = price.skew(), np.log1p(price).skew()
print(f"\nPrice distribution: skewness {sk_raw:.2f} (raw) -> {sk_log:.2f} (after log). "
      f"Median price {rs(price.median())}, mean {rs(price.mean())}.")
print("  Insight: " + ("price is right-skewed, so the models learn log(price) and convert back to rupees."
                       if sk_raw > 0.5 else "price is only mildly skewed, but the log still makes it more symmetric, so the models learn log(price)."))

sns.scatterplot(data=df, x="INT_SQFT", y=TARGET, alpha=0.4, s=15)
save("03_price_vs_size")
print(f"\nPrice vs size: correlation {df['INT_SQFT'].corr(price):.2f}.")
print("  Insight: bigger houses cost more, but the spread around the trend shows location and other features matter too.")

area_med = df.groupby("AREA")[TARGET].median().sort_values()
area_med.plot.barh()
save("04_price_by_area")
print(f"\nPrice by locality: highest median in {area_med.index[-1]} ({rs(area_med.iloc[-1])}), "
      f"lowest in {area_med.index[0]} ({rs(area_med.iloc[0])}), a {area_med.iloc[-1] / area_med.iloc[0]:.1f}x gap.")
print("  Insight: locality strongly affects price, which is why clean, consistent locality labels matter.")

corr = df.select_dtypes(include="number").corr()
plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f")
save("05_correlation_heatmap")
top3 = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(3).index
print("\nNumeric features most related to price: " + ", ".join(f"{c} ({corr.loc[c, TARGET]:.2f})" for c in top3))

# ================================================================ 4. SPLIT + PIPELINE
features = [c for c in df.columns if c not in (TARGET, "PRT_ID", "DATE_SALE", "DATE_BUILD")]
X, y = df[features], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])


def make(model):  # preprocessing + model; learns log(price), predicts rupees
    return TransformedTargetRegressor(regressor=Pipeline([("prep", preprocess), ("model", model)]),
                                      func=np.log1p, inverse_func=np.expm1)


# ================================================================ 5. MODELS
header("4. MODELLING")
print(f"Split: {len(X_train):,} training rows (used for cross-validation) | {len(X_test):,} test rows (used once, at the end)")
print(f"Features used ({len(features)}): {num_cols + cat_cols}")
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = "neg_root_mean_squared_error"
models = {
    "Baseline (average)": make(DummyRegressor()),
    "Linear Regression": make(LinearRegression()),
    "Random Forest": GridSearchCV(
        make(RandomForestRegressor(n_estimators=200, random_state=SEED)),
        {"regressor__model__max_depth": [10, None], "regressor__model__min_samples_leaf": [1, 3]},
        cv=cv, scoring=scoring, n_jobs=-1),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    cv_rmse = -model.best_score_ if hasattr(model, "best_score_") else \
        -cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
    pred = model.predict(X_test)
    rows.append({"Model": name, "CV RMSE": cv_rmse,
                 "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                 "Test MAE": mean_absolute_error(y_test, pred),
                 "Test R2": r2_score(y_test, pred),
                 "Test MAPE %": np.mean(np.abs((y_test - pred) / y_test)) * 100})
results = pd.DataFrame(rows).set_index("Model")
results.to_csv("model_results.csv")

shown = results.copy()
for c in ["CV RMSE", "Test RMSE", "Test MAE"]:
    shown[c] = shown[c].map(lambda v: f"{v:,.0f}")
shown["Test R2"] = shown["Test R2"].map("{:.3f}".format)
shown["Test MAPE %"] = shown["Test MAPE %"].map("{:.1f}".format)
print("\nModel comparison (errors in rupees; CV = 5-fold cross-validation on training data):")
print(shown.to_string())
print(f"Random Forest best settings: {models['Random Forest'].best_params_}")

best_name = results.drop(index="Baseline (average)")["CV RMSE"].idxmin()   # chosen on CV, not on test
best = models[best_name]
b, base = results.loc[best_name], results.loc["Baseline (average)"]
print(f"\nBest model (lowest cross-validated RMSE): {best_name}")

# ================================================================ 6. EVALUATION
header("5. EVALUATION")
pred = best.predict(X_test)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(y_test, pred, s=10, alpha=0.4)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax[1].scatter(pred, y_test - pred, s=10, alpha=0.4)
ax[1].axhline(0, color="r", ls="--")
save("06_predicted_vs_actual_and_residuals")

print(f"Final model: {best_name}, scored once on the {len(X_test):,} unseen test houses:")
print(f"  R2   = {b['Test R2']:.3f}  (share of price variation explained)")
print(f"  RMSE = {rs(b['Test RMSE'])}  (typical error, large misses weigh more)")
print(f"  MAE  = {rs(b['Test MAE'])}  (average error)")
print(f"  MAPE = {b['Test MAPE %']:.1f}%  (average percentage error)")
print(f"  It cuts the baseline's RMSE by {(1 - b['Test RMSE'] / base['Test RMSE']) * 100:.0f}% "
      f"({rs(base['Test RMSE'])} -> {rs(b['Test RMSE'])}).")
lin, rf = results.loc["Linear Regression", "Test R2"], results.loc["Random Forest", "Test R2"]
print(f"  Linear vs Random Forest R2: {lin:.3f} vs {rf:.3f} -> " +
      ("the forest finds non-linear patterns the linear model misses." if rf - lin > 0.02
       else "the simple linear model already captures most of the signal."))

err = X_test.assign(pct_error=(pred - y_test).abs() / y_test * 100)
by_area = err.groupby("AREA")["pct_error"].mean().sort_values()
print(f"\nError analysis by locality (average % error): lowest in {by_area.index[0]} ({by_area.iloc[0]:.1f}%), "
      f"highest in {by_area.index[-1]} ({by_area.iloc[-1]:.1f}%).")
print("\n".join(f"  {a}: {v:.1f}%" for a, v in by_area.items()))

imp = permutation_importance(best, X_test, y_test, n_repeats=5, random_state=SEED, scoring="r2", n_jobs=-1)
imp = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
imp.head(10).sort_values().plot.barh()
save("07_feature_importance")
print("\nMost important features (drop in R2 when the feature is shuffled):")
print("\n".join(f"  {c}: {v:.3f}" for c, v in imp.head(5).items()))

# Example predictions: 10 random houses from the test set (never seen during training).
sample = X_test.sample(10, random_state=SEED)
shown_inputs = [c for c in ["AREA", "BUILDTYPE", "INT_SQFT", "N_BEDROOM", "N_BATHROOM", "PROPERTY_AGE", "SALE_YEAR"]
                if c in sample.columns]
example = sample[shown_inputs].copy()
example["ACTUAL"] = y_test.loc[sample.index]
example["PREDICTED"] = best.predict(sample)
example["ERROR %"] = (example["PREDICTED"] - example["ACTUAL"]) / example["ACTUAL"] * 100
example.round(1).to_csv("example_predictions.csv", index=False)
print("\nExample predictions (10 random unseen test houses; + error = model guessed too high):")
display_example = example.copy()
for c in ["ACTUAL", "PREDICTED"]:
    display_example[c] = display_example[c].map(rs)
display_example["ERROR %"] = display_example["ERROR %"].map("{:+.1f}%".format)
print(display_example.to_string(index=False))

print("\nGoal check (set before modelling):")
print(f"  R2 >= {GOAL_R2}: {b['Test R2']:.3f} -> {'MET' if b['Test R2'] >= GOAL_R2 else 'NOT MET'}")
print(f"  MAPE <= {GOAL_MAPE:.0f}%: {b['Test MAPE %']:.1f}% -> {'MET' if b['Test MAPE %'] <= GOAL_MAPE else 'NOT MET'}")
print(f"  Beats the baseline: {'yes' if b['Test RMSE'] < base['Test RMSE'] else 'no'}")
if b["Test R2"] > 0.98 or b["Test MAPE %"] < 1:
    print("\nWARNING: this score is unusually high for house prices. Check the feature list and the most "
          "important features above for columns that are derived from the price (data leakage).")

print("\nCharts saved in figures/; tables saved in model_results.csv and example_predictions.csv")