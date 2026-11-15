# Predicting House Sale Prices in Chennai

Data Science Project (Part A). The project predicts the sale price of residential properties in Chennai from their features. The focus is on how the data is handled: cleaning, leakage control, a leak-free modelling pipeline, and honest evaluation.

## Problem and goal
- **Question:** can a property's sale price be predicted from its features, and which features matter most?
- **Goal (set before modelling):** test R2 >= 0.85 and MAPE <= 15 %, clearly beating an average-price baseline.

## Dataset
- Public Kaggle dataset of Chennai property sales ("Chennai House Price Prediction", usually the file `train-chennai-sale.csv`).
- Real, several thousand rows, with realistic mess: inconsistent labels, missing values, dates stored as text, a skewed price and columns that leak the price.
- The data is historical, so it describes past prices, not today's market.
- Download it from Kaggle yourself and check the page for its licence. The data is not included in this folder.

## Files
| File | Purpose |
|---|---|
| `chennai_house_price.py` | the whole project, from raw CSV to results |
| `requirements.txt` | Python packages needed |
| `README.md` | this file |


## How to run
1. Install Python 3.9 or newer.
2. Put the dataset CSV in this folder (or in a folder called `data/`).
3. Install the packages: `pip install -r requirements.txt`
4. Run: `python chennai_house_price.py`

The script prints a labelled report: data overview, cleaning steps (with counts), EDA insights, the model comparison table, evaluation, error by locality, top features and the goal check. It also warns if the score looks too good to be true (a sign of data leakage). It saves charts to `figures/` and the results table to `model_results.csv`. A fixed random seed (42) is used everywhere, so results repeat.

## What the script does
1. **Load:** reads the CSV, standardises column names, reports missing values.
2. **Clean:** removes duplicates, fixes messy text labels, converts dates, removes impossible rows (built after sold), drops leakage columns (registration fee, commission, and any column derived from the price, such as `INF_PRICE` and `POWER(Z,Y)` if present, plus anything with a correlation above 0.95 with the price), and removes only implausible price-per-sq-ft rows as outliers.
3. **Feature engineering:** `SALE_YEAR`, `PROPERTY_AGE`, `SQFT_PER_ROOM`.
4. **EDA:** price distribution (raw and log), price vs size, price by locality, correlation heatmap.
5. **Split and pipeline:** 80/20 train/test split. Imputing, scaling and one-hot encoding sit inside a scikit-learn Pipeline, learned from training data only, so there is no leakage.
6. **Models:** an average-price baseline, Linear Regression and Random Forest (small grid search), compared with 5-fold cross-validation. The best model is chosen by cross-validation, not by test score.
7. **Evaluation:** RMSE, MAE, R2, MAPE, predicted-vs-actual and residual plots, error by locality, feature importance, and a check against the goal.

## Output charts (`figures/`)
`01_outliers`, `02_price_distribution`, `03_price_vs_size`, `04_price_by_area`, `05_correlation_heatmap`, `06_predicted_vs_actual_and_residuals`, `07_feature_importance`

## Results
Fill in after running the script.

| Model | CV RMSE | Test RMSE | Test MAE | Test R2 | Test MAPE % |
|---|---|---|---|---|---|
| Baseline (average) | | | | | |
| Linear Regression | | | | | |
| Random Forest | | | | | |

Best model: ____  |  Goal met: ____

## Limitations and next steps
- Historical data: predictions describe past prices.
- No coordinates, floor level, amenities or condition, which limits accuracy.
- The split is random; a time-based split would be a harder, more realistic test.
- Localities with few sales have less evidence, so predictions there are less reliable.
- Next: add location features and recent data, try gradient boosting, report prediction intervals.

## Troubleshooting
- *"Put the Chennai house price CSV..."*: the CSV is not in this folder or in `data/`.
- *KeyError on a column name*: the file's columns differ from what the script expects (`PRT_ID`, `AREA`, `INT_SQFT`, `N_ROOM`, `DATE_SALE`, `DATE_BUILD`, `SALES_PRICE`). Check the header row and adjust the names in the script.
- Tested with Python packages: pandas 3.0, numpy 2.5, scikit-learn 1.9, matplotlib 3.11, seaborn 0.13.
