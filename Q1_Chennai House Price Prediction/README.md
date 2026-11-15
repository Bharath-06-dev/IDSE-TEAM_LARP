
# Predicting House Sale Prices in Chennai

A data science project that predicts the sale price of residential properties in Chennai from their features. The focus is on **how the data is handled**: cleaning, leakage control, a leakage-free modelling pipeline, honest model comparison and error analysis. The model itself is deliberately simple and easy to explain.

## Problem and goal

- **Question:** can a property's sale price be predicted from its features, and which features matter most?
- **Goal (fixed before modelling):** test R² ≥ 0.85 and MAPE ≤ 15 %, and clearly beat an average-price baseline.

## Dataset

- Public Kaggle dataset of Chennai property sales (search Kaggle for *Chennai House Price Prediction*).
- 7,109 rows and 24 columns; the target is `SALES_PRICE`.
- The data is not included in this repository. Download it from Kaggle, check the dataset page for its licence, and place the CSV in the project folder (or in `data/`).
- The sales are historical, so the model describes past prices, not today's market.

## Method

| Step | What was done | Why |
|---|---|---|
| Duplicates | Removed repeated rows and property IDs (none found) | A repeated house would be counted twice and could appear in both train and test |
| Messy labels | Checked all text columns for typos and variants and merged near-identical labels (all columns were already consistent) | Variants split one locality into several categories |
| Dates | Converted text dates to datetime; checked for houses built after they were sold (none found) | Needed for property age and sale year |
| Missing values | `N_BEDROOM` (1), `N_BATHROOM` (5), `QS_OVERALL` (48); filled inside the pipeline with the median (numeric) or most frequent value (categorical) | Learned from training data only, so nothing leaks from the test set |
| Leakage | Dropped `REG_FEE`, `COMMIS`, `INF_PRICE` and `POWER(Z,Y)`, plus any column with correlation above 0.95 with the price | These are fees or values derived from the price; a real predictor would not have them, and they inflate accuracy |
| Outliers | Kept large houses (real data); removed only rows with an implausible price per sq ft (none found) | Removing real expensive homes would bias the model |
| Feature engineering | `SALE_YEAR`, `PROPERTY_AGE`, `SQFT_PER_ROOM` | Prices change over time, older buildings sell for less, and size per room captures spaciousness |
| Target | Models learn `log(1 + price)` and convert predictions back to rupees | Price is right-skewed (skewness 0.77 raw, -0.27 after log) |
| Split | 80 % train (5,687 rows) / 20 % test (1,422 rows), fixed seed 42 | The test set is used once, at the end |
| Pipeline | Imputing, scaling and one-hot encoding inside a scikit-learn `Pipeline` | Prevents data leakage during cross-validation |
| Models | Average-price baseline, **Linear Regression**, **Random Forest** (small grid search) | A floor to beat, a simple explainable model, and a flexible non-linear model |
| Validation | 5-fold cross-validation on the training set; best model chosen by CV RMSE, not test score | Keeps the test set truly unseen |

## Results

Errors are in rupees. CV = 5-fold cross-validation on the training data; Test = the 1,422 held-out houses.

| Model | CV RMSE | Test RMSE | Test MAE | Test R² | Test MAPE % |
|---|---|---|---|---|---|
| Baseline (average) | 3,824,884 | 3,764,102 | 2,889,043 | -0.010 | 30.5 |
| **Linear Regression** | **479,863** | **475,594** | **336,510** | **0.984** | **3.3** |
| Random Forest | 529,831 | 500,919 | 382,289 | 0.982 | 4.0 |

Random Forest best settings: `max_depth=None`, `min_samples_leaf=1`.

**Selected model: Linear Regression** (lowest cross-validated RMSE).

### Key findings

- Linear Regression reaches **R² = 0.984**, a typical error (RMSE) of **Rs 475,594**, an average error (MAE) of **Rs 336,510** and a mean percentage error of **3.3 %**. This cuts the baseline's RMSE by **87 %**.
- The Random Forest scores almost the same (R² 0.982 vs 0.984). The extra flexibility gave no gain, so the simple linear model already captures most of the signal in this data.
- **Goals:** R² ≥ 0.85 is met (0.984), MAPE ≤ 15 % is met (3.3 %), and the model clearly beats the baseline.

### Error by locality

Average percentage error of the selected model on the test set:

| Locality | Average % error |
|---|---|
| KK Nagar | 2.2 |
| Chrompet | 2.4 |
| T Nagar | 2.7 |
| Anna Nagar | 3.1 |
| Adyar | 3.2 |
| Velachery | 3.7 |
| Karapakkam | 5.1 |

Accuracy is good everywhere but varies by locality: Karapakkam has the highest error (5.1 %) and KK Nagar the lowest (2.2 %). The reasons were not investigated; fewer sales or more varied properties in a locality are possible explanations.

### Most important features

TODO: paste the "Most important features" lines from the script output here.

### Charts

The script saves its charts to `figures/`.

| | |
|---|---|
| ![Price distribution](figures/02_price_distribution.png) | ![Price vs size](figures/03_price_vs_size.png) |
| ![Price by locality](figures/04_price_by_area.png) | ![Correlation heatmap](figures/05_correlation_heatmap.png) |
| ![Predicted vs actual](figures/06_predicted_vs_actual_and_residuals.png) | ![Feature importance](figures/07_feature_importance.png) |

## Limitations and next steps

- **Unusually high accuracy.** Scores this high are uncommon for real housing data, so they should be read as specific to this dataset. Columns derived from the price were removed, but the data may still be more regular than real-world transactions.
- **Historical data.** Predictions describe past prices, not today's market.
- **Limited features.** There are no coordinates, floor level, amenities or condition details.
- **Random split.** A time-based split (train on early years, test on later ones) would be a harder, more realistic test.
- **Uneven accuracy.** Error differs by locality, so predictions are not equally reliable everywhere.
- **Next steps:** add location features and recent data, try gradient boosting, use a time-based split, and report prediction intervals.

## Repository structure

```
chennai_house_price.py   # the whole project, from raw CSV to results
requirements.txt         # Python packages
README.md
figures/                 # charts saved by the script
model_results.csv        # model comparison table (created by the script)
example_predictions.csv  # 10 example houses: inputs, actual and predicted price (created by the script)
```

## How to run

1. Install Python 3.9 or newer.
2. Download the dataset from Kaggle and put the CSV in the project folder (or in `data/`).
3. Install the packages: `pip install -r requirements.txt`
4. Run: `python chennai_house_price.py`

The script prints a labelled report (data overview, cleaning steps with counts, EDA insights, model comparison, evaluation, error by locality, top features, 10 example predictions and a goal check). It warns if a score looks too good to be true, which can indicate data leakage.

## Reproducibility

A fixed random seed (`SEED = 42`) is used for the train/test split, cross-validation and models, so results repeat. All preprocessing is inside a scikit-learn `Pipeline` and is learned from training data only.
