# Turbine Energy Yield (TEY) Prediction using Machine Learning

Predicting the energy output of a gas turbine in a cogeneration (CHP) plant from multivariate time-series sensor data, using a progression of models from linear baselines to boosted-tree ensembles.

**Course:** CL653 — Applications of AI and ML for Chemical Engineering, IIT Guwahati
**Author:** Aman Kumar (230107012)

---

## Overview

Cogeneration plants produce electricity and thermal energy together, and gas turbine performance depends heavily on environmental and operational conditions. This project builds a predictive model for **Turbine Energy Yield (TEY)** from ambient and operational sensor readings, with the goal of supporting better fuel efficiency, emissions control, and operational decision-making.

## Problem Statement

Design a reliable predictive model for TEY using multivariate time-series sensor data, and compare classical regression, deep learning, and gradient-boosting approaches to identify which best captures the nonlinear thermodynamic relationships in turbine operation.

## Dataset

- **Source:** [UCI ML Repository — Gas Turbine CO and NOx Emission Data Set](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
- 5 years of hourly sensor readings (2011–2015), concatenated into a single time-ordered dataset
- **Features:** Ambient Temperature (AT), Ambient Pressure (AP), Ambient Humidity (AH), Air Filter Difference Pressure (AFDP), Gas Turbine Exhaust Pressure (GTEP), Turbine Inlet Temperature (TIT), Turbine After Temperature (TAT), Compressor Discharge Pressure (CDP), NOx, CO
- **Target:** TEY (Turbine Energy Yield)

## Methodology

**1. EDA & Preprocessing**
- Per-year distribution plots and a combined multi-line time-series view across all 5 years
- Log-transform applied to CO to correct skew
- Correlation heatmap against TEY; `GTEP` and `TIT` dropped based on the correlation analysis
- Feature scaling via MinMax/Standard scaling ahead of modeling

**2. Models implemented**
| Category | Models |
|---|---|
| Linear baselines | Linear Regression, Linear Regression + PCA, Lasso, Ridge |
| Deep learning | LSTM, GRU (10-step sequence windows) |
| Gradient boosting | XGBoost, LightGBM, CatBoost |
| Statistical baseline | ARIMA(2,1,2), univariate |
| Combination methods | Stacking (meta linear regression), weighted ensembling of boosted models |

**3. Evaluation**
- Linear models, boosting models, and ARIMA evaluated via 10-fold `TimeSeriesSplit` cross-validation (train on past, test on future)
- LSTM/GRU evaluated on a single chronological 80/20 train-test split
- Boosting models additionally evaluated on a final 90/10 chronological holdout, then combined via stacking and weighted ensembling
- Metrics: MSE and R²; actual-vs-predicted plots for visual inspection

## Results

**10-fold TimeSeriesSplit cross-validation (average across folds):**

| Model | Avg. MSE | Avg. R² |
|---|---|---|
| Linear Regression | 76.83 | 0.673 |
| Linear Regression + PCA | 74.26 | 0.682 |
| Lasso Regression | 81.30 | 0.661 |
| Ridge Regression | 76.78 | 0.673 |
| LSTM* | 128.78 | 0.507 |
| GRU* | 89.14 | 0.659 |
| XGBoost | 42.26 | 0.818 |
| LightGBM | 42.76 | 0.816 |
| CatBoost | 44.13 | 0.815 |
| ARIMA(2,1,2) | 327.46 | −0.502 |

*LSTM/GRU numbers come from a single 80/20 split, not the 10-fold CV used for the other models — not directly comparable on an apples-to-apples basis.*

**Final chronological holdout (90/10 split) — boosting models and combinations:**

| Model | R² | MSE |
|---|---|---|
| XGBoost | 0.905 | — |
| CatBoost | 0.902 | — |
| LightGBM | 0.911 | — |
| Stacked (meta linear regression) | 0.901 | 22.17 |
| Weighted ensemble (0.6 LightGBM + 0.4 CatBoost) | **0.912** | **19.81** |

## Key Findings

- Gradient-boosting models (XGBoost, LightGBM, CatBoost) substantially outperformed both linear baselines and the LSTM/GRU sequence models on this dataset.
- A weighted ensemble of LightGBM and CatBoost gave the best overall result (R² = 0.912 on the final holdout), narrowly ahead of any single boosting model.
- ARIMA, as a univariate statistical baseline, failed to capture the multivariate nonlinear relationships and produced a negative R².
- Deep learning models required more tuning and compute for comparatively weaker results than the tree-based ensembles here.

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · TensorFlow/Keras · XGBoost · LightGBM · CatBoost · Statsmodels (ARIMA)

## Project Structure

```
├── 230107012_final_code.ipynb   # Full analysis and modeling pipeline
├── 230107012_AI_Final_Report.docx  # Written project report
├── data/                        # gt_2011.csv ... gt_2015.csv (UCI dataset, not included)
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno tensorflow xgboost lightgbm catboost statsmodels
```

Download the 5 yearly CSVs from the UCI dataset link above into a `data/` folder, then run the notebook top to bottom.

## References

- [UCI Machine Learning Repository — Gas Turbine CO and NOx Emission Data Set](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
