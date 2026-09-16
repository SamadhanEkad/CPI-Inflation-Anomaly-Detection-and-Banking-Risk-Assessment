# CPI Inflation Anomaly Detection & Banking Risk Assessment

Detecting abnormal inflation periods in Indian CPI data (2014–2024) and translating them into a single, interpretable banking risk score using five anomaly-detection methods, a custom risk index, and XGBoost forecasting.

![CBRI Timeline](images/cbri_timeline.png)

## Problem

Banks need early warning when inflation is behaving abnormally — not just when it's high, but when it's *statistically unusual* relative to its own history. A single detection method (say, just flagging high CPI values) misses different kinds of shocks: sudden level jumps, seasonal breaks, volatility clustering, and multivariate pattern shifts all look different mathematically. This project combines five distinct anomaly-detection lenses into one risk index and tests whether that index actually improves forecasting.

## Approach

**Data**: 274,674 monthly records of Indian CPI across states, sectors, and consumption groups (2014–2024), sourced at state/sector/group granularity. 38.4% of records were flagged with an inflation-shock indicator during preprocessing.

**Pipeline**:
1. **Feature engineering & cleaning** — CPI level, MoM/YoY change, rolling volatility, lag features
2. **Feature selection** — Variance threshold → correlation filter → PCA check, narrowing 8 candidate features down to 4 (`Inflation_rate`, `CPI_MoM_change`, `CPI_rolling_std_6m`, `CPI_lag_6`)
3. **Anomaly detection — 5 independent methods**:
   - Z-Score (statistical baseline)
   - ARIMA residuals (autoregressive shocks)
   - SARIMA residuals (seasonal + AR shocks)
   - GARCH volatility (volatility clustering)
   - Isolation Forest (multivariate pattern anomalies)
4. **Combined Banking Risk Index (CBRI)** — a weighted blend of all five methods (GARCH weighted highest at 0.35, since volatility clustering matters most for banking exposure)
5. **Risk tier classification** — Normal / Low / Medium / High / Critical, thresholded on CBRI mean + standard deviation
6. **Forecasting** — XGBoost with and without CBRI as a feature, benchmarked against ARIMA and SARIMA

## Results

### Each method catches different anomalies

![Anomaly Heatmap](images/anomaly_heatmap.png)

| Method | Anomalies Detected | What it uniquely catches |
|---|---|---|
| Z-Score | 4 | Level deviation baseline |
| ARIMA Residuals | 5 | AR structure shocks |
| SARIMA Residuals | 6 | Seasonal + AR shocks |
| GARCH Volatility | 3 | Volatility clustering |
| Isolation Forest | 10 | Multivariate patterns |

No single method dominates — this is the justification for combining them rather than picking one.

### Risk tiers map to real banking-relevant events

![Risk Tier Distribution](images/risk_tier_distribution.png)

69.2% of the 10-year period was classified Normal. The High/Critical periods (17.5% combined) line up with known events:

| Period | Tier | Context |
|---|---|---|
| Jan 2015 | Critical | Global commodity price collapse |
| Jan 2016 | High | Demonetisation — liquidity disruption |
| Jan 2017 | High | GST implementation — supply chain distortion |
| Jan–Feb, Dec 2019 | Critical/High | NBFC crisis — elevated credit risk |
| Aug, Oct, Nov 2020 | High | COVID-19 — demand collapse, NPA risk |
| May 2021 | Critical | Post-COVID input cost surge |
| Jul–Oct 2023 | Critical | Monetary tightening cycle |
| Jul–Aug 2024 | Critical/High | Persistent core inflation |

### Forecast accuracy: XGBoost beats classical time series, but CBRI didn't help this split

![Model Comparison](images/model_comparison.png)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| ARIMA(1,1,1) | 1.0003 | 1.2072 | 19.05% |
| SARIMA(1,1,1)(1,1,1,12) | 0.9796 | 1.2293 | 18.32% |
| **XGBoost — Baseline** | **0.7378** | **1.0443** | **14.37%** |
| XGBoost + CBRI | 0.7771 | 1.0630 | 15.04% |

XGBoost substantially outperforms both classical statistical models. Adding CBRI as a feature to XGBoost did **not** improve forecast accuracy on this test split (MAE worsened ~5.3%) — CBRI ranked 5th of 7 features by importance. This is reported honestly rather than hidden: CBRI's value here is as an **interpretable risk-monitoring signal**, not as a forecasting feature. A likely next step is tuning the Isolation Forest contamination rate, which currently drives a large share of CBRI's variance.

### 12-month volatility outlook

GARCH(1,1) forecasts volatility staying elevated (~0.83–0.85) through 2025, with moderate shock persistence (α+β = 0.428) — meaning shocks propagate but dissipate relatively quickly, so short-term monitoring is sufficient for most CPI-linked exposures.

## 📊 Presentation

[Download Project Presentation](CPI_Anomaly_Banking_Risk_Presentation_Updated.pptx)

## Tech Stack

Python · pandas · NumPy · statsmodels (ARIMA/SARIMA) · arch (GARCH) · scikit-learn (Isolation Forest, feature selection) · XGBoost · matplotlib · seaborn

## Repository Structure

```
├── cpi_anomaly_detection.ipynb   # Full analysis notebook (6 phases, 20 sections)
├── data/
│   └── CPI.xlsx                  # Source dataset
├── images/                        # Charts used in this README
├── CPI_Anomaly_Banking_Risk_Presentation_Updated.pptx
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

```bash
git clone https://github.com/SamadhanEkad/CPI-Inflation-Anomaly-Detection-and-Banking-Risk-Assessment.git
cd CPI-Inflation-Anomaly-Detection-and-Banking-Risk-Assessment
pip install -r requirements.txt
jupyter notebook cpi_anomaly_detection.ipynb
```

## Limitations & Next Steps

- CBRI weights (0.1 / 0.15 / 0.15 / 0.35 / 0.25) are set manually, not learned — a natural extension is optimizing them against a labeled risk-event dataset
- Isolation Forest contamination rate is a likely lever for improving CBRI's contribution to forecasting
- Test set is 24 months — a longer out-of-sample window would give more confidence in the model comparison

## License

MIT — see [LICENSE](LICENSE)
