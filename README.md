# PJM Volatility Forecasting from Weather Ensemble Spread

Does weather forecast *uncertainty* predict electricity price volatility?

Weather forecasts come with a built-in measure of their own uncertainty. NOAA's GEFS model runs 31 slightly different simulations of the atmosphere at once, each starting from a slightly perturbed initial state. When all 31 agree that tomorrow will be 40°F, the forecast is confident. When they spread from 28°F to 45°F, the atmosphere is genuinely hard to predict. That disagreement — the ensemble spread — is a free, publicly available signal about how uncertain tomorrow's weather is.

Electricity demand is mostly weather. Electricity can't be stored at scale, so supply has to match demand in real time, and the supply stack is lumpy: as demand climbs, you move from cheap baseload to expensive peakers in discrete jumps. So uncertain weather should mean uncertain demand, which should mean uncertain prices.

This project tests that chain on PJM (the grid operator covering ~65 million people across the mid-Atlantic and Midwest) using real-time hourly LMPs from 2020 to 2025.

## What I found

**Ensemble spread predicts volatility, but only when it's cold.**

Splitting the sample at median heating degree days:

| | n | spread coefficient | p-value | 95% CI |
|---|---|---|---|---|
| Cold days | 810 | **0.671** | <0.0001 | [0.38, 0.96] |
| Mild days | 811 | −0.343 | 0.040 | [−0.67, −0.02] |

On cold days a 1K increase in ensemble spread is associated with roughly a 67 log-point increase in next-day realized volatility. On mild days the effect is absent (the small negative is marginal and I don't read anything causal into it).

The same thing shows up as an interaction term on the full sample. Adding `spread × HDD` to the model:

| Model | R² | Adj. R² |
|---|---|---|
| Controls only | 0.3449 | 0.3425 |
| + ensemble spread | 0.3480 | 0.3452 |
| + spread × HDD | **0.3603** | 0.3571 |

The interaction coefficient is 0.066 with t = 4.6 (HAC standard errors). The main spread effect on its own is only marginal (p ≈ 0.06) — which makes sense in hindsight, since pooling averages a strong cold-weather effect together with nothing.

**Why cold and not hot?** Heating load is steeper in temperature than cooling load, so the same forecast error moves winter demand more. Cold snaps also stress supply at the same time (gas freeze-offs, forced outages), so demand uncertainty and supply fragility arrive together. And a heat dome builds slowly and visibly while winter storm timing stays genuinely uncertain until close in — by the time a hot day arrives the market has already committed peakers and scheduled reserves, so marginal uncertainty is worth less.

I originally expected the opposite and specified the interaction with cooling degree days first. It came out significant and *negative*, which is what sent me looking at HDD.

**The effect is stronger in the tail.** Quantile regression on log volatility:

| Quantile | Coefficient | p-value | Survives bootstrap? |
|---|---|---|---|
| 0.10 | 0.230 | 0.159 | – |
| 0.25 | 0.149 | 0.226 | – |
| 0.50 | 0.344 | 0.003 | yes, CI [0.08, 0.57] |
| 0.75 | 0.210 | 0.106 | no |
| 0.90 | 0.401 | 0.016 | yes, CI [0.02, 0.70] |
| 0.95 | 0.511 | 0.024 | **no** |

Point estimates rise toward the tail, and nothing shows up on the calmest days — consistent with uncertainty only mattering once the supply stack is steep. But `quantreg` reports IID standard errors, so I bootstrapped (500 resamples). The 90th percentile survives. **The 95th does not** — its confidence interval includes zero, since only ~80 observations sit above that threshold. It has the largest point estimate in the table and I'm not claiming it.

**It holds up out of sample.** Expanding-window walk-forward, retraining on everything before each test year:

| Test year | spread × HDD coef | OOS R² | n train | n test |
|---|---|---|---|---|
| 2022 | 0.084 | 0.327 | 440 | 363 |
| 2023 | 0.080 | 0.143 | 803 | 306 |
| 2024 | 0.069 | 0.160 | 1109 | 340 |
| 2025 | 0.071 | −0.018 | 1449 | 172 |

Positive coefficient in every window. 2025 is weak for out-of-sample R² — it's a partial year (price data ends June 22) and the coefficient has been drifting down since 2022. Could be market adaptation, could be noise. Not enough data to say.

## Two things that didn't work

I'm including these because they're part of the result.

**Spike prediction: null.** Logistic regression on whether any hour exceeded $200/MWh. Adding ensemble spread to the controls gives a likelihood ratio statistic of 0.67 (p = 0.41). No improvement. This actually fits the story — the extreme spikes in PJM tend to be scarcity events driven by forced outages and reserve shortfalls, which atmospheric uncertainty has no way to anticipate. Weather uncertainty predicts *how much prices move*, not *whether a discrete scarcity event fires*.

**Gradient boosting: loses badly.** I wanted to know whether a model free to find any nonlinear structure would discover the cold-weather regime dependence on its own. LightGBM, same features, same walk-forward splits, heavy regularization (depth 3, 50 minimum samples per leaf, early stopping):

| Test year | OLS OOS R² | LightGBM OOS R² |
|---|---|---|
| 2022 | 0.327 | −0.120 |
| 2023 | 0.143 | −0.004 |
| 2024 | 0.160 | −0.026 |
| 2025 | −0.018 | −0.196 |

Negative every year — worse than predicting the test-year mean. In 2024 early stopping halted at iteration 16. With ~1,600 noisy daily observations there isn't enough signal for the trees to find stable structure, and the hand-built specification (log transform, degree days, the interaction) encodes domain knowledge the algorithm can't recover from this sample size. I didn't include SHAP plots because interpreting feature attributions from a model that doesn't generalize doesn't tell you anything.

**Also worth mentioning:** I built a set of "forecast trajectory" features — for each target date, the full path of how the temperature forecast evolved across seven lead times (total path length, trajectory volatility, near-term vs far-term revision). They had the strongest raw correlations with volatility of anything I built (path length: 0.13 vs spread's 0.075), and then went completely insignificant once controls were in. The raw correlation was seasonality, not signal.

## Data

**Weather.** GEFS ensemble 2-meter temperature via [Herbie](https://herbie.readthedocs.io), pulled from the NOAA archive on AWS. For each target date I pull all 31 members at seven lead times (24h through 168h), so every forecast describes the same target day from a different distance in the past. Ensemble spread is the standard deviation across members at each grid point, spatially averaged over the PJM footprint (35–43°N, 74–83°W). That's roughly 400,000 GRIB2 files and about a week of wall-clock time. The date range starts 2020-09-23 because that's when GEFS expanded from 21 to 31 members — spread computed from different ensemble sizes isn't comparable.

**Prices.** Real-time hourly LMPs from the [EIA Wholesale Electricity Market Portal](https://www.eia.gov/electricity/wholesalemarkets). I use `PJM Total LMP` (the system-wide load-weighted aggregate) since the weather signal is region-wide, and local Eastern dates rather than UTC since demand follows local time. Real-time rather than day-ahead because day-ahead clears the night before against the best available forecast — much of the weather uncertainty is already priced in by then. Real-time settles during actual operations, where surprises hit with no time to adjust.

**Alignment.** Features for target date T come from forecasts initialized before T (T−1 at 24h, T−7 at 168h), so there's no lookahead by construction. Weather days are UTC and price days are Eastern, which smears day boundaries by a few hours. That biases against finding a signal rather than manufacturing one, so I left it.

After merging and dropping DST days: **1,621 daily observations.**

## Method notes

A few choices that mattered more than I expected:

**Log the target.** Raw realized volatility has skew of 13.8 and kurtosis of 331 — the max is $1,100/MWh against a median of $13. OLS minimizes squared errors, so a handful of extreme days were effectively running the whole regression, and coefficients on correlated features flipped signs at random. Log transform fixed it (skew drops to 0.41) and roughly doubled the baseline R².

**Degree days, not linear temperature.** Demand is V-shaped in temperature — cold *and* hot both drive it up. A single linear temperature coefficient can't represent that, and when I used one it partly absorbed the cold-side effect while the hot-side leaked into the seasonal terms. Separate HDD and CDD terms (base 65°F) fix it.

**HAC standard errors everywhere.** Volatility is autocorrelated, which violates the OLS independence assumption and makes naive standard errors too small. Newey-West throughout.

**Sine/cosine seasonality, not dummies.** December 31 and January 1 should be nearly identical; a winter dummy creates an artificial discontinuity at the year boundary.

**Don't put seven correlated spread features in one regression.** My first attempt threw all seven lead times in together. Every coefficient came out insignificant with signs scattered (+5.4, +0.9, −10.2, −29.4, +26.5...) while the F-test was strongly significant — the classic signature of correlated features fighting over the same variance. Condition number was 24,600. I use `spread_24` as the single representative measure instead.

## Repo layout

```
src/
  data_collection/
    fetch_gefs.py      GEFS pull, spread + trajectory features, resumable
    fetch_lmp.py       EIA real-time LMPs, daily volatility metrics
  features/
    build_features.py  merge, controls, log transform, degree days, interaction
  models/
    ols_model.py       nested specs, cold/mild split, walk-forward
    logistic_model.py  spike prediction + likelihood ratio test
    ml_models.py       quantile regression, bootstrap, LightGBM comparison
  visualization/
    plots.py           all figures
notebooks/             exploration and results
config.py              all constants
```

## Reproducing

```bash
pip install -r requirements.txt

python3 src/data_collection/fetch_gefs.py    # days. resumable if it dies.
python3 src/data_collection/fetch_lmp.py     # ~1 minute
python3 src/features/build_features.py
python3 src/models/ols_model.py
python3 src/models/logistic_model.py
python3 src/models/ml_models.py
python3 src/visualization/plots.py
```

`fetch_gefs.py` writes incrementally and resumes from the last saved date, which you'll want — mine died several times over the week it took to run, once from a router problem I didn't notice for two days.

Raw and processed data aren't committed (large, and reproducible from the fetch scripts).

## Things I'd do next

- **Population-weight the spatial average.** Right now every grid point counts equally, so rural West Virginia weighs as much as Philadelphia. Demand follows people.
- **Separate demand-driven from scarcity-driven spikes.** The spike-prediction null suggests these are different animals. Identifying reserve-shortage intervals and modeling them separately would test that directly.
- **Probabilistic forecasting with CRPS.** Instead of predicting a point estimate, predict the whole distribution and let ensemble spread control its width. This is standard in weather verification and rare in electricity price forecasting, and it's the natural next step for the tail result.
- **Zonal rather than system-wide prices.** The PJM aggregate smooths out localized congestion, which is where a lot of the interesting volatility lives.

---

Built by [Adam Zachry](https://www.adamzachry.com). Questions and corrections welcome.