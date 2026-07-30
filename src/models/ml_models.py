"""
Quantile regression (with bootstrap SEs) and gradient boosting + SHAP.

Quantile: tests whether spread predicts the tail of the volatility
distribution more strongly than the center.
Gradient boosting: tests whether an algorithm free to find any
nonlinear structure discovers more than the hand-built HDD interaction.
"""
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shap
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning
warnings.filterwarnings('ignore', category=IterationLimitWarning)

FORMULA = ('log_vol ~ log_lagged_vol + season_sin + season_cos + '
           'CDD + HDD + gas_price + spread_24')
GBM_FEATURES = ['log_lagged_vol', 'season_sin', 'season_cos',
                'CDD', 'HDD', 'gas_price', 'spread_24']
TARGET = 'log_vol'


def run_quantile_models(df, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90, 0.95)):
    data = df.dropna(subset=['log_vol', 'log_lagged_vol', 'spread_24'])
    rows = []
    for q in quantiles:
        m = smf.quantreg(FORMULA, data).fit(q=q)
        rows.append({'quantile': q,
                     'spread_24_coef': m.params['spread_24'],
                     'pvalue': m.pvalues['spread_24']})
    return pd.DataFrame(rows)


def bootstrap_quantile(df, q, n_boot=500, seed=42):
    """Bootstrap SE and CI for the spread_24 coefficient at quantile q.
    quantreg's analytic p-values assume IID errors; bootstrap is the
    honest error estimate here."""
    data = df.dropna(subset=['log_vol', 'log_lagged_vol', 'spread_24']).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(data)
    coefs = []
    for _ in range(n_boot):
        sample = data.iloc[rng.integers(0, n, n)]
        try:
            m = smf.quantreg(FORMULA, sample).fit(q=q)
            coefs.append(m.params['spread_24'])
        except Exception:
            continue
    coefs = np.array(coefs)
    ci = np.percentile(coefs, [2.5, 97.5])
    return {'quantile': q, 'coef_mean': coefs.mean(), 'boot_se': coefs.std(),
            'ci_low': ci[0], 'ci_high': ci[1],
            'significant': bool(ci[0] > 0 or ci[1] < 0)}


def run_gradient_boosting(df, min_train_years=2, min_test_obs=30):
    """LightGBM on log_vol, walk-forward (never random k-fold on time
    series). Heavy regularization: ~1,600 rows will overfit otherwise.
    NOTE: no interaction feature is provided — the test is whether the
    trees discover the spread-HDD structure on their own."""
    import lightgbm as lgb

    data = df.dropna(subset=GBM_FEATURES + [TARGET]).copy()
    data['year'] = pd.to_datetime(data['date']).dt.year
    years = sorted(data['year'].unique())

    params = dict(
        objective='regression',
        max_depth=3,
        num_leaves=7,
        learning_rate=0.05,
        n_estimators=2000,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        verbose=-1,
    )

    rows, models = [], {}
    for i, test_year in enumerate(years):
        if i < min_train_years:
            continue
        train = data[data['year'] < test_year]
        test = data[data['year'] == test_year]
        if len(test) < min_test_obs:
            continue

        # last 20% of train (chronologically) as early-stopping validation
        cut = int(len(train) * 0.8)
        tr, val = train.iloc[:cut], train.iloc[cut:]

        model = lgb.LGBMRegressor(**params)
        model.fit(tr[GBM_FEATURES], tr[TARGET],
                  eval_set=[(val[GBM_FEATURES], val[TARGET])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        preds = model.predict(test[GBM_FEATURES])
        y = test[TARGET]
        oos_r2 = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({'test_year': test_year, 'oos_r2': oos_r2,
                     'n_train': len(train), 'n_test': len(test),
                     'best_iter': model.best_iteration_})
        models[test_year] = model

    return pd.DataFrame(rows), models, data


def compute_shap(model, data):
    """SHAP values for a fitted LightGBM model."""
    explainer = shap.TreeExplainer(model)
    X = data[GBM_FEATURES]
    return explainer.shap_values(X), X


if __name__ == "__main__":
    df = pd.read_csv('data/processed/merged_dataset.csv')

    print("=" * 60)
    print("QUANTILE REGRESSION")
    print("=" * 60)
    print(run_quantile_models(df).round(4).to_string(index=False))

    print("\nBootstrap (500 resamples) — this takes a few minutes:")
    for q in [0.50, 0.90]:
        r = bootstrap_quantile(df, q)
        print(f"q={r['quantile']}: coef={r['coef_mean']:.4f} "
              f"CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
              f"sig={r['significant']}")

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING (walk-forward)")
    print("=" * 60)
    wf, models, data = run_gradient_boosting(df)
    print(wf.to_string(index=False))
    print(f"\nMean OOS R2: {wf['oos_r2'].mean():.4f}")