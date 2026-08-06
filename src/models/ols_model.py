"""
Final OLS specifications: nested models, cold/mild regime split,
and walk-forward validation. All models use Newey-West HAC errors
because volatility is autocorrelated.
"""
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

CONTROLS = ['log_lagged_vol', 'season_sin', 'season_cos', 'CDD', 'HDD', 'gas_price']
TARGET = 'log_vol'


def run_hac_ols(df, features, target=TARGET):
    """OLS with Newey-West HAC standard errors."""
    data = df[features + [target]].dropna()
    X = sm.add_constant(data[features])
    y = data[target]
    maxlags = int(0.75 * len(data) ** (1 / 3))
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})


def run_final_models(df):
    """The nested sequence that tells the story."""
    models = {
        'baseline': run_hac_ols(df, CONTROLS),
        'spread': run_hac_ols(df, CONTROLS + ['spread_24']),
        'interaction': run_hac_ols(df, CONTROLS + ['spread_24', 'spread_x_hdd']),
    }
    return models


def run_regime_split(df):
    """Cold vs mild halves split at median HDD.
    Cleanest demonstration of the regime-dependence finding."""
    hdd_median = df['HDD'].median()
    cold = df[df['HDD'] > hdd_median]
    mild = df[df['HDD'] <= hdd_median]
    return {
        'hdd_median': hdd_median,
        'cold': run_hac_ols(cold, CONTROLS + ['spread_24']),
        'mild': run_hac_ols(mild, CONTROLS + ['spread_24']),
        'n_cold': len(cold),
        'n_mild': len(mild),
    }

def run_efficiency_test(df):
    """
    Does the weather signal contain volatility information beyond what
    the day-ahead curve already implies?

    log_da_dispersion is the market's forward-looking view, priced before
    the fact. If spread_x_hdd stays significant after controlling for it,
    the market underprices weather uncertainty on cold days.

    Caveat to state explicitly: DA dispersion reflects expected peak/
    off-peak SHAPE as well as uncertainty, so it is an imperfect
    implied-vol proxy and a lower bound on market information.
    """
    base = CONTROLS + ['log_da_dispersion']
    return {
        'implied_only': run_hac_ols(df, base),
        'implied_plus_signal': run_hac_ols(
            df, base + ['spread_24', 'spread_x_hdd']),
    }


def walk_forward(df, features, coef_of_interest, target=TARGET,
                 min_train_years=2, min_test_obs=30):
    """Expanding-window walk-forward. Returns year-by-year coefficient
    and out-of-sample R^2 for the coefficient of interest."""
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    years = sorted(df['year'].unique())
    rows = []

    for i, test_year in enumerate(years):
        if i < min_train_years:
            continue
        train = df[df['year'] < test_year]
        test = df[df['year'] == test_year]
        if len(test) < min_test_obs:
            continue

        m = run_hac_ols(train, features, target)

        test_data = test[features + [target]].dropna()
        X_test = sm.add_constant(test_data[features], has_constant='add')
        preds = m.predict(X_test)
        y_test = test_data[target]
        ss_res = ((y_test - preds) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()

        rows.append({
            'test_year': test_year,
            'coef': m.params[coef_of_interest],
            'oos_r2': 1 - ss_res / ss_tot,
            'n_train': len(train),
            'n_test': len(test),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv('data/processed/merged_dataset.csv')

    print("=" * 60)
    print("NESTED MODELS")
    print("=" * 60)
    models = run_final_models(df)
    for name, m in models.items():
        print(f"{name:<14} R2={m.rsquared:.4f}  AdjR2={m.rsquared_adj:.4f}")
    print()
    print(models['interaction'].summary().tables[1])

    print("\n" + "=" * 60)
    print("REGIME SPLIT (median HDD)")
    print("=" * 60)
    split = run_regime_split(df)
    for regime in ['cold', 'mild']:
        m = split[regime]
        ci = m.conf_int().loc['spread_24']
        print(f"{regime.upper():<6} n={split['n_' + regime]}  "
              f"spread_24={m.params['spread_24']:.4f}  "
              f"p={m.pvalues['spread_24']:.4f}  "
              f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
        
    print("\n" + "=" * 60)
    print("EFFICIENCY TEST: signal vs day-ahead implied dispersion")
    print("=" * 60)
    eff = run_efficiency_test(df)
    for name, m in eff.items():
        print(f"{name:<22} R2={m.rsquared:.4f}")
    print()
    print(eff['implied_plus_signal'].summary().tables[1])

    print("\n" + "=" * 60)
    print("WALK-FORWARD: spread_x_hdd")
    print("=" * 60)
    wf = walk_forward(df, CONTROLS + ['spread_24', 'spread_x_hdd'], 'spread_x_hdd')
    print(wf.to_string(index=False))
    print(f"\nAll positive: {(wf['coef'] > 0).all()}  Mean: {wf['coef'].mean():.4f}")