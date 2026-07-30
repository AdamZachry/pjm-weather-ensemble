"""
All final figures. Each function saves to results/figures/.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap #nope

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

FIGDIR = 'results/figures'
os.makedirs(FIGDIR, exist_ok=True)


def plot_regime_comparison(split):
    """Figure 1: spread_24 coefficient, cold vs mild days.
    Input: dict from ols_model.run_regime_split."""
    fig, ax = plt.subplots(figsize=(7, 5))
    names, coefs, errs = [], [], []
    for regime in ['cold', 'mild']:
        m = split[regime]
        ci = m.conf_int().loc['spread_24']
        names.append(f"{regime.capitalize()} days\n(n={split['n_' + regime]})")
        coefs.append(m.params['spread_24'])
        errs.append((ci[1] - ci[0]) / 2)
    colors = ['#2166ac', '#d6604d']
    ax.bar(names, coefs, yerr=errs, capsize=8, color=colors, alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('spread_24 coefficient (log volatility)')
    ax.set_title('Ensemble spread predicts volatility only on cold days\n'
                 '(split at median HDD, 95% HAC confidence intervals)')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig1_regime_comparison.png', dpi=200)
    plt.close()


def plot_quantile_coefficients(qtable, boot_results=None):
    """Figure 2: spread_24 coefficient across quantiles."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qtable['quantile'], qtable['spread_24_coef'],
            marker='o', color='#2166ac', linewidth=2)
    if boot_results is not None:
        for r in boot_results:
            ax.errorbar(r['quantile'], r['coef_mean'],
                        yerr=[[r['coef_mean'] - r['ci_low']],
                              [r['ci_high'] - r['coef_mean']]],
                        fmt='s', color='#d6604d', capsize=6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Quantile of log realized volatility')
    ax.set_ylabel('spread_24 coefficient')
    ax.set_title('The spread effect strengthens toward the volatility tail\n'
                 '(red: bootstrap 95% CIs)')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig2_quantile_coefficients.png', dpi=200)
    plt.close()


def plot_walkforward_stability(wf, coef_name='spread_x_hdd'):
    """Figure 3: interaction coefficient by out-of-sample year."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(wf['test_year'].astype(str), wf['coef'], color='#2166ac', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel(f'{coef_name} coefficient (training window)')
    ax.set_title('The spread-HDD interaction is positive in every '
                 'out-of-sample year')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig3_walkforward_stability.png', dpi=200)
    plt.close()


def plot_timeseries_overlay(df, model, features, target='log_vol'):
    """Figure 4: predicted vs realized log volatility over time."""
    import statsmodels.api as sm
    data = df.dropna(subset=features + [target]).copy()
    X = sm.add_constant(data[features])
    data['pred'] = model.predict(X)
    data['date'] = pd.to_datetime(data['date'])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data['date'], data[target], label='Realized', alpha=0.6, linewidth=0.8)
    ax.plot(data['date'], data['pred'], label='Predicted', alpha=0.8, linewidth=0.8)
    ax.set_ylabel('log realized volatility')
    ax.legend()
    ax.set_title('Predicted vs realized log volatility (final specification)')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig4_timeseries_overlay.png', dpi=200)
    plt.close()

def plot_oos_comparison(ols_wf, gbm_wf):
    """Figure 5: out-of-sample R² by year, OLS vs LightGBM.
    The theory-driven specification generalizes; the GBM overfits."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ols_wf))
    width = 0.38
    ax.bar(x - width/2, ols_wf['oos_r2'], width,
           label='OLS (spread × HDD)', color='#2166ac', alpha=0.85)
    ax.bar(x + width/2, gbm_wf['oos_r2'], width,
           label='LightGBM', color='#d6604d', alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ols_wf['test_year'].astype(str))
    ax.set_ylabel('Out-of-sample $R^2$')
    ax.set_xlabel('Test year')
    ax.set_title('Theory-driven specification generalizes better than\n'
                 'gradient boosting at this sample size')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig5_oos_comparison.png', dpi=200)
    plt.close()

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models.ols_model import (run_final_models, run_regime_split,
                                  walk_forward, CONTROLS)
    from models.ml_models import (run_quantile_models, bootstrap_quantile)

    df = pd.read_csv('data/processed/merged_dataset.csv')

    print("Fig 1: regime comparison...")
    plot_regime_comparison(run_regime_split(df))

    print("Fig 2: quantile coefficients (bootstrap takes a few minutes)...")
    qtable = run_quantile_models(df)
    boots = [bootstrap_quantile(df, q) for q in [0.50, 0.90]]
    plot_quantile_coefficients(qtable, boots)

    print("Fig 3: walk-forward stability...")
    wf = walk_forward(df, CONTROLS + ['spread_24', 'spread_x_hdd'], 'spread_x_hdd')
    plot_walkforward_stability(wf)

    print("Fig 4: time series overlay...")
    models = run_final_models(df)
    plot_timeseries_overlay(df, models['interaction'],
                            CONTROLS + ['spread_24', 'spread_x_hdd'])

    print("Fig 5: OLS vs GBM out-of-sample comparison...")
    from models.ml_models import run_gradient_boosting
    gbm_wf, _, _ = run_gradient_boosting(df)
    plot_oos_comparison(wf, gbm_wf)

    print("All figures saved to results/figures/")