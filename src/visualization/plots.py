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

def plot_qlike_by_year(per_year):
    """Figure 6: QLIKE loss by year, HAR-RV vs HAR-X. Lower is better."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(per_year))
    w = 0.38
    ax.bar(x - w/2, per_year['qlike_har'], w,
           label='HAR-RV', color='#d6604d', alpha=0.85)
    ax.bar(x + w/2, per_year['qlike_harx'], w,
           label='HAR-X (with weather)', color='#2166ac', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(per_year.index.astype(str))
    ax.set_ylabel('Mean QLIKE loss (lower is better)')
    ax.set_xlabel('Test year')
    ax.set_title('Volatility forecast accuracy vs the HAR-RV benchmark')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig6_qlike_by_year.png', dpi=200)
    plt.close()


def plot_forecast_vs_realized(res):
    """Figure 7: HAR-X forecast against realized volatility."""
    fig, ax = plt.subplots(figsize=(13, 5))
    d = pd.to_datetime(res['dates'])
    ax.plot(d, res['actual'], label='Realized', alpha=0.55, linewidth=0.8)
    ax.plot(d, res['pred_harx'], label='HAR-X forecast',
            alpha=0.9, linewidth=1.0)
    ax.set_ylabel('Realized volatility ($/MWh)')
    ax.set_title('Out-of-sample volatility forecast vs realized')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig7_forecast_vs_realized.png', dpi=200)
    plt.close()


def plot_sizing_equity(base, sized_har, sized_x, dates):
    """Figure 8: cumulative P&L under each sizing rule."""
    fig, ax = plt.subplots(figsize=(12, 5))
    d = pd.to_datetime(dates)
    ax.plot(d, base.cumsum(), label='Constant size',
            color='#999999', linewidth=1.2)
    ax.plot(d, sized_har.cumsum(), label='Vol-scaled (HAR-RV)',
            color='#d6604d', linewidth=1.2)
    ax.plot(d, sized_x.cumsum(), label='Vol-scaled (HAR-X)',
            color='#2166ac', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Cumulative P&L ($/MWh, normalized size)')
    ax.set_title('Position sizing on the volatility forecast')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig8_sizing_equity.png', dpi=200)
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

    print("Figs 6-8: HAR benchmark and economic value...")
    from models.har_benchmark import compare_har_models
    from models.economic_value import run_sizing_comparison
    res = compare_har_models(df)
    plot_qlike_by_year(res['per_year'])
    plot_forecast_vs_realized(res)
    table, base, s_har, s_x, sz_dates = run_sizing_comparison(df)
    plot_sizing_equity(base, s_har, s_x, sz_dates)

    print("All figures saved to results/figures/")