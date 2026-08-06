"""
HAR-RV benchmark and forecast comparison.

HAR-RV (Corsi 2009) is the standard benchmark for realized volatility.
GARCH is deliberately NOT used: it infers latent volatility from daily
returns, while realized volatility is directly observable from hourly
prices. The electricity volatility literature finds HAR-type models
dominate GARCH-type models for this target.

Comparison uses QLIKE loss rather than MSE. MSE over-penalizes errors
on high-volatility days -- precisely the days this signal targets --
so it would flatter whichever model is most conservative.
"""
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

HAR_FEATURES = ['rv_d', 'rv_w', 'rv_m']
HARX_FEATURES = HAR_FEATURES + ['spread_24', 'spread_x_hdd']
TARGET = 'log_vol'


def qlike_loss(actual_vol, pred_vol):
    """
    QLIKE on the volatility level (not log). Standard loss for
    volatility forecast evaluation. Requires strictly positive inputs.
    """
    r = actual_vol / pred_vol
    return r - np.log(r) - 1


def diebold_mariano(loss_a, loss_b, h=1):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample
    correction. Negative statistic means model A has lower loss.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dbar = d.mean()
    gamma0 = np.sum((d - dbar) ** 2) / n
    var_d = gamma0 / n
    dm = dbar / np.sqrt(var_d)
    # HLN correction for small samples
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * correction
    p = 2 * (1 - norm.cdf(abs(dm_star)))
    return dm_star, p


def walk_forward_forecasts(df, features, target=TARGET,
                           min_train_years=2, min_test_obs=30):
    """Expanding-window forecasts. Returns actual and predicted vol levels."""
    data = df.dropna(subset=features + [target]).copy()
    data['year'] = pd.to_datetime(data['date']).dt.year
    years = sorted(data['year'].unique())

    actual, pred, dates, yrs = [], [], [], []
    for i, test_year in enumerate(years):
        if i < min_train_years:
            continue
        train = data[data['year'] < test_year]
        test = data[data['year'] == test_year]
        if len(test) < min_test_obs:
            continue

        X_tr = sm.add_constant(train[features])
        m = sm.OLS(train[target], X_tr).fit()
        X_te = sm.add_constant(test[features], has_constant='add')

        # Retransformation correction. exp(E[log X]) is the conditional
        # MEDIAN, not the mean, so naive exponentiation biases forecasts
        # low. Under log-normal errors, adding sigma^2/2 recovers the mean.
        sigma2 = m.mse_resid

        actual.append(np.exp(test[target].values))
        pred.append(np.exp(m.predict(X_te).values + sigma2 / 2))
        dates.append(test['date'].values)
        yrs.append(np.repeat(test_year, len(test)))

    return (np.concatenate(actual), np.concatenate(pred),
            np.concatenate(dates), np.concatenate(yrs))


def compare_har_models(df):
    """HAR-RV vs HAR-X (HAR plus the weather uncertainty signal)."""
    a_har, p_har, dates, yrs = walk_forward_forecasts(df, HAR_FEATURES)
    a_x, p_x, _, _ = walk_forward_forecasts(df, HARX_FEATURES)

    loss_har = qlike_loss(a_har, p_har)
    loss_x = qlike_loss(a_x, p_x)

    dm, pval = diebold_mariano(loss_x, loss_har)

    def oos_r2(a, p):
        return 1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()

    per_year = pd.DataFrame({'year': yrs,
                             'qlike_har': loss_har,
                             'qlike_harx': loss_x}).groupby('year').mean()

    return {
        'qlike_har': loss_har.mean(),
        'qlike_harx': loss_x.mean(),
        'qlike_improvement_pct': 100 * (1 - loss_x.mean() / loss_har.mean()),
        'dm_stat': dm,
        'dm_pvalue': pval,
        'oos_r2_har': oos_r2(a_har, p_har),
        'oos_r2_harx': oos_r2(a_x, p_x),
        'per_year': per_year,
        'actual': a_har, 'pred_har': p_har, 'pred_harx': p_x, 'dates': dates,
    }


def mincer_zarnowitz(actual, pred):
    """
    Unbiasedness test: regress actual on predicted. A well-calibrated
    forecast has intercept 0 and slope 1. Tested jointly.
    """
    X = sm.add_constant(pred)
    m = sm.OLS(actual, X).fit(cov_type='HAC', cov_kwds={'maxlags': 8})
    f_test = m.f_test("const = 0, x1 = 1")
    return {'intercept': m.params[0], 'slope': m.params[1],
            'joint_pvalue': float(f_test.pvalue)}


if __name__ == "__main__":
    df = pd.read_csv('data/processed/merged_dataset.csv')
    res = compare_har_models(df)

    print("=" * 60)
    print("HAR-RV BENCHMARK vs HAR-X (with weather signal)")
    print("=" * 60)
    print(f"Mean QLIKE, HAR-RV : {res['qlike_har']:.4f}")
    print(f"Mean QLIKE, HAR-X  : {res['qlike_harx']:.4f}")
    print(f"Improvement        : {res['qlike_improvement_pct']:.2f}%")
    print(f"OOS R2, HAR-RV     : {res['oos_r2_har']:.4f}")
    print(f"OOS R2, HAR-X      : {res['oos_r2_harx']:.4f}")
    print(f"\nDiebold-Mariano    : {res['dm_stat']:.4f}  (p = {res['dm_pvalue']:.4f})")
    print("(negative statistic = HAR-X has lower loss)")
    print("\nPer-year mean QLIKE:")
    print(res['per_year'].round(4).to_string())

    print("\nMincer-Zarnowitz (HAR-X):")
    mz = mincer_zarnowitz(res['actual'], res['pred_harx'])
    print(f"  intercept={mz['intercept']:.4f}, slope={mz['slope']:.4f}, "
          f"joint p={mz['joint_pvalue']:.4f}")