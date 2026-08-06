"""
Economic value of the volatility forecast via position sizing.

Not a standalone alpha claim. The exercise: hold a constant one-day
day-ahead-to-real-time position every day (P&L per MWh = dart_premium),
then scale exposure by 1 / predicted volatility. If the forecast is
informative, risk-adjusted returns improve even with unchanged mean P&L.

Sizes are normalized to average 1.0 so the comparison isolates the
SHAPE of the sizing rule rather than any leverage effect.
"""
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

HAR_FEATURES = ['rv_d', 'rv_w', 'rv_m']
HARX_FEATURES = HAR_FEATURES + ['spread_24', 'spread_x_hdd']


def _stats(pnl, label):
    cum = pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    return {'strategy': label,
            'total_pnl': pnl.sum(),
            'mean_daily': pnl.mean(),
            'std_daily': pnl.std(),
            'sharpe': pnl.mean() / pnl.std() * np.sqrt(252),
            'max_drawdown': dd}


def run_sizing_comparison(df, min_train_years=2, min_test_obs=30):
    data = df.dropna(subset=HARX_FEATURES + ['log_vol', 'dart_premium']).copy()
    data['year'] = pd.to_datetime(data['date']).dt.year
    years = sorted(data['year'].unique())

    base, sized_har, sized_x, dates = [], [], [], []

    for i, test_year in enumerate(years):
        if i < min_train_years:
            continue
        train = data[data['year'] < test_year]
        test = data[data['year'] == test_year]
        if len(test) < min_test_obs:
            continue

        def predict(features):
            X_tr = sm.add_constant(train[features])
            m = sm.OLS(train['log_vol'], X_tr).fit()
            X_te = sm.add_constant(test[features], has_constant='add')
            v = np.exp(m.predict(X_te))
            s = 1.0 / v
            return s / s.mean()

        # Short the day-ahead risk premium (INC side): sell real-time,
        # buy day-ahead. Earns the documented premium most days, loses
        # during real-time spikes. This is a short-volatility payoff --
        # exactly the position a volatility forecast should size.
        pnl = -test['dart_premium']
        base.append(pnl)
        sized_har.append(pnl * predict(HAR_FEATURES))
        sized_x.append(pnl * predict(HARX_FEATURES))
        dates.append(test['date'])

    base = pd.concat(base).reset_index(drop=True)
    sized_har = pd.concat(sized_har).reset_index(drop=True)
    sized_x = pd.concat(sized_x).reset_index(drop=True)

    table = pd.DataFrame([
        _stats(base, 'Constant size'),
        _stats(sized_har, 'Vol-scaled (HAR-RV)'),
        _stats(sized_x, 'Vol-scaled (HAR-X, weather)'),
    ])
    return table, base, sized_har, sized_x, pd.concat(dates).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv('data/processed/merged_dataset.csv')
    table, base, s_har, s_x, dates = run_sizing_comparison(df)
    print("=" * 60)
    print("ECONOMIC VALUE: vol-scaled position sizing")
    print("=" * 60)
    print(table.round(4).to_string(index=False))