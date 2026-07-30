"""
Spike prediction: logistic regression with likelihood ratio test.
Result is a null (spread does not improve spike_200 prediction beyond
fundamentals) — reported as part of the story: spikes are driven by
discrete events (outages, scarcity pricing) that atmospheric
uncertainty alone cannot anticipate.
"""
import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

CONTROLS = ['log_lagged_vol', 'season_sin', 'season_cos', 'CDD', 'HDD', 'gas_price']


def likelihood_ratio_test(m_full, m_restricted):
    stat = 2 * (m_full.llf - m_restricted.llf)
    df_diff = m_full.df_model - m_restricted.df_model
    p_value = 1 - chi2.cdf(stat, df=df_diff)
    return stat, p_value


def run_spike_logistic(df, spike_col='spike_200'):
    data = df.dropna(subset=CONTROLS + [spike_col, 'spread_24'])

    X0 = sm.add_constant(data[CONTROLS])
    m0 = sm.Logit(data[spike_col], X0).fit(disp=0)

    X1 = sm.add_constant(data[CONTROLS + ['spread_24']])
    m1 = sm.Logit(data[spike_col], X1).fit(disp=0)

    lr_stat, p_value = likelihood_ratio_test(m1, m0)
    return {'baseline': m0, 'full': m1, 'lr_stat': lr_stat, 'lr_pvalue': p_value}


if __name__ == "__main__":
    df = pd.read_csv('data/processed/merged_dataset.csv')
    res = run_spike_logistic(df)
    print(f"LR statistic: {res['lr_stat']:.3f},  p-value: {res['lr_pvalue']:.4f}")
    print(f"spread_24 coef: {res['full'].params['spread_24']:.4f}  "
          f"(p={res['full'].pvalues['spread_24']:.4f}, "
          f"odds ratio={np.exp(res['full'].params['spread_24']):.3f})")