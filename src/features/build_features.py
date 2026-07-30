"""
Merge weather and price datasets, add control variables and
final derived features, produce analysis-ready dataset.
"""
import os
import sys
import pandas as pd
import numpy as np
import pandas_datareader as pdr

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


def load_and_merge():
    spread = pd.read_csv('data/processed/ensemble_spread.csv')
    lmp = pd.read_csv('data/processed/lmp_daily.csv')

    spread['date'] = pd.to_datetime(spread['date'])
    lmp['date'] = pd.to_datetime(lmp['date'])

    # Inner join on date. No shifting needed: spread features for date T
    # were computed from forecasts initialized BEFORE T (T-1 for fxx=24,
    # T-7 for fxx=168), so there is no lookahead bias by construction.
    df = pd.merge(spread, lmp, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df


def add_controls(df):
    # 1. lagged realized vol — persistence control
    df['lagged_vol'] = df['realized_vol'].shift(1)

    # 2. day of week
    df['day_of_week'] = df['date'].dt.dayofweek

    # 3. smooth season encoding (sin + cos of day-of-year)
    doy = df['date'].dt.dayofyear
    df['season_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['season_cos'] = np.cos(2 * np.pi * doy / 365.25)

    # 4. Henry Hub gas price from FRED (business days only -> ffill)
    gas = pdr.get_data_fred('DHHNGSP',
                            start=df['date'].min(),
                            end=df['date'].max())
    gas = gas.reset_index()
    gas.columns = ['date', 'gas_price']
    df = pd.merge(df, gas, on='date', how='left')
    df['gas_price'] = df['gas_price'].ffill()

    return df


def add_derived_features(df):
    """Final-specification variables identified during modeling."""
    # log target and log lag: realized vol is right-skewed (Uri max ~1100
    # vs median ~13); OLS on raw vol is dominated by a handful of extreme
    # days. min vol is ~1.4 so plain log is safe.
    df['log_vol'] = np.log(df['realized_vol'])
    df['log_lagged_vol'] = np.log(df['lagged_vol'])

    # degree days: demand is V-shaped in temperature; a linear temp term
    # cannot represent heating AND cooling response. Base 291.5K (~65F).
    df['CDD'] = np.maximum(df['temp_24'] - 291.5, 0)
    df['HDD'] = np.maximum(291.5 - df['temp_24'], 0)

    # demeaned spread x HDD interaction — the headline specification.
    # Demeaning before multiplying keeps the interaction orthogonal to
    # the main effects (avoids mechanical multicollinearity).
    df['spread_dm'] = df['spread_24'] - df['spread_24'].mean()
    df['HDD_dm'] = df['HDD'] - df['HDD'].mean()
    df['spread_x_hdd'] = df['spread_dm'] * df['HDD_dm']

    return df


def build_features():
    df = load_and_merge()
    df = add_controls(df)
    df = add_derived_features(df)

    print("Missing values before cleaning:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    df = df.dropna().reset_index(drop=True)

    df.to_csv('data/processed/merged_dataset.csv', index=False)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    build_features()