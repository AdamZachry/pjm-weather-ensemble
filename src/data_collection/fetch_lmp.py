"""
Fetch real-time hourly LMPs from PJM Data Miner 2,
compute daily volatility metrics.
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

def fetch_lmp_year(year, market='rt'):
    """
    Fetch one year of PJM hourly LMPs from EIA.
    market: 'rt' for real-time, 'da' for day-ahead.
    """
    url = (f"https://www.eia.gov/electricity/wholesalemarkets/csv/"
           f"pjm_lmp_{market}_hr_zones_{year}.csv")
    df = pd.read_csv(url, skiprows=3)
    df = df[['Local Date', 'Hour Number', 'PJM Total LMP']]
    df.columns = ['date', 'hour', 'lmp']
    return df

def daily_metrics(group):
    """
    
    """
    prices = group['lmp']
    daily_mean = prices.mean()
    upside = prices[prices > daily_mean]
    return pd.Series({
        'realized_vol': prices.std(),
        'mean_price': daily_mean,
        'price_range': prices.max() - prices.min(),
        'upside_variance': upside.std() if len(upside) > 0 else 0.0,
        'spike_100': int((prices > 100).any()),
        'spike_200': int((prices > 200).any()),
        'spike_500': int((prices > 500).any()),
        'hour_count': len(prices)
    })

def compute_daily_metrics(hourly_df):
    """
    
    """
    return hourly_df.groupby('date').apply(daily_metrics, include_groups=False).reset_index()

def build_lmp_dataset(start_year, end_year):
    """
    
    """
    all_years = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        hourly = fetch_lmp_year(year)
        daily = compute_daily_metrics(hourly)
        all_years.append(daily)

    df = pd.concat(all_years, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # drop DST days (23 or 25 hours) — only 2 per year, keeps metrics comparable
    df = df[df['hour_count'] == 24]

    # filter to match weather dataset range
    df = df[(df['date'] >= '2020-09-23') & (df['date'] <= '2025-12-31')]

    df.to_csv('data/processed/lmp_daily.csv', index=False)
    print(f"Saved {len(df)} rows")
    return df

def build_da_features(start_year, end_year):
    """
    Day-ahead features.

    da_dispersion: std dev of the 24 day-ahead hourly prices. This is the
    market's own forward-looking view of tomorrow's price variation, priced
    the day before -- an imperfect but usable implied-volatility proxy.
    It measures expected SHAPE (peak vs off-peak) as well as uncertainty,
    so it is a lower bound on what the market knew. State this limitation.

    dart_premium: mean(RT) - mean(DA). Per-MWh P&L of holding a one-day
    day-ahead-to-real-time position. Used as the P&L series for the
    position-sizing exercise.
    """
    rows = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year} (DA + RT)...")
        da = fetch_lmp_year(year, market='da')
        rt = fetch_lmp_year(year, market='rt')

        da_daily = da.groupby('date')['lmp'].agg(['mean', 'std', 'count'])
        da_daily.columns = ['da_price', 'da_dispersion', 'da_hours']

        rt_daily = rt.groupby('date')['lmp'].agg(['mean', 'count'])
        rt_daily.columns = ['rt_price', 'rt_hours']

        rows.append(da_daily.join(rt_daily).reset_index())

    df = pd.concat(rows, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['da_hours'] == 24) & (df['rt_hours'] == 24)]
    df['dart_premium'] = df['rt_price'] - df['da_price']
    df = df[['date', 'da_price', 'da_dispersion', 'dart_premium']]
    df = df.sort_values('date').reset_index(drop=True)

    df.to_csv('data/processed/da_features.csv', index=False)
    print(f"Saved {len(df)} rows")
    print(df[['da_dispersion', 'dart_premium']].describe())
    return df

if __name__ == "__main__":
    build_lmp_dataset(2020, 2025)
    build_da_features(2020, 2025)