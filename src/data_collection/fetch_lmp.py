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

def fetch_lmp_year(year):
    """
    Returns Local Date, Hour Number and PJM Total LMP for specified year, with columns renamed to date, hour, lmp respectively
    year : int, 2020<=year<=2025
    """
    year_csv = pd.read_csv(f"https://www.eia.gov/electricity/wholesalemarkets/csv/pjm_lmp_rt_hr_zones_{year}.csv", skiprows=3)
    year_csv = year_csv[['Local Date', 'Hour Number', 'PJM Total LMP']]
    year_csv.columns = ['date', 'hour', 'lmp']
    return year_csv

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

if __name__ == "__main__":
    build_lmp_dataset(2020, 2025)