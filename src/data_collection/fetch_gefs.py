"""
Return ensemble spread for a given date
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config
import pandas as pd
import numpy as np
from herbie import Herbie
from datetime import datetime, timedelta


def get_daily_spread(date_time, fxx):
    """
    Pulls all 31 members, stacks them, computes std dev and mean, subsets to PJM, returns daily_spread and mean_temp as floats

    date_time: 'YYYY-MM-DD 00:00'
    """
    member_temps = []

    for i in range(31):
        try:
            H = Herbie(date_time, model='gefs', member=i, fxx=fxx, save_dir='/tmp')
            ds = H.xarray("TMP:2 m")
            pjm = ds.sel(
                latitude=slice(43, 35),
                longitude=slice(277, 286)
            )
            member_temps.append(pjm['t2m'].values)
        except Exception as e:
            print(f"Member {i} failed, skipping")
            continue
    print(f"Successfully pulled {len(member_temps)} members")
    if len(member_temps) < 10:
        return None, None
    members_stacked = np.stack(member_temps, axis=0)
    spread = members_stacked.std(axis=0)
    mean_temp_map = members_stacked.mean(axis=0)
    daily_spread = spread.mean()
    mean_temp = mean_temp_map.mean()
    return daily_spread, mean_temp


def get_target_date_features(target_date_str):
    """
    For a single target date T, pulls all 7 forecasts that describe T
    (from 7 different initialization dates at 7 different lead times),
    and computes the forecast evolution signal.
    """
    target_datetime = datetime.strptime(target_date_str, '%Y-%m-%d %H:%M')
    results = {}
    for fxx in config.fxx_list:
         init_date = target_datetime - timedelta(hours=fxx)
         init_date_str = init_date.strftime('%Y-%m-%d %H:%M')
         daily_spread, mean_temp = get_daily_spread(init_date_str, fxx)
         results[f'spread_{fxx}'] = daily_spread
         results[f'temp_{fxx}'] = mean_temp
    if results['temp_168'] is not None and results['temp_24'] is not None:
         forecast_shift = abs(results['temp_168'] - results['temp_24'])
    else:
         forecast_shift = None
    results['forecast_shift'] = forecast_shift

    n_valid = 0
    for fxx in config.fxx_list:
        if results[f'spread_{fxx}'] is not None:
            n_valid += 1
    results['n_valid_leads'] = n_valid
    return results


def get_forecast_trajectory_features(results):
    """
    Extends the basic forecast evolution signal into richer trajectory features
    that capture the full shape of how the forecast evolved, not just endpoints.
    """
    features = {}
    temp_keys = ['temp_168', 'temp_144', 'temp_120', 'temp_96', 'temp_72', 'temp_48', 'temp_24']
    temps = [results[key] for key in temp_keys if results[key] is not None]
    if len(temps) < 5:
        return {
            'trajectory_vol': None,
            'path_length': None,
            'near_term_shift': None,
            'far_term_shift': None
        }

    path_length=0
    for i in range(len(temps)-1):
        path_length += abs(temps[i]-temps[i+1])

    far_term_shift = 0 
    for i in range(len(temps[:4])-1):
        far_term_shift += abs(temps[i]-temps[i+1])

    near_term_shift = 0
    near_temps = temps[-3:]
    for i in range(len(near_temps)-1):
        near_term_shift += abs(near_temps[i]-near_temps[i+1])

    trajectory_vol = np.std(temps)

    features['trajectory_vol'] = trajectory_vol
    features['path_length'] = path_length
    features['near_term_shift'] = near_term_shift
    features['far_term_shift'] = far_term_shift
    return features
    

def build_spread_dataset(start_date, end_date):
    """
    Loops over every target date in your range, calls both
    get_target_date_features and get_forecast_trajectory_features,
    combines results into one row, saves incrementally.
    """
    filepath = 'data/processed/ensemble_spread.csv'
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d %H:%M')

    if os.path.exists(filepath): #fix for resets
        existing_df = pd.read_csv(filepath)
        last_date = existing_df['date'].iloc[-1]
        start_datetime = datetime.strptime(last_date, '%Y-%m-%d %H:%M') + timedelta(days=1)
        first_row = False
        print(f"Resuming from {start_datetime.strftime('%Y-%m-%d %H:%M')}")
    else:
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
        first_row = True
        print(f"Starting fresh from {start_datetime.strftime('%Y-%m-%d %H:%M')}")

    current_datetime = start_datetime
    total_days = (end_datetime - start_datetime).days + 1
    current_day = 0

    while current_datetime <= end_datetime:
        current_day += 1 
        pct = (current_day / total_days) * 100
        print(f"[{current_day}/{total_days}] Processing {current_datetime.strftime('%Y-%m-%d %H:%M')} ({pct:.1f}%)") # console logging
        date_str = current_datetime.strftime('%Y-%m-%d %H:%M')
        results = get_target_date_features(date_str)
        trajectory = get_forecast_trajectory_features(results)
        row = {'date': date_str, **results, **trajectory}
        row_df = pd.DataFrame([row])

        if first_row:
            row_df.to_csv(filepath, mode='w', header=True, index=False)
            first_row = False
        else:
            row_df.to_csv(filepath, mode='a', header=False, index=False)
        current_datetime += timedelta(days=1)


if __name__ == "__main__":
    build_spread_dataset('2020-09-23 00:00', '2025-12-31 00:00')

