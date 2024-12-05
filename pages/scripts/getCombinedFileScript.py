from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import bioread as br
import h5py
import webview
# import dash_core_components as dcc
from flask import Flask
import subprocess
import platform
import base64
import io
from pymongo import MongoClient
import dash_ag_grid as dag
import dash
import datetime
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import configparser
import pytz
import numpy as np
import polars as pl
import logging
import pickle
import shutil
import json
import time
import sys
import os
import re
import tkinter as tk
from tkinter import filedialog
if os.path.exists(rf'C:\Users\PsyLab-6028'):
    sys.path.append(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages')
else:
    sys.path.append(r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages')

import UTILS.utils as ut
# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings
warnings.filterwarnings('ignore')





def main(project, now, username):
    FIRST = 0
    LAST = -1
    if os.path.exists(rf'C:\Users\PsyLab-6028'):
        exeHistory_path = Path(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\ExecutionHis\exeHistory.parquet')
    else:
        exeHistory_path = Path(r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\ExecutionHis\exeHistory.parquet')   

    exeHistory = pl.read_parquet(exeHistory_path)
    if os.path.exists(rf'C:\Users\PsyLab-6028'):
        paths_json = json.load(open(r"C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\Pconfigs\paths.json", "r"))
    else:
        paths_json = json.load(open(r"C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\Pconfigs\paths.json", "r"))    

    project_path = Path(paths_json[project])



    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

    AGGREGATED_OUTPUT_PATH_HISTORY = ut.output_record(OUTPUT_PATH, 'Aggregated Output',username, now)

    if not AGGREGATED_OUTPUT_PATH_HISTORY.exists():
        os.makedirs(AGGREGATED_OUTPUT_PATH_HISTORY)


    if os.path.exists(rf'C:\Users\PsyLab-6028'):    
        PROJECT_CONFIG = json.load(open(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\Pconfigs\paths data.json', 'r'))
    else:
        PROJECT_CONFIG = json.load(open(r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\Pconfigs\paths data.json', 'r'))   

    SUBJECTS_DATES = METADATA_PATH.joinpath('Subjects Dates.csv')


    subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                try_parse_dates=True)



    subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')
    if os.path.exists(rf'C:\Users\PsyLab-6028'):
        selected_subjects_path = Path(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_gen_Combined.parquet')
    else:
        selected_subjects_path = Path(rf'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\sub_selection\{project}_sub_selection_gen_Combined.parquet')
        
    subjects_to_run_on = []

    if not selected_subjects_path.exists():
        print(f'Selected subjects file does not exist for {project}')
        time.sleep(10)
        quit()
    else:
        selected_subjects = pl.read_parquet(selected_subjects_path)
    
        subjects_to_run_on = selected_subjects['Subject'].to_list()

    run_on_specific_subjects = True

    sleep_all_subjects_path = Path(OUTPUT_PATH.joinpath('Aggregated Output', 'Sleep All Subjects.csv'))
    steps_all_subjects_path = Path(OUTPUT_PATH.joinpath('Aggregated Output', 'Steps Aggregated.csv'))

    if not sleep_all_subjects_path.exists():
        print('Sleep data does not exist')
        time.sleep(10)
        quit()

    if not steps_all_subjects_path.exists():
        print('Steps data does not exist')
        time.sleep(10)
        quit()
    

    sleep_all_subjects = pl.read_csv(sleep_all_subjects_path, try_parse_dates=True)
    steps_all_subjects = pl.read_csv(steps_all_subjects_path, try_parse_dates=True)

    if sleep_all_subjects.is_empty():
        print('Sleep data is empty')
        time.sleep(10)
        quit()

    if steps_all_subjects.is_empty():
        print('Steps data is empty')
        time.sleep(10)
        quit()

    missing_values_df = pl.DataFrame()

    all_subjects_1_min_resolution = pl.DataFrame(strict=False)

    subjects_with_missing_heart_rate_files = []
    print('Upload and basic files stats')
    tqdm_subjects = tqdm(subjects_to_run_on)
    for subject in tqdm_subjects:
        if not re.search(r'\d{3}$', subject):
            continue
        if run_on_specific_subjects and subject not in subjects_to_run_on:
            continue
        tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)

        if not os.path.exists(OUTPUT_PATH.joinpath(subject)):
            os.mkdir(OUTPUT_PATH.joinpath(subject))

        latest_heart_rate_file_path = ut.new_get_latest_file_by_term("Heart Rate", subject=subject, root=OUTPUT_PATH)
        if not latest_heart_rate_file_path.exists():
            print(f'Heart Rate.csv is missing from {subject} subject folder')
            continue
        # get experiment date from file
        experiment_dates = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)

        experiment_start_datetime = experiment_dates['ExperimentStartDateTime'].values[0]
        experiment_end_datetime = experiment_dates['ExperimentEndDateTime'].values[0]

        experiment_start_date = experiment_dates['ExperimentStartDate'].values[0]
        experiment_end_date = experiment_dates['ExperimentEndDate'].values[0]

        experiment_date_range = (
            pl.DataFrame({
                'FullDate': pl.date_range(experiment_start_date, experiment_end_date, interval='1d', eager=True)
                })
        )

        range_of_experiment_datetimes = (
            pl.DataFrame({
                'FullDateTime': pl.datetime_range(experiment_start_datetime, experiment_end_datetime, interval='1m', eager=True, time_unit='us')
                })
        )

        if experiment_dates['NotInIsrael'].values[0]:
            range_of_experiment_dates_before = pd.date_range(experiment_dates['ExperimentStartDate'].values[0], pd.to_datetime(experiment_dates['NotInIsraelStartDate'].values[0]) - pd.Timedelta(days=1), freq='D')
            range_of_experiment_dates_after = pd.date_range(pd.to_datetime(experiment_dates['NotInIsraelEndDate'].values[0]) + pd.Timedelta(days=1), experiment_dates['ExperimentEndDate'].values[0], freq='D')
            range_mask = range_of_experiment_dates_before.append(range_of_experiment_dates_after)

        if experiment_dates['NotInIsrael_1'].values[0]:
            range_of_experiment_dates_before = pd.date_range(experiment_dates['ExperimentStartDate'].values[0], pd.to_datetime(experiment_dates['NotInIsraelStartDate_1'].values[0]) - pd.Timedelta(days=1), freq='D')
            range_of_experiment_dates_after = pd.date_range(pd.to_datetime(experiment_dates['NotInIsraelEndDate_1'].values[0]) + pd.Timedelta(days=1), experiment_dates['ExperimentEndDate'].values[0], freq='D')
            range_mask = range_of_experiment_dates_before.append(range_of_experiment_dates_after)



        

        # Read the latest heart rate file
        subject_heart_rate_df = pl.read_csv(latest_heart_rate_file_path, try_parse_dates=True)

        subject_heart_rate_df = (
            subject_heart_rate_df
            .filter(
                pl.col('DateTime').is_in(range_of_experiment_datetimes['FullDateTime'])
            )
        )

        # for date in subject_heart_rate_df['DateTime']:
        #     if date._date_repr not in range_of_experiment_datetimes:
        #         # drop the row if the date is not in the range of experiment dates
        #         subject_heart_rate_df = subject_heart_rate_df[subject_heart_rate_df['DateTime'] != date]

        # Select rows of current subject of *** sleep_df ***
        subject_sleep_df = (
            sleep_all_subjects
            .filter(
                pl.col('Id') == subject
            )
        )


        # Get sleep levels from the sleep jsons of the subject

        sleep_json_dir = DATA_PATH.joinpath(subject, 'FITBIT', 'Sleep')
        sleep_json_files = [file_path for file_path in os.listdir(sleep_json_dir) if re.search(r'sleep-\d{4}-\d{2}-\d{2}.json', file_path)]

        sleep_levels_df = pl.DataFrame()
        for sleep_json_file in sleep_json_files:
            
        
            with open(sleep_json_dir.joinpath(sleep_json_file), 'r') as f:
                sleep_data = json.load(f)
                if sleep_data == []:
                    print(f'Empty file: {sleep_json_file}')
                    continue
            data_df = (
                pl.read_json(sleep_json_dir.joinpath(sleep_json_file))
                .select('levels')
                .unnest('levels')
                .select(pl.col('data').explode())
                .unnest('data')
            ).select(
                pl.col('dateTime').str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M:%S.%f').dt.round('1m').alias('dateTime'),
                pl.col('level').alias('level'),
                pl.col('seconds').alias('seconds')
            )


            data_df = data_df.with_columns(
                pl.col('dateTime').dt.truncate('1d').alias('date')
            )

            dates_edited = pl.DataFrame({
                'minute_ranges': pl.Series([], dtype=pl.Datetime),
                'level': pl.Series([], dtype=pl.Utf8),
            })

            for df in data_df.partition_by('date'):
                df = (
                    df
                    .select(
                        pl.col('dateTime').alias('start_time'),
                        (pl.col('seconds')*1000000).cast(pl.Duration('us')).alias('seconds'),
                        'level'
                    )
                    .with_columns(
                        end_time = pl.col('start_time') + pl.col('seconds')
                    )
                    .select(
                        pl.datetime_ranges('start_time', 'end_time', interval='1m').alias('minute_ranges'),
                        'level',
                        'seconds'
                    )
                    .explode('minute_ranges')
                    .sort('seconds', descending=True)
                    .unique(subset=['minute_ranges'], keep='first')
                    # .drop('level')  # Modify this part if you need to keep level
                )
                dates_edited = pl.concat([dates_edited, df], how='diagonal')
                
                


                
                
            
            sleep_levels_df = pl.concat([sleep_levels_df, dates_edited], how='diagonal')

        
        

        minutes_of_sleep = (
            pl.DataFrame(subject_sleep_df)
            .select(
                BedStartTime = pl.col('BedStartTime').dt.cast_time_unit('us'),
                BedEndTime = pl.col('BedEndTime').dt.cast_time_unit('us')
            ).select(
                sleep_time = pl.datetime_ranges('BedStartTime', 'BedEndTime', interval='1m')
            ).explode('sleep_time')
            .with_columns(
                sleep_time = pl.col('sleep_time').dt.round('1m')
            )
        )

        sleep_levels_df = (
            minutes_of_sleep
            .join(
                sleep_levels_df,
                left_on='sleep_time',
                right_on='minute_ranges',
                how='left'
            )
            .sort('sleep_time')
            .drop_nulls(
                subset=['level']
            ).with_columns(
                level=pl.col('level').replace({
                    'wake': 'sleep_awake',
                    'awake': 'sleep_awake'
                })
            )
        )

        

        subject_sleep_df = (
            subject_sleep_df
            .filter(
                pl.col('DateOfSleepEvening').is_in(experiment_date_range['FullDate'])
            )
        )

        # for date in subject_sleep_df['DateOfSleepEvening']:
        #     if date not in range_of_experiment_dates:
        #         # drop the row if the date is not in the range of experiment dates
        #         subject_sleep_df = subject_sleep_df[subject_sleep_df['DateOfSleepEvening'] != date]

        # Select rows of current subject of *** steps_df ***
        subject_steps_df = (
            steps_all_subjects
            .filter(
                pl.col('Id') == subject
            )
        )

        subject_steps_df = (
            subject_steps_df
            .filter(
                pl.col('DateAndMinute').is_in(range_of_experiment_datetimes['FullDateTime'])
            )
        )



        # for date in subject_steps_df['DateAndMinute']:
        #     if date._date_repr not in range_of_experiment_dates:
        #         # drop the row if the date is not in the range of experiment dates
        #         subject_steps_df = subject_steps_df[subject_steps_df['DateAndMinute'] != date]

        subject_heart_rate_steps_df = (
            subject_heart_rate_df
            .join(
                subject_steps_df,
                left_on='DateTime',
                right_on='DateAndMinute',
                how='outer'
            )
        )

        # add 'steps' column by merging steps csv with heart rate csv (on 'dateTime' column: per minute in both)
        # 'outer' join to keep all rows from both dataframes
        # subject_heart_rate_steps_df = pd.merge(subject_heart_rate_df, subject_steps_df[['DateAndMinute', 'StepsInMinute']],
        #                                     left_on='DateTime',
        #                                     right_on='DateAndMinute',
        #                                     how='outer')
        # Copy the values of DateAndMinute to DateTime column where it's missing (because it's missing after the merge. so now its complete.)
        # subject_heart_rate_steps_df['DateTime'] = subject_heart_rate_steps_df['DateTime'].fillna(subject_heart_rate_steps_df['DateAndMinute'])
        # experiment_start_datetime = experiment_dates['ExperimentStartDateTime'].values[0]
        # experiment_end_datetime = experiment_dates['ExperimentEndDateTime'].values[0]
        
        subject_heart_rate_steps_df = (
            subject_heart_rate_steps_df
            .join(
                range_of_experiment_datetimes,
                left_on='DateTime',
                right_on='FullDateTime',
                how='outer'
            )
        )

        subject_heart_rate_steps_df = (
            subject_heart_rate_steps_df
            .with_columns(
                DateTime = pl.col('DateTime').fill_null(pl.col('FullDateTime'))
            )
            .sort('DateTime')
            .drop('FullDateTime')
            .drop('DateAndMinute')
            .with_columns(
                DateAndMinute = pl.col('DateTime').dt.cast_time_unit('ns')
            )
            .to_pandas()

        )
        # subject_heart_rate_steps_df = subject_heart_rate_steps_df.dropna(subset=['DateTime'])
        # Drop 'DateAndMinute' column
        # subject_heart_rate_steps_df = subject_heart_rate_steps_df.drop('DateAndMinute')
        # Rename 'DateTime' column to 'DateAndMinute'
        # subject_heart_rate_steps_df = subject_heart_rate_steps_df.rename(columns={'DateTime': 'DateAndMinute'})
        
        # Add 'Weekday' column
        subject_heart_rate_steps_df['Weekday'] = subject_heart_rate_steps_df['DateAndMinute'].dt.day_name()

        # iterrows version
        # Add 'Mode' column by merging sleep csv with heart rate csv (on 'dateTime' column: per minute in both)
        # Mode values are: 'awake', 'in_bed', 'sleeping', 'unknown
        # subject_heart_rate_steps_df.loc[subject_heart_rate_steps_df['Mode']!='unknown', 'Mode'] = 'awake'  # default value for Mode column except 'unknown' values.
        subject_heart_rate_steps_df['Mode'] = 'awake' # default value for Mode column except 'unknown' values, and neglect the 'unknown' values.
        # Set SleepFeature column by time of sleep and time in bed.
        # SleepFeature values are: 'pre_sleep', 'post_wake', pd.nan
        subject_heart_rate_steps_df['SleepFeature'] = np.nan

        subject_sleep_df = (
            subject_sleep_df
            .to_pandas()
        )
        if subject_sleep_df['SleepEndTime'].isna().all():
            print(f'subject {subject} has no sleep data, skipping...')
            continue
        for index, row in subject_sleep_df.iterrows():
            if index == 0:
                pass
            else:
                # If the SleepEndTime of today is nan, set unknown to this day and night
                if pd.isna(subject_sleep_df.loc[index, 'SleepEndTime']):
                    # Find the last index from the beginning of the dataframe to current index
                    last_valid_index = subject_sleep_df.loc[0 : index, 'SleepEndTime'].last_valid_index()
                    if last_valid_index is None:
                        # Find the next non-NaN index
                        next_valid_index = subject_sleep_df['SleepEndTime'].index[(subject_sleep_df['SleepEndTime'].index > index) & (~pd.isna(subject_sleep_df['SleepEndTime']))].min()
                        next_valid_value = subject_sleep_df['SleepStartTime'].iloc[next_valid_index]
                        subject_heart_rate_steps_df.loc[subject_heart_rate_steps_df['DateAndMinute'] < next_valid_value, 'Mode'] = 'unknown'
                        continue
                    last_valid_value = subject_sleep_df['SleepEndTime'].iloc[last_valid_index]
                    if last_valid_index is not None:
                        # Find the next non-NaN index
                        next_valid_index = subject_sleep_df['SleepEndTime'].index[(subject_sleep_df['SleepEndTime'].index > last_valid_index) & (~pd.isna(subject_sleep_df['SleepEndTime']))].min()
                        if pd.isna(next_valid_index): # meaning, that current row is the last row of the dataframe and it is missing a sleep. so set unknown to the rest of the dataframe.
                            subject_heart_rate_steps_df.loc[last_valid_value <= subject_heart_rate_steps_df['DateAndMinute'], 'Mode'] = 'unknown'
                            continue
                        # Get the next non-NaN value
                        next_valid_value = subject_sleep_df['SleepStartTime'].iloc[next_valid_index]
                        subject_heart_rate_steps_df.loc[
                            (last_valid_value + datetime.timedelta(minutes=60) < subject_heart_rate_steps_df['DateAndMinute']) &
                            (subject_heart_rate_steps_df['DateAndMinute'] < next_valid_value - datetime.timedelta(minutes=60)), 'Mode'] = 'unknown' # datetime.timedelta(minutes=60) add to not delete the post_wake and pre_sleep because we know that the subject was sleep
                    else: # if there is no data from the last valid index to the end of the dataframe.
                        subject_heart_rate_steps_df.loc[last_valid_value <= subject_heart_rate_steps_df['DateAndMinute'], 'Mode'] = 'unknown'


            

            # Set Mode column by time of sleep and time in bed.
            in_bed_mask = (subject_heart_rate_steps_df['DateAndMinute'] >= row['BedStartTime']) & (subject_heart_rate_steps_df['DateAndMinute'] <= row['BedEndTime'])
            subject_heart_rate_steps_df.loc[in_bed_mask, 'Mode'] = 'awake'
            sleeping_mask = (subject_heart_rate_steps_df['DateAndMinute'] >= row['SleepStartTime']) & (subject_heart_rate_steps_df['DateAndMinute'] <= row['SleepEndTime'])
            subject_heart_rate_steps_df.loc[sleeping_mask, 'Mode'] = 'sleeping'

            


            if row['ValidSleep']:
                subject_heart_rate_steps_df.loc[sleeping_mask, 'ValidSleep'] = True
            else:
                subject_heart_rate_steps_df.loc[sleeping_mask, 'ValidSleep'] = False

            # valid_sleep_mask = (sleep_df['ValidSleep'] == True) | (sleep_df['MainSleep (Fitbit Calculation)'] == True)
            # invalid_sleeping_mask = (sleep_df['ValidSleep'] == False) | (sleep_df['MainSleep (Fitbit Calculation)'] == False)
            # subject_heart_rate_steps_df.loc[sleeping_mask & valid_sleep_mask, 'ValidSleep'] = True
            # subject_heart_rate_steps_df.loc[sleeping_mask & invalid_sleeping_mask, 'ValidSleep'] = False

            # Add sleep features (pre sleep, post wake) only for valid sleep
            if row['ValidSleep']:
                hour_before_sleep = row['SleepStartTime'] - pd.Timedelta(hours=1)
                hour_after_wake = row['SleepEndTime'] + pd.Timedelta(hours=1)
                # Set SleepFeature column by time of sleep and time in bed.
                # SleepFeature values are: 'pre_sleep', 'post_wake', pd.nan
                subject_heart_rate_steps_df.loc[
                    (subject_heart_rate_steps_df['DateAndMinute'] >= hour_before_sleep) &
                    (subject_heart_rate_steps_df['DateAndMinute'] < row['SleepStartTime']),
                    'SleepFeature'] = 'pre_sleep'
                subject_heart_rate_steps_df.loc[
                    (subject_heart_rate_steps_df['DateAndMinute'] > row['SleepEndTime']) &
                    (subject_heart_rate_steps_df['DateAndMinute'] <= hour_after_wake),
                    'SleepFeature'] = 'post_wake'


        subject_heart_rate_steps_df = (
            pl.DataFrame(subject_heart_rate_steps_df)
            .with_columns(
                DateAndMinute = pl.col('DateAndMinute').dt.cast_time_unit('us'),
            )
            .join(
                sleep_levels_df,
                left_on='DateAndMinute',
                right_on='sleep_time',
                how='left'
            )
            .with_columns(
                Mode2 = pl.when(
                    pl.col('Mode') == pl.lit('unknown')
                ).then(
                    pl.lit('unknown')
                ).when(
                    pl.col('Mode') == pl.lit('awake')
                ).then(
                    pl.lit('awake')
                ).otherwise(
                    pl.col('level')
                )

            )
            .with_columns(
                pl.col('Mode2').fill_null(strategy = 'forward')
            )
            .drop('level')
            .with_columns(
                pl.col('DateAndMinute').dt.cast_time_unit('ns').alias('DateAndMinute'),
            )
            .to_pandas()
        )
        # Add 'Weekend' column, True if it's weekend, False if it's not weekend
        # get experiment date from file
        experiment_dates = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
        # get range of dates
        range_of_experiment_dates = pd.date_range(experiment_dates['ExperimentStartDate'].values[0], experiment_dates['ExperimentEndDate'].values[0], freq='D')
        range_mask = range_of_experiment_dates
        if experiment_dates['NotInIsrael'].values[0]:
            range_of_experiment_dates_before = pd.date_range(experiment_dates['ExperimentStartDate'].values[0], pd.to_datetime(experiment_dates['NotInIsraelStartDate'].values[0]) - pd.Timedelta(days=1), freq='D')
            range_of_experiment_dates_after = pd.date_range(pd.to_datetime(experiment_dates['NotInIsraelEndDate'].values[0]) + pd.Timedelta(days=1), experiment_dates['ExperimentEndDate'].values[0], freq='D')
            range_mask = range_of_experiment_dates_before.append(range_of_experiment_dates_after)
        if experiment_dates['NotInIsrael_1'].values[0]:
            range_of_experiment_dates_before = pd.date_range(experiment_dates['ExperimentStartDate'].values[0], pd.to_datetime(experiment_dates['NotInIsraelStartDate_1'].values[0]) - pd.Timedelta(days=1), freq='D')
            range_of_experiment_dates_after = pd.date_range(pd.to_datetime(experiment_dates['NotInIsraelEndDate_1'].values[0]) + pd.Timedelta(days=1), experiment_dates['ExperimentEndDate'].values[0], freq='D')
            range_mask = range_of_experiment_dates_before.append(range_of_experiment_dates_after)

        # Create Weekend column
        subject_heart_rate_steps_df['Weekend'] = False # default value
        # Create Dataframe contains only valid sleeps and only 'SleepEndTime' column without nan values
        end_sleep_mean = pd.DataFrame(
            [datetime.datetime.combine(datetime.date(1900, 1, 1), date.time()) for date in
            subject_sleep_df['SleepEndTime'][subject_sleep_df["ValidSleep"] == 1] if
            pd.notnull(date)],
            columns=["time"])
        
        if end_sleep_mean.empty:
            print(f'subject {subject} has no sleep end time in the date ...')
            
        # Calculate mean of 'SleepEndTime' column
        
        if isinstance(end_sleep_mean["time"].mean(),float):
            end_sleep_mean = datetime.time(8,0)
        else:
            end_sleep_mean = end_sleep_mean["time"].mean().time()
        # Create Dataframe contains only valid sleeps and only 'SleepStartTime' column without nan values
        start_sleep_mean = pd.DataFrame(
            [datetime.datetime.combine(datetime.date(1900, 1, 1), date.time()) for date in
            subject_sleep_df['SleepStartTime'][subject_sleep_df["ValidSleep"] == 1] if
            pd.notnull(date)],
            columns=["time"])
        # Calculate mean of 'SleepEndTime' column
        if isinstance(start_sleep_mean["time"].mean(),float):
            start_sleep_mean = start_sleep_mean["time"].mean()
        else:
            start_sleep_mean = start_sleep_mean["time"].mean().time()

        # Filter only valid rows because we want to iterate only on valid sleeps (1 row for each sleep)
        subject_valid_sleep_df = subject_sleep_df[(subject_sleep_df['ValidSleep'] == True) & (subject_sleep_df['MainSleep (Fitbit Calculation)'] == True)]
        weekends = [] # list of dictionaries, each dictionary is a weekend with 'weekend_start' and 'weekend_end' keys

        # Find weekend start and weekend end for each weekend in the experiment
        # Weekend is from end of thursday sleep to end of saturday sleep (sunday wake up).
        # Assumption: 2 days after thursday is saturday because there is no duplicate dates when selecting only valid sleep
        #            3 days after thursday is sunday because there is no duplicate dates when selecting only valid sleep
        # Iterate over each day of the experiment and find Thursdays.
        # Then calculate weekend start from the end of thursday sleep and weekend end from the end of saturday sleep.
        # In short there are 4 options:
        # 1. Thursday sleep and Saturday sleep are valid - use them
        # 2. Thursday sleep is valid (so use it) and Saturday sleep is invalid (so use mean end sleep time of each day)
        # 3. Thursday sleep is invalid (so use mean end sleep time fo each day) and Saturday sleep is valid (so use it)
        # 4. Thursday sleep and Saturday sleep are invalid (use mean end sleep time for both)
        for date in range_of_experiment_dates:
        
            if date.day_name() == 'Thursday':
                if subject_valid_sleep_df['DateOfSleepEvening'].isin([date]).any(): # If there is sleep data for this day (= if the date is in the dataframe)
                    if subject_valid_sleep_df['DateOfSleepEvening'].isin([date + pd.Timedelta(days=2)]).any(): # If there is sleep data for this day + 2 (Saturday)
                        weekends.append({'weekend_start': subject_valid_sleep_df.loc[subject_valid_sleep_df['DateOfSleepEvening'] == date]['SleepEndTime'].values[0], # The end of Thursday sleep
                                        'weekend_end': subject_valid_sleep_df.loc[subject_valid_sleep_df['DateOfSleepEvening'] == date + pd.Timedelta(days=2)]['SleepEndTime'].values[0]}) # The end of Saturday sleep
                    else: # If there is no sleep data for this day + 2 (Saturday) use mean end time of all sleeps
                        weekends.append({'weekend_start': subject_valid_sleep_df.loc[subject_valid_sleep_df['DateOfSleepEvening'] == date]['SleepEndTime'].values[0], # The end of Thursday sleep
                                        'weekend_end': pd.Timestamp.combine(date + pd.Timedelta(days=3), end_sleep_mean)}) # days=3 because we want the end of the weekend to be Sunday
                else: # If there is no sleep data for Thursday use mean end time of all sleeps
                    if subject_valid_sleep_df['DateOfSleepEvening'].isin([date + pd.Timedelta(days=2)]).any(): # If there is sleep data for this day + 2 (Saturday)
                        weekends.append({'weekend_start': pd.Timestamp.combine(date, end_sleep_mean), # use mean end time of all sleeps because there is no sleep data for Thursday
                                        'weekend_end': subject_valid_sleep_df.loc[subject_valid_sleep_df['DateOfSleepEvening'] == date + pd.Timedelta(days=2)]['SleepEndTime'].values[0]}) # The end of Saturday sleep
                    else: # If there is no sleep data for Thursday and Saturday use mean end time of all sleeps
                        weekends.append({'weekend_start': pd.Timestamp.combine(date, end_sleep_mean), # use mean end time of all sleeps because there is no sleep data for Thursday
                                        'weekend_end': pd.Timestamp.combine(date + pd.Timedelta(days=3), end_sleep_mean)}) # use mean end time of all sleeps because there is no sleep data for Saturday
        # Set Weekend column to True for all dates between weekend_range
        for weekend in weekends:
            subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['DateAndMinute'] > weekend['weekend_start']) &
                                            (subject_heart_rate_steps_df['DateAndMinute'] <= weekend['weekend_end']), 'Weekend'] = True

        # Keep the order of the statements.

        #### Set Activity values ####
        # Activity values are: 'sleep', 'rest', 'awake', 'unknown'.
        subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['Mode'] == 'sleeping'), 'Activity'] = 'sleep'
        subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['Mode'] == 'in_bed'), 'Activity'] = 'rest'
        subject_heart_rate_steps_df.loc[subject_heart_rate_steps_df['Mode'] == 'awake', 'Activity'] = 'awake'
        subject_heart_rate_steps_df.loc[subject_heart_rate_steps_df['Mode'] == 'unknown', 'Activity'] = 'unknown'
        # Add 'rest' value to Activity column by steps values
        rest_condition = ((subject_heart_rate_steps_df['StepsInMinute'] == 0) |
                        subject_heart_rate_steps_df['StepsInMinute'].isna()) & \
                        (subject_heart_rate_steps_df['Mode'] == 'awake')
        subject_heart_rate_steps_df.loc[rest_condition, 'Activity'] = 'rest'

        # Add 'low_activity', 'med_activity', 'high_activity' values in Activity column by steps values
        low_activity_condition = subject_heart_rate_steps_df['StepsInMinute'].between(1, 9) & (subject_heart_rate_steps_df['Mode'] == 'awake')
        subject_heart_rate_steps_df.loc[low_activity_condition, 'Activity'] = 'low_activity'

        med_activity_condition = subject_heart_rate_steps_df['StepsInMinute'].between(10, 49) & (subject_heart_rate_steps_df['Mode'] == 'awake')
        subject_heart_rate_steps_df.loc[med_activity_condition, 'Activity'] = 'med_activity'

        high_activity_condition = (subject_heart_rate_steps_df['StepsInMinute'] >= 50) & (subject_heart_rate_steps_df['Mode'] == 'awake')
        subject_heart_rate_steps_df.loc[high_activity_condition, 'Activity'] = 'high_activity'

        # Create a flag named "sequence_number" column to identify sequences of the same activity
        # For example:
        # Activity | sequence_number
        # sleep    | 1
        # sleep    | 1
        # rest     | 2
        # rest     | 2
        # low_activity | 3
        # low_activity | 3
        subject_heart_rate_steps_df['sequence_number'] = (subject_heart_rate_steps_df['Activity'] != subject_heart_rate_steps_df['Activity'].shift(1)).cumsum()
        # Create a dictionary of the groups of the same activity with the keys: 'is_more_then_5_minutes_sequence', 'activity', 'start_index', 'end_index'
        more_then_5_minutes_groups_final = subject_heart_rate_steps_df.groupby('sequence_number').apply(lambda group:
                                                                                                        {'is_more_then_5_minutes_sequence': len(group['sequence_number']) >= 5,
                                                                                                        'activity':group['Activity'].iloc[0],
                                                                                                        'start_index':group['sequence_number'].index[0],
                                                                                                        'end_index':group['sequence_number'].index[-1]})
        # Append to pre_list and post_list the indexes of 10 minutes before the start of high activity and 10 minutes after the end of the high activity workout
        pre_list, post_list = [], []
        for flag, group_info in more_then_5_minutes_groups_final.items():
            if group_info['activity'] == 'high_activity' and group_info['is_more_then_5_minutes_sequence']:
                pre_list.append((group_info['start_index'] - 10, group_info['start_index']))  # 10 minutes before the start of the group
                post_list.append((group_info['end_index'] + 1, group_info['end_index'] + 11))  # 10 minutes after the end of the group

        # Add '10_min_pre_high_activity' and '10_min_post_high_activity' columns to the DataFrame to store the durations of pre and post high activity periods
        # The values of the columns are: '10_min_pre_high_activity', '10_min_post_high_activity', ''
        # If there is no high activity in the 10 minute duration, the value of the column is 'pre_10_min_high_activity' or 'post_10_min_high_activity'
        # If there is high activity in the 10 minute duration, the value of the column is ''
        subject_heart_rate_steps_df['10_min_pre_high_activity'] = ''
        subject_heart_rate_steps_df['10_min_post_high_activity'] = ''
        # Iterate over the pre_list and set the '10_min_pre' column to 'pre_10_min' for rows with no high activity in the 10 minute duration
        for start, end in pre_list:
            # Ten minutes pre high activity mask
            ten_minutes_pre_mask = (subject_heart_rate_steps_df.index >= start) & (subject_heart_rate_steps_df.index < end)
            # Get the rows in the specified range
            ten_rows_pre = subject_heart_rate_steps_df.loc[ten_minutes_pre_mask]
            # Check if there are no high activity events in the 10 minutes duration
            if not (ten_rows_pre['Activity'] == 'high_activity').any():
                # Set the '10_min_pre' column to '10_min_pre' for the rows in the specified range
                subject_heart_rate_steps_df.loc[ten_minutes_pre_mask, '10_min_pre_high_activity'] = '10_min_pre_high_activity'
        for start, end in post_list:
            # Ten minutes post high activity mask
            ten_minutes_post_mask = (subject_heart_rate_steps_df.index >= start) & (subject_heart_rate_steps_df.index < end)
            # Get the rows in the specified range
            ten_rows_post = subject_heart_rate_steps_df.loc[ten_minutes_post_mask]
            # Check if there are no high activity events in the 10 minutes duration
            if not (ten_rows_post['Activity'] == 'high_activity').any():
                # Set the '10_min_post' column to '10_min_post' for the rows in the specified range
                subject_heart_rate_steps_df.loc[ten_minutes_post_mask, '10_min_post_high_activity'] = '10_min_post_high_activity'

        # Drop the 'sequence_number' column because it is not needed anymore
        subject_heart_rate_steps_df = subject_heart_rate_steps_df.drop(columns=['sequence_number'])

        # Create the Feature columns from the activity column.
        # Default value is the activity column
        subject_heart_rate_steps_df['Feature'] = subject_heart_rate_steps_df['Activity']
        # Set the Feature column to the sleep feature if the sleep feature is not null and the activity is not high activity
        subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['Feature'] != 'high_activity') &
                                        (subject_heart_rate_steps_df['SleepFeature'].notnull()) &
                                        (subject_heart_rate_steps_df['Mode'] != 'unknown'),
                                        'Feature'] = subject_heart_rate_steps_df['SleepFeature']
        # Set the Feature column to the 10 minute pre feature if the 10 minute pre feature is not null
        subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['10_min_pre_high_activity'] != '') &
                                        (subject_heart_rate_steps_df['Mode'] != 'unknown'),
                                        'Feature'] = subject_heart_rate_steps_df['10_min_pre_high_activity']
        # Set the Feature column to the 10 minute post feature if the 10 minute post feature is not null
        subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['10_min_post_high_activity'] != '') &
                                        (subject_heart_rate_steps_df['Mode'] != 'unknown'),
                                        'Feature'] = subject_heart_rate_steps_df['10_min_post_high_activity']
        # Set the Feature column to 'invalid_sleep' where ValidSleep column is False
        subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['ValidSleep'] == False) &
                                        (subject_heart_rate_steps_df['Mode'] != 'unknown'),
                                        'Feature'] = 'invalid_sleep'

        # Set pre_sleep/post_wake in Feature where the Mode is unknown
        # subject_heart_rate_steps_df.loc[(subject_heart_rate_steps_df['Feature'] == 'sleep'), 'Feature'] = 'sleeping'



        # Remove ValidSleep column
        # subject_heart_rate_steps_df = subject_heart_rate_steps_df.drop(columns=['ValidSleep'])

        # Group by the Feature column and the then calculate the mean and standard deviation bpm
        # z score calculation of the mean bpm of each feature (which is value in the Feature column)
        measures = {}
        for feature, group in subject_heart_rate_steps_df.groupby('Feature'):
            measures[feature] = {'MeanBpm': group['BpmMean'].mean(), 'StdBpm':group['BpmMean'].std()}
        # Add the z_score column to the DataFrame. z score = (x-μ) / σ
        subject_heart_rate_steps_df['z_score'] = subject_heart_rate_steps_df.apply(
                                                lambda row:
                                                (row['BpmMean'] - measures[row['Feature']]['MeanBpm']) / measures[row['Feature']]['StdBpm'], axis=1)

        # Filter z_scores to only include values that are less than absolute 2.7
        subject_heart_rate_steps_df.loc[abs(subject_heart_rate_steps_df['z_score']) <= 2.7, ['z_score_outlier_above_2_7']] = False
        # Set the z_score_outlier_above_2_7 column to False if the z_score is greater or equal than absolute 2.7
        subject_heart_rate_steps_df.loc[abs(subject_heart_rate_steps_df['z_score']) > 2.7, ['z_score_outlier_above_2_7']] = True
        # Set the z_score_outlier_above_2_7 column to nan if the BpmMean is nan
        subject_heart_rate_steps_df.loc[subject_heart_rate_steps_df['BpmMean'].isnull(), 'z_score_outlier_above_2_7'] = np.nan

        # remove outliers with "1.5 IQR rule" / "Outlier Formula"
        # outliers is a dictionary with the feature as the key and the q1, q3 and iqr as the values (dict in dict)
        # for example: {'10_min_post_high_activity': {'iqr': 14.621568627450984, 'q1': 74.96666666666667, 'q3': 89.58823529411765}, ...}
        outliers = {}
        for feature, group in subject_heart_rate_steps_df.groupby('Feature'):
            q1 = group['BpmMean'].quantile(0.25)
            q3 = group['BpmMean'].quantile(0.75)
            # Inter quartile range
            iqr = q3 - q1
            outliers[feature] = {'q1': q1, 'q3': q3, 'iqr': iqr}



        # Iterate over the rows of the DataFrame and find outliers using "1.5 IQR rule" / "Outlier Formula" .
        # Create a list of True and False values for the outliers column.
        outliers_list = []
        for index, row in subject_heart_rate_steps_df.iterrows():
            q1 = outliers[row['Feature']]['q1']
            q3 = outliers[row['Feature']]['q3']
            iqr = outliers[row['Feature']]['iqr']
            lower_boundary = q1 - (1.5 * iqr)
            upper_boundary = q3 + (1.5 * iqr)
            if (row['BpmMean'] < lower_boundary) or (upper_boundary < row['BpmMean']):
                outliers_list.append(True)
            else:
                outliers_list.append(False)
        # Add the outliers list to the DataFrame
        subject_heart_rate_steps_df['outliers'] = outliers_list
        # Set the outliers column to nan if the bpm mean is nan
        subject_heart_rate_steps_df.loc[subject_heart_rate_steps_df['BpmMean'].isnull(), 'outliers'] = np.nan

        # Sort the DataFrame by the DateAndMinute column
        subject_heart_rate_steps_df = subject_heart_rate_steps_df.sort_values(by=['DateAndMinute'])


        last_day_march = datetime.datetime.now().replace(month=3, day=31)
        last_day_october = datetime.datetime.now().replace(month=10, day=31)

        subject_heart_rate_steps_df = (
            pl.DataFrame(subject_heart_rate_steps_df)
            .with_columns(
                pl.col("DateAndMinute").map_elements(lambda x: is_dst_change(x), return_dtype=pl.Boolean).alias("is_dst_change")
            )
            .unique(subset=["DateAndMinute"], keep="first")
            .to_pandas()  # If you need to convert to pandas, otherwise, remove this line
        )
                

        columns_ordered = ['DateAndMinute', 'Weekday', 'BpmMean', 'NumOfValidSamples', 'NumOfAllSamples',
                                    'NumOfValidSamplesAfterBfill', 'StepsInMinute', 'SleepFeature',
                                    'Mode', 'Mode2', 'Activity', '10_min_pre_high_activity', '10_min_post_high_activity',
                                    'Feature', 'ValidSleep', 'Weekend', 'z_score', 'z_score_outlier_above_2_7',
                                    'outliers', 'not_in_israel', 'is_dst_change']

        # Add Feature_2 column to the DataFrame
        feature_2_df = pd.DataFrame()
        if not feature_2_df.empty:
            # Merge the Feature_2 column to the DataFrame by the DateAndMinute column
            subject_heart_rate_steps_df = pd.merge(subject_heart_rate_steps_df,
                                                feature_2_df,
                                                on='DateAndMinute',
                                                how='left')
            columns_ordered = ['DateAndMinute', 'Weekday', 'BpmMean', 'NumOfValidSamples', 'NumOfAllSamples',
                            'NumOfValidSamplesAfterBfill', 'StepsInMinute', 'SleepFeature',
                            'Mode', 'Mode2' ,'Activity', '10_min_pre_high_activity', '10_min_post_high_activity',
                            'Feature', 'Feature_2', 'ValidSleep', 'Weekend', 'z_score', 'z_score_outlier_above_2_7',
                            'outliers', 'not_in_israel', 'is_dst_change']

        

        # aDD 'not_in_israel' column to the DataFrame that show when the subject was not in israel
        subject_heart_rate_steps_df = (
            pl.DataFrame(subject_heart_rate_steps_df)
            .with_columns(
                not_in_israel=~(pl.col('DateAndMinute').dt.truncate("1d").is_in(range_mask))
            )
            .to_pandas()
        )
        # Save output CSV file to each subject output folder
        subject_output_path = OUTPUT_PATH.joinpath(subject)
        
        subject_output_path_history = ut.output_record(OUTPUT_PATH, subject, user_name, now)
        
        if not subject_output_path_history.exists():
            os.makedirs(subject_output_path_history)

        subject_heart_rate_steps_df.sort_values(by=['DateAndMinute'], inplace=True)
        subject_heart_rate_steps_df = (
            pl.DataFrame(subject_heart_rate_steps_df)
            .with_columns(
                Id = pl.col('Id').fill_null(subject)
            )
            .to_pandas()
        )
        # Select specific columns to reorder the columns and save the resulting DataFrame to a CSV file
        subject_heart_rate_steps_df[columns_ordered].to_csv(subject_output_path_history.joinpath(f'{subject} Heart Rate and Steps and Sleep Aggregated.csv'), index=False)  

        if all_subjects_1_min_resolution.is_empty():
            all_subjects_1_min_resolution = pl.DataFrame(subject_heart_rate_steps_df, strict=False)
        else:
            all_subjects_1_min_resolution = pl.concat([all_subjects_1_min_resolution, pl.DataFrame(subject_heart_rate_steps_df, strict=False)], how='vertical_relaxed')

        ut.check_for_duplications(subject_output_path, subject_output_path_history)

    all_subjects_1_min_resolution_path = OUTPUT_PATH.joinpath('Aggregated Output').joinpath('All_Subjects_1_Minute_Resolution.parquet')

    if not all_subjects_1_min_resolution_path.exists():
        all_subjects_1_min_resolution.select(['Id'] + columns_ordered).write_parquet(all_subjects_1_min_resolution_path)
    else:
        old_all_subjects_1_min_resolution = pl.read_parquet(all_subjects_1_min_resolution_path)
        old_all_subjects_1_min_resolution = (
            old_all_subjects_1_min_resolution
            .filter(
                ~pl.col('Id').is_in(all_subjects_1_min_resolution['Id'])
            )
        )

        all_subjects_1_min_resolution = all_subjects_1_min_resolution.select(['Id'] + columns_ordered)

        all_subjects_1_min_resolution = pl.concat([old_all_subjects_1_min_resolution, all_subjects_1_min_resolution])

        all_subjects_1_min_resolution.write_parquet(all_subjects_1_min_resolution_path)



        ######################## generating the by activity aggregated file ########################
        def by_activity(exclude_weekends):
            print(f'\n Aggregate Heart Rate Means By Activity, exclude_weekends={exclude_weekends}')
            
            # Create an empty list to hold any subjects that are missing HRV or respiratory files
            subjects_with_missing_raw_sleep_data, subjects_with_missing_aggregated_sleep_files, subjects_with_missing_aggregated_files = [], [], []

            sleep_df = (
                sleep_all_subjects
                .select(
                    'Id', 'DayOfExperiment', 'ExperimentDates', 'DateOfSleepEvening', 'WeekdayOfSleepEvening',
                    'WeekdayOfSleep', 'SleepStartTime', 'BedStartTime', 'SleepEndTime', 'BedEndTime', 'ValidSleep',
                    'MainSleep (Fitbit Calculation)'
                )
            )

            # # Read the latest full sleep file
            # sleep_df = pd.read_csv(latest_full_sleep_file_path,
            #                     usecols=['Id', 'DayOfExperiment', 'ExperimentDates', 'DateOfSleepEvening', 'WeekdayOfSleepEvening',
            #                                 'WeekdayOfSleep','SleepStartTime', 'BedStartTime', 'SleepEndTime', 'BedEndTime', 'ValidSleep', 'MainSleep (Fitbit Calculation)'],
            #                     dtype={'DayOfExperiment': 'int16', 'WeekdayOfSleepEvening': 'category', 'WeekdayOfSleep': 'category'},
            #                     parse_dates=['DateOfSleepEvening', 'BedStartTime',
            #                                     'SleepStartTime', 'SleepEndTime', 'BedEndTime'])
            # Filter only the valid sleep rows. By filtering only valid sleeps we make sure that we have 1 sleep per day.
            sleep_df = (
                sleep_df
                .filter(
                    (pl.col('ValidSleep') == True) & (pl.col('MainSleep (Fitbit Calculation)') == True)
                )
            )
            # sleep_df = sleep_df.loc[(sleep_df['ValidSleep'] == True) & (sleep_df['MainSleep (Fitbit Calculation)'] == True)]

            # Create new dataframe with subject id, date of sleep, and all the features
            subjects_full_week_means_list = []
            all_subjects_details_df = pd.DataFrame()
            # subjects_full_week_no_weekends_means_list = []
            # Iterate over the subjects in the Data folder
            print('\n Aggregate Heart Rate Means By Activity')
        
            tqdm_subjects = tqdm(os.listdir(DATA_PATH))
            for subject in tqdm_subjects:

                if not re.search(r'\d{3}$', subject):
                    continue
                if run_on_specific_subjects and subject not in subjects_to_run_on:
                    continue
                # Update the tqdm description with the current subject
                tqdm_subjects.set_description(f'Subject {subject}')
                # create set of features for each subject
                subject_features_set = set()
                # Create DataFrame with the current subject sleep data
                subject_sleep_df = sleep_df.filter(pl.col('Id') == subject).to_pandas()
                if subject_sleep_df.empty:
                    subjects_with_missing_raw_sleep_data.append(subject)
                    continue
                # Create a DataFrame with the subject's sleep data and the sleep features
                subject_means_df = pd.DataFrame({'Id': subject,
                                                'DayOfExperiment': subject_sleep_df['DayOfExperiment'],
                                                'ExperimentDates': subject_sleep_df['ExperimentDates'],
                                                'DateOfSleepEvening': subject_sleep_df['DateOfSleepEvening'],
                                                'WeekdayOfSleepEvening': subject_sleep_df['WeekdayOfSleepEvening'],
                                                'WeekdayOfSleep': subject_sleep_df['WeekdayOfSleep'],
                                                'SleepStartTime': subject_sleep_df['SleepStartTime'],
                                                'SleepEndTime': subject_sleep_df['SleepEndTime'],
                                                'BedStartTime': subject_sleep_df['BedStartTime'],
                                                'BedEndTime': subject_sleep_df['BedEndTime']})

                # Find the latest "Heart Rate and Steps and Sleep Aggregated" file that exists in each subject's output folder. if it's not exists, return.
                latest_aggregated_hr_steps_sleep_file_path = ut.new_get_latest_file_by_term('Heart Rate and Steps and Sleep Aggregated', subject=subject, root=OUTPUT_PATH)
                if not latest_aggregated_hr_steps_sleep_file_path.exists():
                    subjects_with_missing_aggregated_files.append(subject)
                    continue
                # Read the latest "Heart Rate and Steps" and "Sleep Aggregated" file
                hr_steps_sleep_df = pd.read_csv(latest_aggregated_hr_steps_sleep_file_path, parse_dates=['DateAndMinute'])

                # After removing invalid sleeps we add rows with missing data for the mean calculation.
                # We will replace the nan values of night sleeps with the mean of the day sleeps.
                # Get dates of experiment from metadata_of_project.txt file from Metadata folder.
                subject_experiment_metadata_row = experiment_dates
                subject_experiment_dates = pd.date_range(subject_experiment_metadata_row['ExperimentStartDate'].values[0],
                                                        subject_experiment_metadata_row['ExperimentEndDate'].values[0])

                # Convert the DatetimeIndex (subject_experiment_dates) to a DataFrame with a column named 'ExperimentStartDate'
                subject_experiment_dates_df = pd.DataFrame({'ExperimentDates': subject_experiment_dates})
                # Because we are going to merge by 'DateOfSleepEvening' column, we don't want to include the last night of the experiment.
                # So we remove the last day/row of the experiment from the range of experiment dates.
                subject_experiment_dates_df = subject_experiment_dates_df[:-1]
                # Convert to datetime
                subject_experiment_dates_df['ExperimentDates'] = pd.to_datetime(subject_experiment_dates_df['ExperimentDates'], format='%d/%m/%Y %H:%M')
                try:
                    subject_means_df['ExperimentDates'] = pd.to_datetime(subject_means_df['ExperimentDates'], format='%Y-%m-%d %H:%M:%S')
                except:
                    subject_means_df['ExperimentDates'] = pd.to_datetime(subject_means_df['ExperimentDates'], format='%Y-%m-%d')
                # Merge subject's sleep dataframe with the range of experiment dates.
                # Dropping the dates of sleep data that is not in dates of experiment.
                # Using 'right' join: "right: use only keys from right frame, similar to a SQL right outer join; preserve key order."
                # Creating nan rows with dates that are missing by doing a right join with the experiment dates.
                subject_experiment_dates_df = (
                    pl.DataFrame(subject_experiment_dates_df)
                    .with_columns(
                        ExperimentDates = pl.col('ExperimentDates').dt.cast_time_unit('ms')
                    )
                    .to_pandas()
                )
                subject_means_df = subject_means_df.merge(subject_experiment_dates_df, how='right',
                                                        left_on='ExperimentDates', right_on='ExperimentDates')

                # Create a new column that contains the day of the Experiment.
                # For each unique date of 'DateOfSleepEvening' column a consecutive rank value is assigned.
                # 'dense' = 'consecutive values'
                subject_means_df['DayOfExperiment'] = subject_means_df['ExperimentDates'].rank(method='dense')

                if subject_experiment_metadata_row['NotInIsrael'].values[0]:
                    non_relervant_dates = pd.date_range(subject_experiment_metadata_row['NotInIsraelStartDate'].values[0],
                                                        subject_experiment_metadata_row['NotInIsraelEndDate'].values[0])
                    subject_means_df = subject_means_df[~subject_means_df['ExperimentDates'].isin(non_relervant_dates)]
                    subject_means_df.reset_index(drop=True, inplace=True)
                
                if subject_experiment_metadata_row['NotInIsrael_1'].values[0]:
                    non_relervant_dates = pd.date_range(subject_experiment_metadata_row['NotInIsraelStartDate_1'].values[0],
                                                        subject_experiment_metadata_row['NotInIsraelEndDate_1'].values[0])
                    subject_means_df = subject_means_df[~subject_means_df['ExperimentDates'].isin(non_relervant_dates)]
                    subject_means_df.reset_index(drop=True, inplace=True)


                # We want to find the start and end of each day.
                # We are doing it by iterating over the sleep data and for each day and find the start and end of the day by calculating the start and end of the sleep.
                # Also, We take the end of the sleep of yesterday , which is the wake up hour of today, and add 1 minute to it and that is the start of the day.
                # We take the end of the sleep of today and subtract 1 minute from it and that is the end of the day.
                # For each sleep, we will select the data between start and end of the current day.
                # Then, for each day we iterate over the day features (which are different activities) and calculate different measures: mean, std, cv, min, max, count, skew and empirical_skew_by_noa_from_paper.
                subject_means_df['StartOfDay'] = np.nan
                subject_means_df['EndOfDay'] = np.nan
                # Iterate over the sleep data of the subject
                # TODO: if there is less then 3 nights dont use it.
                for i, sleep_row in subject_means_df.iterrows():
                    if not pd.isna(sleep_row['SleepEndTime']):

                        # skipping the first day because there is no data for the first day.
                        # Only calculate of the sleep features!
                        if i == 0:
                            # Calculate only the night features/data for the first day
                            # Using BedStartTime and BedEndTime to get the 'pre_sleep', 'sleep'and 'post_wake' features
                            start_of_sleep = subject_means_df.loc[i, 'BedStartTime']
                            end_of_sleep = subject_means_df.loc[i, 'BedEndTime']
                            # Select the data that between start_of_sleep and end_of_sleep
                            relevant_hr_steps_sleep_df = hr_steps_sleep_df.loc[hr_steps_sleep_df['DateAndMinute'].between(start_of_sleep, end_of_sleep)]
                            # Filter outliers
                            relevant_hr_steps_sleep_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['outliers']==False]
                            # select only rows/minutes that are not weekend.
                            if exclude_weekends:
                                relevant_hr_steps_sleep_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['Weekend']==False]
                            # Update current day features with the current day features of the subject
                            subject_current_day_features = relevant_hr_steps_sleep_df['Feature'].unique()
                            # Check if Feature_2 is exists and add it to the list of subject_current_day_features
                            if 'Feature_2' in relevant_hr_steps_sleep_df.columns:
                                subject_current_day_features = np.append(subject_current_day_features, relevant_hr_steps_sleep_df['Feature_2'].unique())
                                subject_features_set.update(subject_current_day_features)
                                # remove nan
                                subject_features_set = {x for x in subject_features_set if pd.notnull(x)}


                            # Iterate over sleep features only and calculate different measures: mean, std, cv, min, max, count, skew and empirical_skew_by_noa_from_paper.
                            for feature in subject_current_day_features:
                                if feature in ['sleep', 'pre_sleep', 'post_wake']:
                                    # filter rows of current filter where  outliers == True ( meaning the its not an outlier.)
                                    relevant_feature_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['Feature'] == feature]
                                    subject_means_df.loc[i, feature + '_bpm_mean'] = relevant_feature_df['BpmMean'].mean()
                                    subject_means_df.loc[i, feature + '_bpm_std'] = relevant_feature_df['BpmMean'].std()
                                    subject_means_df.loc[i, feature + '_bpm_cv'] = relevant_feature_df['BpmMean'].std() / relevant_feature_df['BpmMean'].mean()
                                    subject_means_df.loc[i, feature + '_bpm_min'] = relevant_feature_df['BpmMean'].min()
                                    subject_means_df.loc[i, feature + '_bpm_max'] = relevant_feature_df['BpmMean'].max()
                                    subject_means_df.loc[i, feature + '_bpm_count'] = relevant_feature_df['BpmMean'].count()
                                    # Set nan to 0 in the count column for weights calculation
                                    subject_means_df.loc[subject_means_df[feature + '_bpm_count'] == 0, feature + '_bpm_count'] = np.nan
                                    subject_means_df.loc[i, feature + '_bpm_skew'] = relevant_feature_df['BpmMean'].skew()
                                    subject_means_df.loc[i, feature + '_empirical_skew_by_noa_from_paper'] = (3 * (relevant_feature_df['BpmMean'].mean() - relevant_feature_df['BpmMean'].median())) / relevant_feature_df['BpmMean'].std()
                                    # minutes_percentage_of_ can't be calculated because we don't have the BpmMean of the subject.  (only for first day)
                            continue

                        # If its not the first day of the experiment do this:
                        # If the SleepEndTime of yesterday is nan or the SleepEndTime of today is nan, skip this day.
                        if pd.isna(subject_means_df.loc[i-1, 'SleepEndTime']):
                            continue
                        elif pd.isna(subject_means_df.loc[i, 'SleepEndTime']):
                            continue
                        # find the start time of the day by calculating: SleepEndTime of the yesterday's sleep + 1 minute (Assumption)
                        start_of_day = pd.to_datetime(subject_means_df.loc[i-1, 'SleepEndTime']) + pd.Timedelta(minutes=1)
                        # find the end time of the day by calculating the SleepStartTime of current day.
                        end_of_day = pd.to_datetime(subject_means_df.loc[i, 'SleepEndTime'])

                        # Select the data that between start_of_day and end_of_day
                        relevant_hr_steps_sleep_df = hr_steps_sleep_df.loc[hr_steps_sleep_df['DateAndMinute'].between(start_of_day, end_of_day)]
                        # Filter outliers
                        relevant_hr_steps_sleep_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['outliers'] == False]
                        # if exclude_weekends=True - select only rows/minutes that are not weekend.
                        if exclude_weekends:
                            relevant_hr_steps_sleep_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['Weekend'] == False]
                        # Update current day features with the current day features of the subject, it will be empty if its Weekend so the next loop will be skipped.
                        subject_current_day_features = relevant_hr_steps_sleep_df['Feature'].unique()
                        # Check if Feature_2 is exists and add it to the list of subject_current_day_features
                        if 'Feature_2' in relevant_hr_steps_sleep_df.columns:
                            subject_current_day_features = np.append(subject_current_day_features, relevant_hr_steps_sleep_df['Feature_2'].unique())
                            # remove nan
                            subject_current_day_features = [x for x in subject_current_day_features if pd.notnull(x)]
                        subject_features_set.update(subject_current_day_features)
                        # Create a new columns StartOfDay and EndOfDay and set the start and end of the day
                        subject_means_df.loc[i, 'StartOfDay'] = start_of_day
                        subject_means_df.loc[i, 'EndOfDay'] = end_of_day

                        # Create a new column sum_of_steps_that_sampled and calculate the sum of steps that sampled in the current day
                        subject_means_df.loc[i, 'sum_of_steps_that_sampled'] = relevant_hr_steps_sleep_df['StepsInMinute'].sum(axis=0)
                        # Create a new column mean_of_steps_that_sampled and calculate the mean of steps that sampled in the current day
                        subject_means_df.loc[i, 'mean_of_steps_that_sampled'] = relevant_hr_steps_sleep_df['StepsInMinute'].mean(axis=0)
                        # Create a new column std_of_steps_that_sampled and calculate the standard deviation of steps that sampled in the current day
                        subject_means_df.loc[i, 'std_of_steps_that_sampled'] = relevant_hr_steps_sleep_df['StepsInMinute'].std(axis=0)
                        if subject_means_df.loc[i, 'sum_of_steps_that_sampled'] == 0:
                            subject_means_df.loc[i, 'sum_of_steps_that_sampled'] = np.nan
                            subject_means_df.loc[i, 'mean_of_steps_that_sampled'] = np.nan
                            subject_means_df.loc[i, 'std_of_steps_that_sampled'] = np.nan

                        # old code. to delete after validate that the function works well.
                        # # Filter outliers
                        # relevant_hr_steps_sleep_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['outliers']==False]
                        # # select only rows/minutes that are not weekend.
                        # if exclude_weekends:
                        #     relevant_hr_steps_sleep_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['Weekend']==False]
                        # Iterate over the sleep features for each day and calculate the mean, std, cv, min, max, count, skew.
                        for feature in subject_current_day_features:
                            # filter rows of current filter where  outliers == True ( meaning the its not an outlier.)
                            if 'Feature_2' in relevant_hr_steps_sleep_df.columns:
                                relevant_feature_df = relevant_hr_steps_sleep_df.loc[(relevant_hr_steps_sleep_df['Feature'] == feature) |
                                                                                    (relevant_hr_steps_sleep_df['Feature_2'] == feature)]
                            else:
                                relevant_feature_df = relevant_hr_steps_sleep_df.loc[relevant_hr_steps_sleep_df['Feature'] == feature]
                                # mean of the heart rate during feature
                            subject_means_df.loc[i, feature+'_bpm_mean'] = relevant_feature_df['BpmMean'].mean()
                            # std of the heart rate during feature
                            subject_means_df.loc[i, feature+'_bpm_std'] = relevant_feature_df['BpmMean'].std()
                            # cv of the heart rate during feature
                            subject_means_df.loc[i, feature+'_bpm_cv'] = relevant_feature_df['BpmMean'].std() / relevant_feature_df['BpmMean'].mean()
                            # min of the heart rate during feature
                            subject_means_df.loc[i, feature+'_bpm_min'] = relevant_feature_df['BpmMean'].min()
                            # max of the heart rate during feature
                            subject_means_df.loc[i, feature+'_bpm_max'] = relevant_feature_df['BpmMean'].max()
                            # count of the heart rate during feature
                            subject_means_df.loc[i, feature+'_bpm_count'] = relevant_feature_df['BpmMean'].count()
                            # Set nan to 0 in the count column for weights calculation
                            subject_means_df.loc[subject_means_df[feature + '_bpm_count'] == 0, feature + '_bpm_count'] = np.nan
                            # skew of the heart rate during feature
                            subject_means_df.loc[i, feature+'_bpm_skew'] = relevant_feature_df['BpmMean'].skew()
                            # empirical skew that Noa Magal found in a paper.
                            subject_means_df.loc[i, feature+'_empirical_skew_by_noa_from_paper'] = (3 * (relevant_feature_df['BpmMean'].mean() - relevant_feature_df['BpmMean'].median())) / relevant_feature_df['BpmMean'].std()
                            # percentage of current feature from all the features where outliers are filtered
                            subject_means_df.loc[i, 'minutes_percentage_of_' + feature] = relevant_feature_df['BpmMean'].count() / relevant_hr_steps_sleep_df['BpmMean'].count() * 100
                            # Set nan to 0 in the minutes_percentage_of_ column for weights calculation
                            subject_means_df.loc[subject_means_df['minutes_percentage_of_' + feature] == 0, 'minutes_percentage_of_' + feature] = np.nan

                # Save output CSV file to each subject output folder and output the DataFrame to a CSV file
                subject_output_path = OUTPUT_PATH.joinpath(subject)
                subject_output_path_history = ut.output_record(OUTPUT_PATH, subject, user_name, now)
                
                if not subject_output_path_history.exists():
                    os.makedirs(subject_output_path_history)
                    
                subject_means_df['Id'] = subject
                if exclude_weekends:
                    subject_means_df.to_csv(subject_output_path_history.joinpath(f'{subject} Metrics of Heart Rate By Activity (exclude weekends).csv'),index=False)
                else:
                    subject_means_df.to_csv(subject_output_path_history.joinpath(f'{subject} Metrics of Heart Rate By Activity.csv'), index=False) 

                ut.check_for_duplications(subject_output_path, subject_output_path_history)
                
                subject_means_df['Id'] = subject
                all_subjects_details_df = pd.concat([all_subjects_details_df, subject_means_df])


                # All the rows of subject_means_df are full now.
                # Create 2 new files - one for the full week and one for the full week without the weekend. (thursday's start sleep time to sunday wake time)
                subject_full_week_dict = {}
                subject_full_week_dict['Id'] = subject
                if len(subject_features_set) == 0:
                    continue
                # indicate that there is something like 1 row or not enough data for the subject
                if 'mean_of_steps_that_sampled' not in subject_means_df.columns:
                    continue
                subject_full_week_dict['mean_of_steps_that_sampled_mean'] = subject_means_df['mean_of_steps_that_sampled'].mean()
                subject_full_week_dict['std_of_steps_mean'] = subject_means_df['std_of_steps_that_sampled'].mean()
                subject_full_week_dict['sum_of_steps_mean'] = subject_means_df['sum_of_steps_that_sampled'].mean()
                # start calculating statistics for each feature @TODO
                for feature in subject_features_set:
                    if subject_means_df[feature+'_bpm_count'].dropna().empty:
                        continue
                    subject_full_week_dict[feature+'_bpm_mean'] = np.average(subject_means_df[feature+'_bpm_mean'].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna())
                    if subject_means_df[feature+'_bpm_std'].dropna().empty:
                        subject_full_week_dict[feature+'_bpm_std'] = np.nan
                    else:
                        subject_full_week_dict[feature+'_bpm_std'] = np.average(subject_means_df[feature+'_bpm_std'].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna().reindex(subject_means_df[feature+'_bpm_std'].dropna().index))
                    if subject_means_df[feature+'_bpm_cv'].dropna().empty:
                        subject_full_week_dict[feature + '_bpm_cv'] = np.nan
                    else:
                        subject_full_week_dict[feature+'_bpm_cv'] = np.average(subject_means_df[feature+'_bpm_cv'].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna().reindex(subject_means_df[feature+'_bpm_cv'].dropna().index))
                    if subject_means_df[feature+'_bpm_skew'].dropna().empty:
                        subject_full_week_dict[feature+'_bpm_skew'] = np.nan
                    else:
                        subject_full_week_dict[feature+'_bpm_skew'] = np.average(subject_means_df[feature+'_bpm_skew'].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna().reindex(subject_means_df[feature+'_bpm_skew'].dropna().index))
                    if subject_means_df[feature+'_empirical_skew_by_noa_from_paper'].dropna().empty:
                        subject_full_week_dict[feature+'_empirical_skew_by_noa_from_paper'] = np.nan
                    else:
                        subject_full_week_dict[feature+'_empirical_skew_by_noa_from_paper'] = np.average(subject_means_df[feature+'_empirical_skew_by_noa_from_paper'].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna().reindex(subject_means_df[feature+'_empirical_skew_by_noa_from_paper'].dropna().index))

                    subject_full_week_dict[feature+'_bpm_min'] = np.average(subject_means_df[feature+'_bpm_min'].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna())
                    subject_full_week_dict[feature+'_bpm_max'] = np.average(subject_means_df[feature+'_bpm_max'].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna())
                    subject_full_week_dict[feature+'_bpm_count'] = subject_means_df[feature+'_bpm_count'].dropna().sum()
                    subject_full_week_dict['minutes_percentage_of_' + feature] = np.average(subject_means_df['minutes_percentage_of_' + feature].dropna(), weights=subject_means_df[feature+'_bpm_count'].dropna().reindex(subject_means_df['minutes_percentage_of_' + feature].dropna().index))
                    # add a column that indicates on how many days the average is based on
                    subject_full_week_dict[feature+'_days_based'] = subject_means_df[feature+'_bpm_count'].dropna().count()
                    subject_full_week_dict[feature + '_bpm_based'] = subject_means_df[feature + '_bpm_count'].dropna().sum()

                    ### Add sleep_regularity_index column ###
                # According to: * https://www.jmir.org/2018/6/e210/#figure5 *
                # Create new temporary dataframe for the sleep_regularity_index calculation
                temp_df = pd.DataFrame()
                temp_df['Mode'] = hr_steps_sleep_df['Mode']
                # Replace "unknown" to np.nan and 'in_bed' to 'awake' and convert to int:
                temp_df['Mode'] = temp_df['Mode'].replace({"sleeping": -1, "in_bed": 1, "awake": 1, "unknown": np.nan})
                # τ = 1440 minutes (from the formula)
                minutes_in_a_day = 1440
                temp_df['24h_lag_Mode'] = temp_df[["Mode"]].shift(minutes_in_a_day)
                # Remove the last 1440 rows (doesn't have shift data anyway and its according to the formula mentioned in comment above.)
                temp_df = temp_df[0:len(temp_df) - minutes_in_a_day]
                # Calculate the product of s(t)*s(t+τ)
                temp_df["Modes_multiply"] = temp_df["Mode"] * temp_df["24h_lag_Mode"]
                # Sum of products
                sum_of_products = temp_df["Modes_multiply"].sum()
                # Total number of valid minutes (count() ignores nan values)
                denominator = temp_df["Modes_multiply"].count() - minutes_in_a_day

                # Calculate the final sleep regularity index :
                sri = 200 * ((sum_of_products / denominator) - 0.5)
                # Add the sleep_regularity_index to the subject_full_week_dict
                subject_full_week_dict['sleep_regularity_index'] = sri

                subjects_full_week_means_list.append(subject_full_week_dict)
            # Create a DataFrame from the lists of dictionaries # Save output CSV file to project aggregated output folder
            subjects_full_week_means_df = pd.DataFrame(subjects_full_week_means_list)

            # Validate that the DataFrame is not empty
            if subjects_full_week_means_df.empty:
                print('Missing Heart Rate and Steps and Sleep Aggregated.csv files in the subjects folders.')
                print('can\'t create \"Full Week of Heart Rate Metrics By Activity\" and \"No Weekends Summary of Heart Rate Metrics By Activity\" files')
                return

            # sort by id column and reset index
            subjects_full_week_means_df = subjects_full_week_means_df.sort_values(by=['Id']).reset_index(drop=True)
            all_subjects_details_df = all_subjects_details_df.sort_values(by=['Id', 'SleepStartTime']).reset_index(drop=True)
            # Remove the columns that start with 'unknown_'
            subjects_full_week_means_df = subjects_full_week_means_df.loc[:, ~subjects_full_week_means_df.columns.str.startswith('unknown_')]
            all_subjects_details_df = all_subjects_details_df.loc[:, ~all_subjects_details_df.columns.str.startswith('unknown_')]
            # Remove the columns that end with 'unknown_'
            subjects_full_week_means_df = subjects_full_week_means_df.loc[:, ~subjects_full_week_means_df.columns.str.endswith('_unknown')]
            all_subjects_details_df = all_subjects_details_df.loc[:, ~all_subjects_details_df.columns.str.endswith('_unknown')]
            
            
            
            if exclude_weekends:
                old_subjects_full_week_means_df = ut.new_get_latest_file_by_term('No Weekends Summary of Heart Rate Metrics By Activity', root=AGGREGATED_OUTPUT_PATH)
                if old_subjects_full_week_means_df.exists():
                    old_subjects_full_week_means_df = pd.read_csv(old_subjects_full_week_means_df)
                    for subject in subjects_full_week_means_df['Id']:
                        if subject in old_subjects_full_week_means_df['Id'].values:
                            old_subjects_full_week_means_df = old_subjects_full_week_means_df[old_subjects_full_week_means_df['Id'] != subject]
                    subjects_full_week_means_df = pd.concat([old_subjects_full_week_means_df, subjects_full_week_means_df])
                old_all_subjects_details_df = ut.new_get_latest_file_by_term('No Weekends All Subjects of Heart Rate Metrics By Activity', root=AGGREGATED_OUTPUT_PATH)
                if old_all_subjects_details_df.exists():
                    old_all_subjects_details_df = pd.read_csv(old_all_subjects_details_df)
                    for subject in all_subjects_details_df['Id']:
                        if subject in old_all_subjects_details_df['Id'].values:
                            old_all_subjects_details_df = old_all_subjects_details_df[old_all_subjects_details_df['Id'] != subject]
                    all_subjects_details_df = pd.concat([old_all_subjects_details_df, all_subjects_details_df])
                    
                subjects_full_week_means_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'No Weekends Summary of Heart Rate Metrics By Activity.csv'), index=False) 
                all_subjects_details_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'No Weekends All Subjects of Heart Rate Metrics By Activity.csv'), index=False)
            else:
                old_subjects_full_week_means_df = ut.new_get_latest_file_by_term('Full Week Summary of Heart Rate Metrics By Activity', root=AGGREGATED_OUTPUT_PATH)
                if old_subjects_full_week_means_df.exists():
                    old_subjects_full_week_means_df = pd.read_csv(old_subjects_full_week_means_df)
                    for subject in subjects_full_week_means_df['Id']:
                        if subject in old_subjects_full_week_means_df['Id'].values:
                            old_subjects_full_week_means_df = old_subjects_full_week_means_df[old_subjects_full_week_means_df['Id'] != subject]
                    subjects_full_week_means_df = pd.concat([old_subjects_full_week_means_df, subjects_full_week_means_df])
                old_all_subjects_details_df = ut.new_get_latest_file_by_term('Full Week All Subjects of Heart Rate Metrics By Activity', root=AGGREGATED_OUTPUT_PATH)
                if old_all_subjects_details_df.exists():
                    old_all_subjects_details_df = pd.read_csv(old_all_subjects_details_df)
                    for subject in all_subjects_details_df['Id']:
                        if subject in old_all_subjects_details_df['Id'].values:
                            old_all_subjects_details_df = old_all_subjects_details_df[old_all_subjects_details_df['Id'] != subject]
                    all_subjects_details_df = pd.concat([old_all_subjects_details_df, all_subjects_details_df])
                    
                subjects_full_week_means_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Full Week Summary of Heart Rate Metrics By Activity.csv'), index=False) 
                all_subjects_details_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Full Week All Subjects of Heart Rate Metrics By Activity.csv'), index=False)

            ut.check_for_duplications(AGGREGATED_OUTPUT_PATH, AGGREGATED_OUTPUT_PATH_HISTORY)

        by_activity(exclude_weekends=False)
        by_activity(exclude_weekends=True)






def is_dst_change(date: datetime.datetime) -> bool:
    # Set the timezone to Jerusalem
    jerusalem = pytz.timezone("Asia/Jerusalem")
    
    # Localize the provided date at midnight and the next day at midnight
    midnight = jerusalem.localize(datetime.datetime(date.year, date.month, date.day))
    next_midnight = jerusalem.localize(datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(days=1))
    
    # Check if DST offset changes between the two midnights
    return midnight.dst() != next_midnight.dst() 


                    


            

            



            
    # except Exception as e:
    #     print(e)
    #     time.sleep(10)



def concate_to_old(term, path, new_df):
    old_path = ut.new_get_latest_file_by_term(term, root=path)
    if new_df.empty:
        if not old_path.exists():
            return pd.DataFrame()
        
        return pd.read_csv(old_path)
    if old_path.exists():
        old_df = pd.read_csv(old_path)
        
        for subject in new_df['Id'].unique():
            if subject in old_df['Id'].unique():
                old_df = old_df[old_df['Id'] != subject]
        new_df = pd.concat([old_df, new_df])
        new_df = new_df.sort_values(by=['Id'])
        return new_df
    else:
        return new_df



if __name__ == '__main__':
    
    try:
        param = sys.argv[1]
        now = sys.argv[2]
        user_name = sys.argv[3]

    except IndexError:
        param = 'FIBRO_TESTS'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        user_name = 'Unknown'


    print(f'Generating the files for {param}...')
    print(f'Include weekend: {include_weekend}')
    print(f'Exclude weekend: {exclude_weekend}')
    main(param, now, user_name, include_weekend, exclude_weekend)