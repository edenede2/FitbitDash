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
from datetime import datetime
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import configparser
import numpy as np
import polars as pl
import datetime
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


import UTILS.utils as ut
# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings
warnings.filterwarnings('ignore')





def main(project, now, username):
    try:
        FIRST = 0
        LAST = -1
        exeHistory_path = Path(r'.\pages\ExecutionHis\exeHistory.parquet')   

        exeHistory = pl.read_parquet(exeHistory_path)
        paths_json = json.load(open(r".\pages\Pconfigs\paths.json", "r")) 


        project_path = Path(paths_json[project])



        DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

        AGGREGATED_OUTPUT_PATH_HISTORY = ut.output_record(OUTPUT_PATH, 'Aggregated Output',username, now)

        if not AGGREGATED_OUTPUT_PATH_HISTORY.exists():
            os.makedirs(AGGREGATED_OUTPUT_PATH_HISTORY)
        
        PROJECT_CONFIG = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))


        SUBJECTS_DATES = METADATA_PATH.joinpath('Subjects Dates.csv')

        try:
            subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                        try_parse_dates=True)
        except:
            subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                        parse_dates=True,
                                        encoding='utf-8')

        subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')

        target_data_path = DATA_PATH
        if not target_data_path.exists():
            os.makedirs(target_data_path)
        # Create Output folder
        if not OUTPUT_PATH.exists():
            os.makedirs(OUTPUT_PATH)
        # Create Aggregated Output folder
        if not AGGREGATED_OUTPUT_PATH.exists():
            os.makedirs(AGGREGATED_OUTPUT_PATH)
        # Create Archive folder
        if not ARCHIVE_PATH.exists():
            os.makedirs(ARCHIVE_PATH)
        # Create Metadata folder
        if not METADATA_PATH.exists():
            os.makedirs(METADATA_PATH)

            quit()


        # Load the subjects dates of experiment file.
        subjects_dates_df = pl.read_parquet(rf'.\pages\sub_selection\{project}_sub_selection_sleep_all_subjects.parquet').sort(by='Id').unique('Id').drop_nulls('Id')    
        subjects_to_run_on = subjects_dates_df['Id']
        print(f"Subjects to run on: {subjects_to_run_on}")
        subjects_dates_df = (
            subjects_dates_df
            .select(
                pl.selectors.exclude('Date')
            )
            .join(
                subjects_dates,
                on='Id',
                how='right'
            )

        )

        subjects_dates_df = subjects_dates_df.filter(pl.col('Id').is_in(subjects_to_run_on))


        # run_it = ut.create_new_sleep_all_subjects_csv_window()
        all_subjects_sleep_df = pd.DataFrame() # Create an empty DataFrame to hold the processed sleep data for all subjects
        subjects_with_missing_sleep_files = [] # Create an empty list to hold any subjects that are missing sleep files
        
        
        # Find relevant sleep.json files
        tqdm_subjects = tqdm(subjects_dates_df['Id'])
        for subject in tqdm_subjects:
            # if run_on_specific_subjects and subject not in subjects_to_run_on:
            #     continue
            tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)

            #
            # if the folder not look like 'sub_203' so skip it
            # if not re.fullmatch(SUBJECT_FOLDER_FORMAT, subject):
            #     continue #@TODO: GET BACK IN
            # Find all 'sleep-yyyy-mm-dd.json' files from 'sub_***/FITBIT/Sleep' folder
            sleep_json_directory = DATA_PATH.joinpath(f'{subject}\FITBIT\Sleep')
            pattern = re.compile(r'^sleep-(\d{4}-\d{2}-\d{2})\.json')  # regular expression of the file format
            if not re.search(r'\d{3}$', subject):
                continue
            sleep_files = [file_name for file_name in os.listdir(sleep_json_directory) if pattern.search(file_name)]

            # if there is no sleep files so skip it and add the subject to the missing list
            if not sleep_files:
                subjects_with_missing_sleep_files.append(subject)
                continue
            # Merge all sleep.json files into a single list
            sleep_json_merged = []
            for file_name in sleep_files:
                file_path = sleep_json_directory.joinpath(file_name)
                with open(file_path, "r") as f:
                    if os.stat(file_path).st_size != 0:  # Check if the file is not empty
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"File {file_path} is not valid JSON")
                            continue
                        else:
                            sleep_json_merged.extend(data)
                    else:
                        print(f"File {file_path} is empty")
                        continue
            if sleep_json_merged == []:
                continue
            # Convert the list of dictionaries to a pandas DataFrame
            subject_sleep_df = pd.json_normalize(sleep_json_merged)  # TODO: small bottleneck
            # Drop unnecessary columns
            subject_sleep_df = subject_sleep_df.drop(columns=['type', 'infoCode', 'logType', 'logId',
                                                            'minutesToFallAsleep', 'levels.summary.deep.thirtyDayAvgMinutes',
                                                            'levels.summary.wake.thirtyDayAvgMinutes',
                                                            'levels.summary.light.thirtyDayAvgMinutes',
                                                            'levels.summary.rem.thirtyDayAvgMinutes',
                                                            'levels.shortData'], errors='ignore')
            # Rename columns
            subject_sleep_df = subject_sleep_df.rename(columns={'dateOfSleep': 'DateOfSleep (Fitbit Calculation)',
                                                                'startTime': 'BedStartTime', 'endTime': 'BedEndTime',
                                                                'minutesToFallAsleep': 'MinutesToFallAsleep?',
                                                                'minutesAsleep': 'SleepTimeInMinutes (Fitbit Calculation)',
                                                                'minutesAwake': 'MinutesAwake (Fitbit Calculation)',
                                                                'minutesAfterWakeup': 'MinutesInBedAfterWakeup (Fitbit Calculation)',
                                                                'timeInBed': 'MinutesInBed (Fitbit Calculation)',
                                                                'efficiency': 'Efficiency (Fitbit Calculation) ?',
                                                                'mainSleep': 'MainSleep (Fitbit Calculation)',
                                                                'isMainSleep': 'MainSleep (Fitbit Calculation)',
                                                                'levels.summary.deep.count': 'NightDeepCount',
                                                                'levels.summary.deep.minutes': 'NightDeepInMinutes',
                                                                'levels.summary.wake.count': 'NightAwakeCount',
                                                                'levels.summary.wake.minutes': 'NightAwakeInMinutes',
                                                                'levels.summary.light.count': 'NightLightCount',
                                                                'levels.summary.light.minutes': 'NightLightInMinutes',
                                                                'levels.summary.rem.count': 'NightRemCount',
                                                                'levels.summary.rem.minutes': 'NightRemInMinutes',
                                                                'levels.summary.restless.count': 'MidDayRestlessCount',
                                                                'levels.summary.restless.minutes': 'MidDayRestlessInMinutes',
                                                                'levels.summary.awake.count': 'MidDayAwakeCount',
                                                                'levels.summary.awake.minutes': 'MidDayAwakeInMinutes',
                                                                'levels.summary.asleep.count': 'MidDaySleepCount',
                                                                'levels.summary.asleep.minutes': 'MidDaySleepInMinutes'})
            # Convert dates columns to datetime object: '2021-11-16T00:16:30.000' --> '2021-11-16 00:16:30'
            # 'coerce': Replacing values which cannot be converted to datetime with Nan. (other options are ignore or create an error.)
            subject_sleep_df[['BedStartTime', 'BedEndTime']] = subject_sleep_df[['BedStartTime', 'BedEndTime']].apply(pd.to_datetime, errors='coerce')
            # Sort by 'DateOfSleep (Fitbit Calculation)', 'BedStartTime', and 'BedEndTime' columns
            subject_sleep_df = subject_sleep_df.sort_values(by=['DateOfSleep (Fitbit Calculation)', 'BedStartTime', 'BedEndTime'])
            # Drop duplicate rows based on 'BedStartTime' and 'BedEndTime' columns and reset the index
            subject_sleep_df = subject_sleep_df.drop_duplicates(subset=['BedStartTime', 'BedEndTime']).reset_index(drop=True)
            # Calculate the number of minutes it took to fall asleep and add it as a new column 'MinutesToFallAsleep'.
            # The *zero* element of levels.data column has 'seconds' field, which indicate the duration the subject was in bed awake (before fall asleep).
            subject_sleep_df['MinutesToFallAsleep'] = subject_sleep_df['levels.data'].apply(lambda row: row[FIRST]['seconds'] / 60 if row[FIRST]['level'] in ['wake','awake'] else 0)
            # Calculate the number of minutes spent in bed after waking up and add it as a new column 'MinutesInBedAfterWakeup'.
            # The *last* element of levels.data column has 'seconds' field, which indicate the duration the subject was in bed after wakeup.
            subject_sleep_df['MinutesInBedAfterWakeup'] = subject_sleep_df['levels.data'].apply(lambda row: row[LAST]['seconds'] / 60 if row[LAST]['level'] in ['wake','awake'] else 0)
            # Drop the 'levels.data' column. Not needed any more.
            subject_sleep_df = subject_sleep_df.drop(columns=['levels.data'])
            # Calculate the sleep start time by adding the 'MinutesToFallAsleep' to 'BedStartTime' and add the result as a new column 'SleepStartTime'.
            subject_sleep_df['SleepStartTime'] = subject_sleep_df['BedStartTime'] + pd.to_timedelta(subject_sleep_df['MinutesToFallAsleep'], unit='m')
            # Calculate the sleep end time by subtracting the 'MinutesInBedAfterWakeup' from 'BedEndTime' and add the result as a new column 'SleepEndTime'.
            subject_sleep_df['SleepEndTime'] = subject_sleep_df['BedEndTime'] - pd.to_timedelta(subject_sleep_df['MinutesInBedAfterWakeup'], unit='m')

            # Mid-day sleep data is unexpected.
            # So where the 'MinutesInBed' and 'SleepTimeInMinutes (Fitbit Calculation)' columns are equal in the same rows,
            # the SleepStartTime and SleepEndTime switched by the code above.
            # That happens because in 'levels.data' the data about wake time in bed is missing, and there is only data about sleep minutes (that why it switched.)
            # So we should switch them again.
            # If 'MinutesInBed' and 'SleepTimeInMinutes (Fitbit Calculation)' columns are not equal that mean that the fitbit gave us the seconds awake so its the true value.
            mask_mid_day_sleep_with_missing_awake_data = (subject_sleep_df['MinutesInBed (Fitbit Calculation)'] == subject_sleep_df['SleepTimeInMinutes (Fitbit Calculation)']) & \
                                                        (subject_sleep_df['MainSleep (Fitbit Calculation)'] == False)
            temp_sleep_start_time = subject_sleep_df.loc[mask_mid_day_sleep_with_missing_awake_data, 'SleepStartTime'].copy()
            subject_sleep_df.loc[mask_mid_day_sleep_with_missing_awake_data, 'SleepStartTime'] = subject_sleep_df.loc[mask_mid_day_sleep_with_missing_awake_data, 'SleepEndTime']
            subject_sleep_df.loc[mask_mid_day_sleep_with_missing_awake_data, 'SleepEndTime'] = temp_sleep_start_time
            # Create a new column that contains the weekday of the sleep start time
            subject_sleep_df['WeekdayOfSleep'] = subject_sleep_df['SleepStartTime'].dt.day_name()
            # Create a new column that contains the date of the sleep start time (without the time component)
            subject_sleep_df['DateOfSleepEvening'] = subject_sleep_df['SleepStartTime'].dt.date # if (subject_sleep_df['SleepStartTime'].dt.hour >= 20 or subject_sleep_df['SleepStartTime'].dt.date <= 8) else np.NaN
            # Convert 'DateOfSleepEvening' column to Datetime object
            subject_sleep_df['DateOfSleepEvening'] = pd.to_datetime(subject_sleep_df['DateOfSleepEvening'])
            # If the sleep start time is between 00 AM to 8 AM, the date of the sleep should be the previous day
            subject_sleep_df.loc[subject_sleep_df['SleepStartTime'].dt.hour < 8, 'DateOfSleepEvening'] -= datetime.timedelta(days=1)
            # Create a new column that contains the weekday of the evening of the sleep
            subject_sleep_df['WeekdayOfSleepEvening'] = subject_sleep_df['DateOfSleepEvening'].dt.day_name()
            # Get dates of experiment from metadata_of_project.txt file from Metadata folder.
            subject_experiment_metadata_row = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
            subject_experiment_dates = pd.date_range(subject_experiment_metadata_row['ExperimentStartDate'].values[0], subject_experiment_metadata_row['ExperimentEndDate'].values[0])

            # Convert the DatetimeIndex (subject_experiment_dates) to a DataFrame with a column named 'ExperimentStartDate'
            subject_experiment_dates_df = pd.DataFrame({'ExperimentDates': subject_experiment_dates})
            # Because we are going to merge by 'DateOfSleepEvening' column, we don't want to include the last night of the experiment.
            # So we remove the last day/row of the experiment from the range of experiment dates.
            subject_experiment_dates_df = subject_experiment_dates_df[:-1]
            # Merge subject's sleep dataframe with the range of experiment dates.
            # Dropping the dates of sleep data that is not in dates of experiment.
            # Using 'right' join: "right: use only keys from right frame, similar to a SQL right outer join; preserve key order."
            # Creating nan rows with dates that are missing by doing a right join with the experiment dates.
            subject_sleep_df = subject_sleep_df.merge(subject_experiment_dates_df, how='right', left_on='DateOfSleepEvening', right_on='ExperimentDates')

            # Create a new column that contains the day of the Experiment.
            # For each unique date of 'DateOfSleepEvening' column a consecutive rank value is assigned.
            # 'dense' = 'consecutive values'
            subject_sleep_df['DayOfExperiment'] = subject_sleep_df['ExperimentDates'].rank(method='dense')
            
            if subject_experiment_metadata_row['NotInIsrael'].values[0]:
                non_relevant_dates = pd.date_range(subject_experiment_metadata_row['NotInIsraelStartDate'].values[0], subject_experiment_metadata_row['NotInIsraelEndDate'].values[0])
                # keep only the experiment dates and dayofexperiment data, put nan on the rest of the columns (in the non relevant dates range)
                day_of_experiment = subject_sleep_df['DayOfExperiment'].copy()
                experiment_dates = subject_sleep_df['ExperimentDates'].copy()
                non_relevant_dates_mask = subject_sleep_df['ExperimentDates'].isin(non_relevant_dates)
                subject_sleep_df = subject_sleep_df.drop(columns=['DayOfExperiment', 'ExperimentDates'])
                # put nan on the columns that are not relevant 
                for column in subject_sleep_df.columns:
                    try:
                        subject_sleep_df[column] = np.where(non_relevant_dates_mask, np.nan, subject_sleep_df[column])
                    except:
                        subject_sleep_df[column] = np.where(non_relevant_dates_mask, pd.NaT, subject_sleep_df[column])
                subject_sleep_df['DayOfExperiment'] = day_of_experiment
                subject_sleep_df['ExperimentDates'] = experiment_dates
            
            if subject_experiment_metadata_row['NotInIsrael_1'].values[0]:
                non_relevant_dates = pd.date_range(subject_experiment_metadata_row['NotInIsraelStartDate_1'].values[0], subject_experiment_metadata_row['NotInIsraelEndDate_1'].values[0])
                # keep only the experiment dates and dayofexperiment data, put nan on the rest of the columns (in the non relevant dates range)
                day_of_experiment = subject_sleep_df['DayOfExperiment'].copy()
                experiment_dates = subject_sleep_df['ExperimentDates'].copy()
                non_relevant_dates_mask = subject_sleep_df['ExperimentDates'].isin(non_relevant_dates)
                subject_sleep_df = subject_sleep_df.drop(columns=['DayOfExperiment', 'ExperimentDates'])
                # put nan on the columns that are not relevant 
                for column in subject_sleep_df.columns:
                    try:
                        subject_sleep_df[column] = np.where(non_relevant_dates_mask, np.nan, subject_sleep_df[column])
                    except:
                        subject_sleep_df[column] = np.where(non_relevant_dates_mask, pd.NaT, subject_sleep_df[column])
                subject_sleep_df['DayOfExperiment'] = day_of_experiment
                subject_sleep_df['ExperimentDates'] = experiment_dates

                

            # Create a new column that contains the length of the sleep in minutes by substract 'SleepEndTime' by 'SleepStartTime',
            # then convert to seconds and finally divide by 60 to get the result in minutes.
            subject_sleep_df['BedEndTime'] = pd.to_datetime(subject_sleep_df['BedEndTime'])
            subject_sleep_df['BedStartTime'] = pd.to_datetime(subject_sleep_df['BedStartTime'])
            subject_sleep_df['DateOfSleepEvening'] = pd.to_datetime(subject_sleep_df['DateOfSleepEvening'])
            subject_sleep_df['SleepEndTime'] = pd.to_datetime(subject_sleep_df['SleepEndTime'])
            subject_sleep_df['SleepStartTime'] = pd.to_datetime(subject_sleep_df['SleepStartTime'])

            subject_sleep_df['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'] = subject_sleep_df.apply(lambda row: (row['SleepEndTime'] - row['SleepStartTime']).total_seconds() / 60, axis=1)
            # Create a new column that contains the sleep efficiency by divide 'SleepTimeInMinutes' by 'MinutesInBed'.
            subject_sleep_df['Sleep Efficiency (SE SleepTimeInMinutes/MinutesInBed Ratio)'] = subject_sleep_df.apply(
                lambda row: (row['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'] - row['MinutesAwake (Fitbit Calculation)']) / row['MinutesInBed (Fitbit Calculation)'], axis=1)

            # Create a new column that contains the time between midnight and the sleep start time in minutes
            # 22:00 - 00:00 = -02:00.total_seconds() = 120*60 / 60 = 120
            midnight = pd.Timestamp('1980-01-01 00:00:00')
            subject_sleep_df.loc[subject_sleep_df['SleepStartTime'].dt.hour > 12, 'SleepStartToMidnight(minutes)'] = -((midnight - subject_sleep_df['SleepStartTime']).dt.seconds) / 60
            subject_sleep_df.loc[subject_sleep_df['SleepStartTime'].dt.hour <= 12, 'SleepStartToMidnight(minutes)'] = ((subject_sleep_df['SleepStartTime'] - midnight).dt.seconds) / 60
            subject_sleep_df['MidnightToSleepEnd(minutes)'] = ((subject_sleep_df['SleepEndTime'] - midnight).dt.seconds) / 60
            # Create a new column that indicates whether the sleep is valid.
            subject_sleep_df['ValidSleep'] = ut.valid_sleep(subject_sleep_df)
            # Find the numbers in "sub_013" and set it to "Id" column of the dataframe.
            subject_sleep_df['Id'] = subject
            # Set to False where 'MainSleep (Fitbit Calculation)' is NaN
            subject_sleep_df['MainSleep (Fitbit Calculation)'] = subject_sleep_df['MainSleep (Fitbit Calculation)'].fillna(False)

            midday_columns = ['MidDaySleepCount', 'MidDaySleepInMinutes', 'MidDayAwakeCount', 'MidDayAwakeInMinutes', 'MidDayRestlessCount', 'MidDayRestlessInMinutes']

            for column in midday_columns:
                if column not in subject_sleep_df.columns:
                    subject_sleep_df[column] = np.nan

            # Concatenate subject's sleep dataframe with the dataframes of all subjects.
            all_subjects_sleep_df = pd.concat([all_subjects_sleep_df, subject_sleep_df])

            if isinstance(now, pd.Timestamp):
                now = now.strftime('%Y-%m-%d %H-%M-%S')
            
            # Replace underscores with spaces
            now = now.replace("_", " ")
            
            # Replace hyphens in the time part with colons
            now = now.replace("-", ":")
            
            # Convert to datetime
            now = pd.to_datetime(now)

            exeHistoryUpdate = pl.DataFrame({
                "Project": [project],
                "Subject": [subject],
                "User": [username],
                'Page': ['sleepAllSubjectsGenerator'],
                "Datetime": [now],
                "Action": ['Generate sleep all subjects file'],
            })
            exeHistory = pl.concat([exeHistory, exeHistoryUpdate], how='diagonal_relaxed')




        # Sort the combined dataframe of all subjects by the columns "Id" and "SleepStartTime"
        all_subjects_sleep_df = all_subjects_sleep_df.sort_values(by=['Id', 'ExperimentDates', 'SleepStartTime', 'SleepEndTime'])



        # # Select specific columns to reorder the columns and save the resulting DataFrame to a CSV file
        all_subjects_sleep_df = all_subjects_sleep_df[['Id', 'DayOfExperiment', 'ExperimentDates',
                                'DateOfSleepEvening', 'WeekdayOfSleepEvening',
                                'DateOfSleep (Fitbit Calculation)', 'WeekdayOfSleep',
                                'SleepStartTime', 'SleepEndTime', 'BedStartTime', 'BedEndTime',
                                'ValidSleep', 'MainSleep (Fitbit Calculation)',
                                'SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)',
                                'SleepTimeInMinutes (Fitbit Calculation)',
                                'MinutesInBed (Fitbit Calculation)',
                                'Sleep Efficiency (SE SleepTimeInMinutes/MinutesInBed Ratio)',
                                'Efficiency (Fitbit Calculation) ?',
                                'SleepStartToMidnight(minutes)', 'MidnightToSleepEnd(minutes)',
                                'MinutesToFallAsleep', 'MinutesInBedAfterWakeup',
                                'MinutesInBedAfterWakeup (Fitbit Calculation)',
                                'MinutesAwake (Fitbit Calculation)',
                                'NightDeepCount', 'NightDeepInMinutes',
                                'NightLightCount', 'NightLightInMinutes',
                                'NightRemCount', 'NightRemInMinutes',
                                'NightAwakeCount', 'NightAwakeInMinutes',
                                'MidDaySleepCount', 'MidDaySleepInMinutes',
                                'MidDayAwakeCount', 'MidDayAwakeInMinutes',
                            'MidDayRestlessCount', 'MidDayRestlessInMinutes']]
        new_df = ut.concate_to_old('Sleep All Subjects', AGGREGATED_OUTPUT_PATH, all_subjects_sleep_df)
        new_df['MainSleep (Fitbit Calculation)'] = [bool(x) for x in new_df['MainSleep (Fitbit Calculation)']]
        new_df = new_df.sort_values(by=['Id', 'ExperimentDates', 'SleepStartTime', 'SleepEndTime'])
        new_df.to_csv(AGGREGATED_OUTPUT_PATH.joinpath('Sleep All Subjects.csv'), index=False)
        exeHistory.write_parquet(exeHistory_path)

        ut.check_for_duplications(AGGREGATED_OUTPUT_PATH, AGGREGATED_OUTPUT_PATH_HISTORY)

        def sleep_summary(exclude_thursday, exclude_friday):
            # find the latest sleep file that exists in the Aggregated Output folder. if it's not exists, return.
            latest_sleep_file_path = ut.new_get_latest_file_by_term("Sleep All Subjects", root=AGGREGATED_OUTPUT_PATH)
            
            if not latest_sleep_file_path.exists():
                print('Can\'t calculate nights because Sleep All Subjects.csv is missing from <Aggregated Output> folder')
                return
            # read the latest sleep file
            raw_sleep_df = pd.read_csv(latest_sleep_file_path, parse_dates=['BedStartTime', 'DateOfSleepEvening', 'ExperimentDates'])
            # Use only valid sleep rows
            raw_sleep_df = raw_sleep_df[(raw_sleep_df['ValidSleep'] == True) & (raw_sleep_df['MainSleep (Fitbit Calculation)'] == True)]
            # get the list of subjects
            subjects = raw_sleep_df['Id'].unique().tolist()
            relevant_columns_to_aggregate = ['MinutesInBed (Fitbit Calculation)',
                                            'SleepTimeInMinutes (Fitbit Calculation)',
                                            'SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)',
                                            'MinutesAwake (Fitbit Calculation)',
                                            'Sleep Efficiency (SE SleepTimeInMinutes/MinutesInBed Ratio)',
                                            'MinutesToFallAsleep',
                                            'NightAwakeCount',
                                            'SleepStartToMidnight(minutes)',
                                            'MidnightToSleepEnd(minutes)']

            # Find the maximum length of a subject's dataframe.
            # Using 'max_length' in the *next* loop to fill the missing rows with NaN values
            max_length = 0
            
            # check if the variable 'run_on' is already defined
            
            
            for subject in subjects:
                if not re.search(r'\d{3}$', subject):
                    continue
                
                # Find current subject data/rows and use only valid rows
                subject_sleep_df = raw_sleep_df[raw_sleep_df['Id'] == subject]
                if subject_sleep_df.shape[0] > max_length:
                    max_length = subject_sleep_df.shape[0]

            daily_details_sleep_subjects, daily_summary_sleep_subjects = [], []
            print(f'\n Aggregate and Summarize Sleep Data | exclude_thursday={exclude_thursday}, exclude_friday={exclude_friday}')
            
            tqdm_subjects = tqdm(subjects)
            for subject in tqdm_subjects:
                if not re.search(r'\d{3}$', subject):
                    continue

                tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)
                subject_detailed_dict = {}
                # Select current subject data/rows
                subject_sleep_df = raw_sleep_df[raw_sleep_df['Id'] == subject]
                # Insert column named 'Id' to subject_output_df with the current subject number
                subject_detailed_dict['Id'] = subject
                subject_detailed_dict['FirstDayOfExperiment'] = pd.to_datetime(subject_sleep_df.iloc[0]['ExperimentDates']).day_name()
                if exclude_thursday:
                    subject_detailed_dict['NumberOfValidSleeps'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name() != 'Thursday'].shape[0]
                    subject_detailed_dict['SumOfSleepValidSleeps'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name() != 'Thursday']['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'].sum()
                elif exclude_friday:
                    subject_detailed_dict['NumberOfValidSleeps'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name() != 'Friday'].shape[0]
                    subject_detailed_dict['SumOfSleepValidSleeps'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name() != 'Friday']['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'].sum()
                elif exclude_thursday and exclude_friday:
                    subject_detailed_dict['NumberOfValidSleeps'] = subject_sleep_df.loc[(pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name() != 'Thursday') & (subject_sleep_df['DateOfSleepEvening'].dt.day_name() != 'Friday')].shape[0]
                    subject_detailed_dict['SumOfSleepValidSleeps'] = subject_sleep_df.loc[(pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name() != 'Thursday') & (subject_sleep_df['DateOfSleepEvening'].dt.day_name() != 'Friday')]['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'].sum()
                else:
                    subject_detailed_dict['NumberOfValidSleeps'] = subject_sleep_df.shape[0]
                    subject_detailed_dict['SumOfSleepValidSleeps'] = subject_sleep_df['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'].sum()
                for column in relevant_columns_to_aggregate:
                    for row_index in range(max_length):
                        if row_index >= subject_sleep_df.shape[0]:
                            subject_detailed_dict[f'{column} Day {row_index + 1}'] = np.nan
                            continue
                        current_day = pd.to_datetime(subject_sleep_df.iloc[row_index]['DateOfSleepEvening']).day_name()
                        # Skipping Thursday nights
                        if exclude_thursday and current_day == 'Thursday':
                            subject_detailed_dict[f'{column} Day {row_index + 1}'] = np.nan
                            continue
                        # Skipping Friday nights
                        if exclude_friday and current_day == 'Friday':
                            subject_detailed_dict[f'{column} Day {row_index + 1}'] = np.nan
                            continue
                        # General/regular/common case
                        subject_detailed_dict[f'{column} Day {row_index + 1}'] = subject_sleep_df.iloc[row_index][column]
                # Append current subject data to daily_details_sleep_subjects
                daily_details_sleep_subjects.append(subject_detailed_dict)

                subject_summary_dict = {}
                subject_summary_dict['Id'] = subject
                subject_summary_dict['FirstDayOfExperiment'] = subject_detailed_dict['FirstDayOfExperiment']
                subject_summary_dict['NumberOfValidSleeps'] = subject_detailed_dict['NumberOfValidSleeps']
                subject_summary_dict['SumOfSleepValidSleeps'] = subject_detailed_dict['SumOfSleepValidSleeps']
                for column in relevant_columns_to_aggregate:
                    if exclude_thursday and exclude_friday:
                        subject_summary_dict[f'Mean {column}'] = subject_sleep_df.loc[(pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday') & (pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday')][column].mean()
                    elif exclude_thursday:
                        subject_summary_dict[f'Mean {column}'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday'][column].mean()
                    elif exclude_friday:
                        subject_summary_dict[f'Mean {column}'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday'][column].mean()
                    else:
                        subject_summary_dict[f'Mean {column}'] = subject_sleep_df[column].mean()
                for column in relevant_columns_to_aggregate:
                    if exclude_thursday and exclude_friday:
                        subject_summary_dict[f'Std {column}'] = subject_sleep_df.loc[(pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday') & (pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday')][column].std()
                    if exclude_thursday:
                        subject_summary_dict[f'Std {column}'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday'][column].std()
                    elif exclude_friday:
                        subject_summary_dict[f'Std {column}'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday'][column].std()
                    else:
                        subject_summary_dict[f'Std {column}'] = subject_sleep_df[column].std()
                for column in relevant_columns_to_aggregate:
                    if exclude_thursday and exclude_friday:
                        subject_summary_dict[f'CV {column}'] = subject_sleep_df.loc[(pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday') & (pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday')][column].std() / subject_sleep_df.loc[(pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday') & (pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday')][column].mean()
                    elif exclude_thursday:
                        subject_summary_dict[f'CV {column}'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday'][column].std() / subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Thursday'][column].mean()
                    elif exclude_friday:
                        subject_summary_dict[f'CV {column}'] = subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday'][column].std() / subject_sleep_df.loc[pd.to_datetime(subject_sleep_df['DateOfSleepEvening']).dt.day_name()!='Friday'][column].mean()
                    else:
                        subject_summary_dict[f'CV {column}'] = subject_sleep_df[column].std() / subject_sleep_df[column].mean()
                # Append current subject data to daily_summary_sleep_subjects
                daily_summary_sleep_subjects.append(subject_summary_dict)

            # Create dataframe from daily_details_sleep_subjects
            daily_details_sleep_subjects_df = pd.DataFrame(daily_details_sleep_subjects)
            # Create dataframe from daily_summary_sleep_subjects
            daily_summary_sleep_subjects_df = pd.DataFrame(daily_summary_sleep_subjects)

            # Save output CSV file to each subject output folder
            if exclude_thursday and exclude_friday:
                daily_details_sleep_subjects_df = concate_to_old('Sleep Daily Details Exclude Thursday and Friday', AGGREGATED_OUTPUT_PATH, daily_details_sleep_subjects_df)
                daily_summary_sleep_subjects_df = concate_to_old('Sleep Daily Summary Exclude Thursday and Friday', AGGREGATED_OUTPUT_PATH, daily_summary_sleep_subjects_df)
                details_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Details Exclude Thursday and Friday.csv')
                summary_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Summary Exclude Thursday and Friday.csv')
            elif exclude_thursday:
                daily_details_sleep_subjects_df = concate_to_old('Sleep Daily Details Exclude Thursday', AGGREGATED_OUTPUT_PATH, daily_details_sleep_subjects_df)
                daily_summary_sleep_subjects_df = concate_to_old('Sleep Daily Summary Exclude Thursday', AGGREGATED_OUTPUT_PATH, daily_summary_sleep_subjects_df)
                details_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Details Exclude Thursday.csv')
                summary_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Summary Exclude Thursday.csv')
            elif exclude_friday:
                daily_details_sleep_subjects_df = concate_to_old('Sleep Daily Details Exclude Friday', AGGREGATED_OUTPUT_PATH, daily_details_sleep_subjects_df)
                daily_summary_sleep_subjects_df = concate_to_old('Sleep Daily Summary Exclude Friday', AGGREGATED_OUTPUT_PATH, daily_summary_sleep_subjects_df)
                details_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Details Exclude Friday.csv')
                summary_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Summary Exclude Friday.csv')
            else:
                daily_details_sleep_subjects_df = concate_to_old('Sleep Daily Details Full Week', AGGREGATED_OUTPUT_PATH, daily_details_sleep_subjects_df)
                daily_summary_sleep_subjects_df = concate_to_old('Sleep Daily Summary Full Week', AGGREGATED_OUTPUT_PATH, daily_summary_sleep_subjects_df)
                details_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Details Full Week.csv')
                summary_output_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Sleep Daily Summary Full Week.csv')

            # # Save output CSV file to aggregated output folder
            daily_details_sleep_subjects_df.to_csv(details_output_path, index=False) 
            daily_summary_sleep_subjects_df.to_csv(summary_output_path, index=False)
            
            ut.check_for_duplications(AGGREGATED_OUTPUT_PATH, AGGREGATED_OUTPUT_PATH_HISTORY)
        

        sleep_summary(True, False)
        sleep_summary(False, True)
        sleep_summary(True, True)
        sleep_summary(False, False)


    except Exception as e:
        print(e)
        time.sleep(10)






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
        param = 'NOVA_TESTS'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        user_name = 'Unknown'

    log_path = f'./logs/sleepAllSubjectsScript_{now}.log'
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(f'param: {param}, now: {now}, user_name: {user_name}')


    print(param, now, user_name)
    main(param, now, user_name)
    logging.info('Script finished successfully')

    
    time.sleep(15)