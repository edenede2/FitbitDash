from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import bioread as br
import h5py
import pytz
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
        selected_subjects_path = Path(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_EDA.parquet')
    else:
        selected_subjects_path = Path(rf'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_EDA.parquet') 
        
    subjects_to_run_on = []

    if not selected_subjects_path.exists():
        print(f'Selected subjects file does not exist for {project}')
        time.sleep(10)
        quit()
    else:
        selected_subjects = pl.read_parquet(selected_subjects_path)
    
        subjects_to_run_on = selected_subjects['subject'].to_list()

    run_on_specific_subjects = True

    subjects_with_missing_stress_files = []
    subjects_with_weird_values = []
   
    # Create an empty list to hold any subjects that are missing any type of files
    subjects_with_missing_computed_temperature_files_files = []
    subjects_with_missing_json_files = []
    subjects_with_missing_respiratory_files = []
    subjects_with_missing_HRV_details_files = []
    subjects_with_missing_device_temperature_files = []
    print('\n Process Hrv, Temperature and Respiratory at sleep')
    
    # Find relevant HRV files and respiratory files
    tqdm_subjects = tqdm(os.listdir(DATA_PATH))
    
    for subject in tqdm_subjects:
        if not re.search(r'\d{3}$', subject):
            continue
        if run_on_specific_subjects and subject not in subjects_to_run_on:
            continue
        if subject == 'sub_018':
            print('sub_018')
        tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)
        # if the folder not look like 'sub_203' so skip it
        # if not re.fullmatch(SUBJECT_FOLDER_FORMAT, subject):
        #     continue #@TODO: GET BACK IN

        # Set up the path to the subject's sleep folder
        sleep_json_directory = DATA_PATH.joinpath(f'{subject}\FITBIT\Sleep')


        json_files_pattern = re.compile(r'^sleep-\d{4}-\d{2}-\d{2}.json')

        json_sleep_files = [file_name for file_name in os.listdir(sleep_json_directory)
                                if json_files_pattern.search(file_name)]
        
        if not json_sleep_files:
            subjects_with_missing_json_files.append(subject)
            continue
        
        # open the json files and merge them into one dataframe
        sleep_json_merged = []
        for file_name in json_sleep_files:
            with open(sleep_json_directory.joinpath(file_name), "r") as f:
                sleep_json_merged.extend(json.load(f))
                
                
        # find the latest sleep file that exists in the Aggregated Output folder. if it's not exists, return.
        latest_sleep_file_path = ut.new_get_latest_file_by_term("Sleep All Subjects", root=AGGREGATED_OUTPUT_PATH)
        if not latest_sleep_file_path.exists():
            print('Can\'t calculate nights because Sleep Daily Summary Full Week.csv is missing from <Aggregated Output> folder')
            return None

        # Read the latest sleep file
        sleep_df = pd.read_csv(latest_sleep_file_path,
                               usecols=['Id', 'MainSleep (Fitbit Calculation)', 'ValidSleep',
                                        'SleepStartTime', 'SleepEndTime', 'BedStartTime', 'BedEndTime',
                                        'DateOfSleep (Fitbit Calculation)', 'DateOfSleepEvening', 'DayOfExperiment'],
                               dtype={'MainSleep (Fitbit Calculation)': bool,
                                      'ValidSleep': bool},
                               parse_dates=['BedStartTime', 'BedEndTime',
                                            'SleepStartTime', 'SleepEndTime',
                                            'DateOfSleep (Fitbit Calculation)', 'DateOfSleepEvening'])
        # Use only valid rows
        sleep_df = sleep_df[(sleep_df['ValidSleep'] == True) & (sleep_df['MainSleep (Fitbit Calculation)'] == True)]

        # filter and use only relevant current subject data
        subject_sleep_data = sleep_df[sleep_df['Id'] == subject]

        # Get start and end dates of experiment for current subject
        subject_dates = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
        # create a DatetimeIndex with the range of experiment dates
        subject_experiment_dates = pd.date_range(subject_dates['ExperimentStartDate'].values[0],
                                                 subject_dates['ExperimentEndDate'].values[0])

        # Convert the DatetimeIndex (subject_experiment_dates) to a DataFrame with a column named 'ExperimentStartDate'
        subject_experiment_dates_df = pd.DataFrame({'ExperimentDates': subject_experiment_dates})

        # Save the dates of sleep of the current subject in a list
        subject_sleep_dates = subject_sleep_data['DateOfSleep (Fitbit Calculation)'].tolist()

        # Convert the subject's sleep dates to datetime objects

        subject_sleep_dates = [pd.to_datetime(date, format="%d/%m/%Y") for date in subject_sleep_dates]

        # Initialize the dataframe that will hold the sleep levels data
        sleep_levels_df = pd.DataFrame()


        # Iterate over the json sleep files from the merged list and create a dataframe of the levels summary values of deep, light, rem, and wake
        # for each sleep date
        for date in sleep_json_merged:
            # Convert the date of sleep to datetime object
            if (pd.to_datetime(date['dateOfSleep'], format='%Y-%m-%d') not in subject_sleep_dates or date['type'] == 'classic' or (date.get('mainSleep') == False or date.get('isMainSleep') == False)):
                continue
            date['levels']['summary'] = pd.json_normalize(date['levels']['summary'])
            
            # add row to the dataframe with the date of sleep and the levels summary values
            sleep_levels_df = pd.concat([sleep_levels_df, pd.DataFrame([pd.Series({
                                                                        'DateOfSleep': date['dateOfSleep'],
                                                                        'sleep_start_time':( datetime.datetime.strptime(date['startTime'], '%Y-%m-%dT%H:%M:%S.%f')).strftime('%H:%M:%S'),
                                                                        'deep_minutes': int(date['levels']['summary']['deep.minutes']),
                                                                        'wake_minutes': int(date['levels']['summary']['wake.minutes']),
                                                                        'light_minutes': int(date['levels']['summary']['light.minutes']),
                                                                        'rem_minutes': int(date['levels']['summary']['rem.minutes'])
                                                                        })])], ignore_index=True)
            

        
        # Filter out the duplications by dateOfSleep and keep only the last one
        sleep_levels_df = sleep_levels_df.drop_duplicates(subset=['DateOfSleep'], keep='last')
        # Find the file "Computed Temperature - YYYY-MM-DD.csv" in the sleep folder
        computed_temperature_file_name_pattern = re.compile(r"^Computed Temperature - \d{4}-\d{2}-\d{2}\.csv")
        computed_temperature_files = [file_name for file_name in os.listdir(sleep_json_directory)
                                      if computed_temperature_file_name_pattern.search(file_name)]
        if not computed_temperature_files:
            subjects_with_missing_computed_temperature_files_files.append(subject)
            computed_temperature_df = pd.DataFrame()
            computed_temperature_df['DateOfSleepEvening'] = pd.to_datetime(subject_experiment_dates_df['ExperimentDates'])
        else:
            # Create a list of dataframes containing the computed temperature files for the current subject.
            computed_temperature_dfs = [pd.read_csv(sleep_json_directory.joinpath(file),
                                                    usecols=['sleep_start', 'sleep_end', 'nightly_temperature'],
                                                    on_bad_lines='skip',
                                                    parse_dates=['sleep_start', 'sleep_end'])
                                        for file in computed_temperature_files if os.stat(sleep_json_directory.joinpath(file)).st_size != 0]
            # Concatenate all computed temperature files into one dataframe and reset index
            computed_temperature_df = pd.concat(computed_temperature_dfs).reset_index(drop=True)
            # convert to datetime and useing format='mixed' because its the only thing that works.
            computed_temperature_df['sleep_start'] = pd.to_datetime(computed_temperature_df['sleep_start'], format='mixed')
            computed_temperature_df = computed_temperature_df.sort_values(by='sleep_start')

            # Mask of rows where subject went to sleep after 00:00 and before 08:00 so the sleep date should be the date before
            mask_of_sleep_after_midnight = (computed_temperature_df['sleep_start'].dt.hour >= 0) & (computed_temperature_df['sleep_start'].dt.hour < 8)
            # Using DateOffset function to change date to date previous date.
            computed_temperature_df.loc[mask_of_sleep_after_midnight, 'DateOfSleepEvening'] = computed_temperature_df.loc[mask_of_sleep_after_midnight, 'sleep_start'] - pd.DateOffset(days=1)
            # set the value of sleep_start where DateOfSleepEvening NaT
            computed_temperature_df.loc[computed_temperature_df['DateOfSleepEvening'].isna(), 'DateOfSleepEvening'] = computed_temperature_df['sleep_start']
            computed_temperature_df['DateOfSleepEvening'] = pd.to_datetime(computed_temperature_df['DateOfSleepEvening'].dt.date)

            computed_temperature_df['sleep_end'] = pd.to_datetime(computed_temperature_df['sleep_end'], format='mixed')
            computed_temperature_df['sleep_start'] = pd.to_datetime(computed_temperature_df['sleep_start'], format='mixed')

            # Rename column to make valid_sleep function to run:
            computed_temperature_df = computed_temperature_df.rename(columns={'sleep_start': 'SleepStartTime', 'sleep_end': 'SleepEndTime'})
            # Add column with the sleep time in minutes
            computed_temperature_df['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'] = computed_temperature_df.apply(lambda row: (row['SleepEndTime'] - row['SleepStartTime']).total_seconds() / 60, axis=1)
            # Add column with the valid sleep value
            computed_temperature_df['ValidSleep'] = valid_sleep(computed_temperature_df)

            # Filter out the rows that are not valid sleep
            computed_temperature_df = computed_temperature_df[computed_temperature_df['ValidSleep'] == True]

            # Sort by DateOfSleepEvening
            computed_temperature_df = computed_temperature_df.sort_values(by='DateOfSleepEvening')
            # mereg subject_sleep_data with computed_temperature_df on DateOfSleepEvening column and keep only the relevant columns
            # computed_temperature_df = computed_temperature_df.merge(subject_sleep_data, on='DateOfSleepEvening', how='outer')
            # computed_temperature_df = computed_temperature_df[['DayOfExperiment', 'DateOfSleepEvening', 'nightly_temperature']]
            # Rename 'nightly_temperature' to 'SleepTemperature'
            computed_temperature_df = computed_temperature_df.rename(columns={'nightly_temperature': 'SleepTemperature'})
        # merge with experiments dates to not loosing data of respiratory.
        computed_temperature_df = computed_temperature_df.merge(subject_experiment_dates_df,
                                                                how='right',
                                                                left_on='DateOfSleepEvening',
                                                                right_on='ExperimentDates')

        computed_temperature_df['DayOfExperiment'] = computed_temperature_df['ExperimentDates'].rank(method='dense')

        # Find all "Daily Respiratory Rate Summary - YYYY-MM-DD.csv" files in the sleep folder
        respiratory_file_name_pattern = re.compile(r"^Daily Respiratory Rate Summary - \d{4}-\d{2}-\d{2}\.csv")
        respiratory_files = [file_name for file_name in os.listdir(sleep_json_directory) if
                             respiratory_file_name_pattern.search(file_name)]
        if not respiratory_files:
            subjects_with_missing_respiratory_files.append(subject)
            continue
        try:
            # Create a list of dataframes containing the respiratory files for the current subject.
            respiratory_dfs = [pd.read_csv(sleep_json_directory.joinpath(file), on_bad_lines='skip',
                                        usecols=['timestamp', 'daily_respiratory_rate'],
                                        parse_dates=['timestamp'])
                            for file in respiratory_files]
            # Concatenate all respiratory files into one dataframe
            full_respiratory_df = pd.concat(respiratory_dfs)
            # daily_respiratory_rate: Breathing rate average estimated from deep sleep when possible,
            # and from light sleep when deep sleep data is not available.

            # Using DateOffset function to change date to date previous date.
            full_respiratory_df['timestamp'] -= pd.DateOffset(days=1)
            full_respiratory_df['timestamp'] = pd.to_datetime(full_respiratory_df['timestamp'].dt.date)

            full_respiratory_df = full_respiratory_df.rename(columns={'timestamp': 'DateOfSleepEvening',
                                                                    'daily_respiratory_rate': 'SleepBreathsInMinute Mean (Fitbit Calculation)'})
            full_respiratory_df = full_respiratory_df.sort_values(by='DateOfSleepEvening')


            # merge full_respiratory_df with subject_experiment_dates_df on DateOfSleepEvening column and keep only the relevant columns
            # "right" because if the there is missing dates in the full_respiratory_df file so it will skip this date but we want to present the date without data.
            full_respiratory_df = full_respiratory_df.merge(subject_experiment_dates_df, how='right',
                                                    left_on='DateOfSleepEvening', right_on='ExperimentDates')
        except:
            logger.info(f'Error with subject {subject} in respiratory file')
            # Resume with the process withot the respiratory file and not continue to the next subject
            full_respiratory_df = subject_experiment_dates_df
            # create a new column with the date of the sleep evening
            full_respiratory_df['DateOfSleepEvening'] = full_respiratory_df['ExperimentDates']
            full_respiratory_df['SleepBreathsInMinute Mean (Fitbit Calculation)'] = np.nan
            
        full_respiratory_df['DayOfExperiment'] = full_respiratory_df['ExperimentDates'].rank(method='dense')

        # full_respiratory_df = full_respiratory_df.merge(subject_sleep_data, on='DateOfSleepEvening', how='right')
        # full_respiratory_df = full_respiratory_df[['DayOfExperiment', 'SleepBreathsInMinute Mean (Fitbit Calculation)', 'DateOfSleepEvening']]

        # Drop ExperimentDates from respiratory df
        full_respiratory_df = full_respiratory_df.drop(columns=['ExperimentDates', 'DateOfSleepEvening'])
        # "inner": use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
        computed_temperature_and_respiratory_df = pd.merge(computed_temperature_df,
                                                           full_respiratory_df,
                                                           on='DayOfExperiment',
                                                           how='inner').sort_values(by='DayOfExperiment').reset_index(drop=True)
        # Set nan in 'SleepBreathInMinute' where 'SleepTemperature' is nan
        if 'SleepTemperature' in computed_temperature_and_respiratory_df.columns:
            computed_temperature_and_respiratory_df.loc[computed_temperature_and_respiratory_df['SleepTemperature'].isna(), 'SleepBreathsInMinute Mean (Fitbit Calculation)'] = np.nan        # Find all "Heart Rate Variability Details - YYYY-MM-DD.csv" files in the sleep folder
        else:
            computed_temperature_and_respiratory_df['SleepTemperature'] = np.nan
            computed_temperature_and_respiratory_df.loc[computed_temperature_and_respiratory_df['SleepTemperature'].isna(), 'SleepBreathsInMinute Mean (Fitbit Calculation)'] = np.nan

        HRV_details_file_name_pattern = re.compile(r"^Heart Rate Variability Details - \d{4}-\d{2}-\d{2}.csv")
        HRV_details_files = [file_name for file_name in os.listdir(sleep_json_directory)
                             if HRV_details_file_name_pattern.search(file_name)]
        if not HRV_details_files:
            subjects_with_missing_HRV_details_files.append(subject)
            continue

        # Create a list of dataframes containing the HRV details files for the current subject. if no columns are found, skip to the next file.
        HRV_details_dfs = []
        for file in HRV_details_files:
            try:
                HRV_details_dfs.append(pd.read_csv(sleep_json_directory.joinpath(file), on_bad_lines='skip',
                                                  usecols=['timestamp', 'rmssd', 'coverage', 'low_frequency', 'high_frequency'],
                                                  parse_dates=['timestamp']))
            except:
                continue
            
        full_HRV_details_df = pd.DataFrame()
        
        if HRV_details_dfs == []:
            subjects_with_missing_HRV_details_files.append(subject)
            # create empty dataframe to not break the code 
            HRV_details_dfs.append(pd.DataFrame(columns=['timestamp', 'rmssd', 'coverage', 'low_frequency', 'high_frequency']))
            full_HRV_details_df = pd.concat(HRV_details_dfs)
            # put 00 in the columns to not break the code
            full_HRV_details_df['timestamp'] = pd.to_datetime('00:00:00')
            full_HRV_details_df['timestamp'] = pd.to_datetime(full_HRV_details_df['timestamp'])
            full_HRV_details_df = full_HRV_details_df.sort_values(by='timestamp')
        else:
            # Concatenate all HRV files into one dataframe
            full_HRV_details_df = pd.concat(HRV_details_dfs)
            # Convert timestamp column to datetime format
            full_HRV_details_df['timestamp'] = pd.to_datetime(full_HRV_details_df['timestamp'])
            # Sort by timestamp
            full_HRV_details_df = full_HRV_details_df.sort_values(by='timestamp')

        # Find all "Device Temperature - YYYY-MM-DD.csv" files in the sleep folder
        device_temperature_file_name_pattern = re.compile(r"^Device Temperature - \d{4}-\d{2}-\d{2}\.csv")
        # Find all "Device Temperature - YYYY-MM-DD API.csv" files in the sleep folder (optional)
        device_temperature_file_name_pattern_api = re.compile(r"^Device Temperature - \d{4}-\d{2}-\d{2} API.csv")

        device_temperature_files = [file_name for file_name in os.listdir(sleep_json_directory)
                                    if device_temperature_file_name_pattern.search(file_name)]
        device_temperature_files_api = [file_name for file_name in os.listdir(sleep_json_directory)
                                        if device_temperature_file_name_pattern_api.search(file_name)]
        
        # Validation: Ignore files that are the first 3 nights
        for file in device_temperature_files:
            # get file date with regex
            file_date = re.search(r'\d{4}-\d{2}-\d{2}', file).group()
            # convert to Timestamp
            file_date = datetime.datetime.strptime(file_date, '%Y-%m-%d')
            # if file date is before the 4th night, remove it from the list
            subject_dates_of_experiment = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
            start_night = pd.to_datetime(subject_dates_of_experiment['ExperimentStartDate'].values[0])
            fourth_night = start_night + datetime.timedelta(days=3)
            # Example: start_day = 2020-11-01,  fourth_night = 2020-11-04, file_date = 2020-11-03/02/01 -> remove file
            if fourth_night > file_date:
                device_temperature_files.remove(file)

        # Only for the api files
        for file in device_temperature_files_api:
            # get file date with regex
            file_date = re.search(r'\d{4}-\d{2}-\d{2}', file).group()
            # convert to Timestamp
            file_date = datetime.datetime.strptime(file_date, '%Y-%m-%d')
            # if file date is before the 4th night, remove it from the list
            subject_dates_of_experiment = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
            start_night = pd.to_datetime(subject_dates_of_experiment['ExperimentStartDate'].values[0])
            fourth_night = start_night + datetime.timedelta(days=3)
            # Example: start_day = 2020-11-01,  fourth_night = 2020-11-04, file_date = 2020-11-03/02/01 -> remove file
            if fourth_night > file_date:
                device_temperature_files_api.remove(file)


        if not device_temperature_files:
            if not device_temperature_files_api:
                subjects_with_missing_device_temperature_files.append(subject)
                continue
            
        device_temperature_dfs = []
        full_device_temperature_df = pd.DataFrame()

        if device_temperature_files != []:
            # Create a list of dataframes containing the temperature files for the current subject.
            device_temperature_dfs = [pd.read_csv(sleep_json_directory.joinpath(file), on_bad_lines='skip',
                                                usecols=['recorded_time', 'temperature'],
                                                parse_dates=['recorded_time'])
                                    for file in device_temperature_files]
            
            # Concatenate all temperature files into one dataframe
            full_device_temperature_df = pd.concat(device_temperature_dfs)
            # Rename 'recorded_time' to 'timestamp'
            full_device_temperature_df = full_device_temperature_df.rename(columns={'recorded_time': 'timestamp'})
            # Sort timestamp column
            full_device_temperature_df = full_device_temperature_df.sort_values(by='timestamp')

        device_temperature_dfs_api = []

        full_device_temperature_df_api = pd.DataFrame()

        if device_temperature_files_api != []:
                
            # Create a list of dataframes containing the temperature api files for the current subject.
            device_temperature_dfs_api = [pd.read_csv(sleep_json_directory.joinpath(file), on_bad_lines='skip',
                                                usecols=['recorded_time', 'temperature'],
                                                parse_dates=['recorded_time'])
                                    for file in device_temperature_files_api]
            full_device_temperature_df_api = pd.concat(device_temperature_dfs_api)
            full_device_temperature_df_api = full_device_temperature_df_api.rename(columns={'recorded_time': 'timestamp'})
            full_device_temperature_df_api = full_device_temperature_df_api.sort_values(by='timestamp')


        # # Add read 'Heart Rate and Steps and Sleep Aggregated' to add heart rate mean per sleep later
        # hr_steps_sleep_path = get_latest_file_by_term("Heart Rate and Steps and Sleep Aggregated", subject)
        # if not hr_steps_sleep_path.exists():
        #     logger.info(
        #         f'Can\'t calculate HRV_temperature_respiratory_at_sleep for subject {subject} because Heart Rate and Steps and Sleep Aggregated.csv is missing from subject\'s folder')
        #     continue
        # # Read the latest heart rate file
        # hr_steps_sleep_df = pd.read_csv(hr_steps_sleep_path, usecols=['DateAndMinute', 'BpmMean'], parse_dates=['DateAndMinute'])

        HRV_details_output_df = pd.DataFrame()
        device_temperature_output_df = pd.DataFrame()
        device_temperature_output_df_api = pd.DataFrame()
        # Loop through each row of sleep data for the current subject #TODO make efficiency
        # locate the relevant HRV data for the current sleep period.
        # data_list = subject_sleep_data.apply(lambda row: full_HRV_details_df[(full_HRV_details_df['timestamp'] >= row['startTime']) & (full_HRV_details_df['timestamp'] <= row['endTime'])], axis=1).tolist()
        for i, sleep_row in subject_sleep_data.iterrows():
            # Extract the relevant HRV data for the current sleep period. Only rows between the BedStartTime and BedEndTime.
            relevant_HRV_data = full_HRV_details_df.loc[full_HRV_details_df['timestamp'].between(sleep_row['SleepStartTime'], sleep_row['SleepEndTime'])]
            if len(relevant_HRV_data) > 0:
                # Calculate and add HRV statistics to output dataframe
                HRV_details_output_df = pd.concat([HRV_details_output_df,
                                           pd.DataFrame({
                                               "DayOfExperiment": [sleep_row['DayOfExperiment']],
                                               "DateOfSleep (Fitbit Calculation)": [sleep_row['DateOfSleep (Fitbit Calculation)']],
                                               "HRV RMSSD Mean": [np.mean(relevant_HRV_data['rmssd'])],
                                               "HRV Coverage Mean": [np.mean(relevant_HRV_data['coverage'])],
                                               "HRV Low Frequency Mean": [np.mean(relevant_HRV_data['low_frequency'])],
                                               "HRV High Frequency Mean": [np.mean(relevant_HRV_data['high_frequency'])]
                                           })], ignore_index=True)
            # Extract the relevant temperature data for the current sleep period. Only rows between the SleepStartTime & SleepEndTime.
            # For example:
            # sleep_row['SleepStartTime'] = Timestamp('2022-08-08 04:31:00')
            # sleep_row['SleepEndTime'] = Timestamp('2022-08-08 09:46:30')
            # full_device_temperature_df['timestamp'] =
            # 271   2022-08-08 04:31:00
            # 272   2022-08-08 04:32:00
            # 273   2022-08-08 04:33:00
            # 274   2022-08-08 04:34:00
            # 275   2022-08-08 04:35:00
            #               ...
            # 582   2022-08-08 09:42:00
            # 583   2022-08-08 09:43:00
            # 584   2022-08-08 09:44:00
            # 585   2022-08-08 09:45:00
            # 586   2022-08-08 09:46:00
            if not full_device_temperature_df.empty:
                relevant_tempe_data = full_device_temperature_df.loc[
                    full_device_temperature_df['timestamp'].between(sleep_row['SleepStartTime'], sleep_row['SleepEndTime'])]
            if not full_device_temperature_df_api.empty:
                relevant_tempe_data = full_device_temperature_df_api
            if len(relevant_tempe_data) > 0:
                device_temperature_output_df = pd.concat([device_temperature_output_df,
                                                   pd.DataFrame({
                                                       # "DateOfSleep (Fitbit Calculation)": [sleep_row['DateOfSleep (Fitbit Calculation)']],
                                                       "DayOfExperiment": [sleep_row['DayOfExperiment']],
                                                       "Skin Temperature Mean": [np.mean(relevant_tempe_data['temperature'])],
                                                       "Skin Temperature Min": [np.min(relevant_tempe_data['temperature'])],
                                                       "Skin Temperature Max": [np.max(relevant_tempe_data['temperature'])]
                                                   })], ignore_index=True)

        # Merge HRV and temperature output dataframes and write to output CSV file #TODO: not a relevant validation?
        # Merge on DateOfSleepEvening column because its unique "outer": use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
        if not HRV_details_output_df.empty and not device_temperature_output_df.empty:
            HRV_details_and_device_temperature_df = HRV_details_output_df.merge(device_temperature_output_df,
                                                                                on='DayOfExperiment',
                                                                                how='outer')
        elif not HRV_details_output_df.empty:
            HRV_details_and_device_temperature_df = HRV_details_output_df
            HRV_details_and_device_temperature_df['Skin Temperature Mean'] = np.nan
            HRV_details_and_device_temperature_df['Skin Temperature Min'] = np.nan
            HRV_details_and_device_temperature_df['Skin Temperature Max'] = np.nan
            
        elif not device_temperature_output_df.empty:
            HRV_details_and_device_temperature_df = device_temperature_output_df
            HRV_details_and_device_temperature_df['HRV RMSSD Mean'] = np.nan
            HRV_details_and_device_temperature_df['HRV Coverage Mean'] = np.nan
            HRV_details_and_device_temperature_df['HRV Low Frequency Mean'] = np.nan
            HRV_details_and_device_temperature_df['HRV High Frequency Mean'] = np.nan
            
        else:
            print(f'No HRV or temperature data for subject {subject}')
            continue
        # Convert the DateOfSleep (Fitbit Calculation) column to datetime
        # HRV_details_and_device_temperature_df['DateOfSleepEvening'] = pd.to_datetime(HRV_details_and_device_temperature_df['DateOfSleepEvening'])
        
        sleep_levels_df['DateOfSleep'] = pd.to_datetime(sleep_levels_df['DateOfSleep']).dt.date
        HRV_details_and_device_temperature_df['DateOfSleep (Fitbit Calculation)'] = pd.to_datetime(
            HRV_details_and_device_temperature_df['DateOfSleep (Fitbit Calculation)'],
            format='%d/%m/%Y'
        ).dt.date
        # Merge the DataFrames on the converted date columns
        HRV_temperature_respiratory_df = pd.merge(HRV_details_and_device_temperature_df,
                            sleep_levels_df,
                            left_on='DateOfSleep (Fitbit Calculation)',
                            right_on='DateOfSleep',
                            how='left')
        
        # Merge the HRV and temperature data with the sleep data
        HRV_temperature_respiratory_df = pd.merge(computed_temperature_and_respiratory_df,
                                                  HRV_temperature_respiratory_df,
                                                  on='DayOfExperiment',
                                                  how='outer')
        


        # Drop the 'DateOfSleep' column from the merged DataFrame
        HRV_temperature_respiratory_df.drop(columns=['DateOfSleep'], inplace=True)

        # fill the missing dates in DateOfSleepEvening column
        HRV_temperature_respiratory_df['DateOfSleepEvening'] = HRV_temperature_respiratory_df['ExperimentDates']
        
        # Add Id column
        HRV_temperature_respiratory_df['Id'] = subject

        # Sort by Id and DayOfExperiment
        HRV_temperature_respiratory_df = HRV_temperature_respiratory_df.sort_values(by=['Id', 'DayOfExperiment'])

        # Select specific columns to reorder the columns and save the resulting DataFrame to a CSV file
        subject_output_path = OUTPUT_PATH.joinpath(subject)
        subject_output_path_history = ut.output_record(OUTPUT_PATH, subject, user_name, now )
        
        if not subject_output_path_history.exists():
            subject_output_path_history.mkdir(parents=True)
        non_relevant_dates = []
        if subject_dates['NotInIsrael'].values[0]:
            non_relevant_dates = pd.date_range(subject_dates['NotInIsraelStartDate'].values[0],
                                                  subject_dates['NotInIsraelEndDate'].values[0])
            
                                               
        HRV_temperature_respiratory_df = HRV_temperature_respiratory_df[~HRV_temperature_respiratory_df['ExperimentDates'].isin(non_relevant_dates)]
        
        if subject_dates['NotInIsrael_1'].values[0]:
            non_relevant_dates = pd.date_range(subject_dates['NotInIsraelStartDate_1'].values[0],
                                                  subject_dates['NotInIsraelEndDate_1'].values[0])
            
                                               
        HRV_temperature_respiratory_df = HRV_temperature_respiratory_df[~HRV_temperature_respiratory_df['ExperimentDates'].isin(non_relevant_dates)]

        HRV_temperature_respiratory_df[['Id', 'DayOfExperiment', 'ExperimentDates', 'DateOfSleepEvening',
                                        'HRV RMSSD Mean', 'HRV Coverage Mean',
                                        'HRV Low Frequency Mean', 'HRV High Frequency Mean',
                                        'SleepTemperature', 'Skin Temperature Mean', 'Skin Temperature Min', 'Skin Temperature Max',
                                        'SleepBreathsInMinute Mean (Fitbit Calculation)','sleep_start_time','deep_minutes',
                                        'wake_minutes','light_minutes', 'rem_minutes']].to_csv(
                                        subject_output_path_history.joinpath(f'{subject} HRV Temperature Respiratory At Sleep.csv'), index=False)

        ut.check_for_duplications(subject_output_path, subject_output_path_history)
        
    # If there are any subjects with missing computed temperature files, logger.info the subject IDs
    if subjects_with_missing_computed_temperature_files_files:
        print(f'Subjects with missing computed temperature files in FITBIT/Sleep folder of each subject:')
        print('\n'.join(subjects_with_missing_computed_temperature_files_files))
    # If there are any subjects without respiratory files, logger.info the subject IDs
    if subjects_with_missing_respiratory_files:
        print(f'Subjects with missing respiratory files in FITBIT/Sleep folder of each subject:')
        print('\n'.join(subjects_with_missing_respiratory_files))
    # If there are any subjects without json files, logger.info the subject IDs
    if subjects_with_missing_json_files:
        print(f'Subjects with missing json files in FITBIT/Sleep folder of each subject:')
        print('\n'.join(subjects_with_missing_json_files))
    # If there are any subjects without HRV_details files, logger.info the subject IDs
    if subjects_with_missing_HRV_details_files:
        print(f'Subjects with missing HRV details files in FITBIT/Sleep folder of each subject:')
        print('\n'.join(subjects_with_missing_HRV_details_files))
    # If there are any subjects without device temperature files, logger.info the subject IDs
    if subjects_with_missing_device_temperature_files:
        print(f'Subjects with missing device temperature files in FITBIT/Sleep folder of each subject:')
        print('\n'.join(subjects_with_missing_device_temperature_files))

    ########################################### generate aggregated file ###########################################
    all_subjects_summary_subjects = []
    all_subjects_details_df = pd.DataFrame()
    print('\n Summary of HRV Temperature Respiratory at sleep')
    
    tqdm_subjects = tqdm(os.listdir(DATA_PATH))
    for subject in tqdm_subjects:
        if not re.search(r'\d{3}$', subject):
            continue
        if run_on_specific_subjects and subject not in subjects_to_run_on:
            continue
        # Update the tqdm description with the current subject
        tqdm_subjects.set_description(f'Subject {subject}')

        # if the folder not look like 'sub_203' so skip it
        # if not re.fullmatch(SUBJECT_FOLDER_FORMAT, subject):
        #     continue #@TODO: GET BACK IN

        # Get latest file of HRV Temperature Respiratory At Night.csv
        HRV_tempe_resp_file = ut.new_get_latest_file_by_term('HRV Temperature Respiratory', subject=subject, root=OUTPUT_PATH)

        # If there is no sub_xxx HRV Temperature Respiratory At Sleep .csv file in the subject folder, continue to the next subject
        if not HRV_tempe_resp_file.exists():
            continue
        # Read the file to a DataFrame
        HRV_tempe_resp_df = pd.read_csv(HRV_tempe_resp_file)
        subject_summary_dict = {}
        subject_summary_dict['Id'] = subject
        subject_summary_dict['HRV_RMSSD_mean'] = HRV_tempe_resp_df['HRV RMSSD Mean'].mean()
        subject_summary_dict['HRV_RMSSD_std'] = HRV_tempe_resp_df['HRV RMSSD Mean'].std()
        # subject_summary_dict['HRV_Entropy_mean'] = HRV_tempe_resp_df['HRV Entropy  (Fitbit Calculation)'].mean()
        # subject_summary_dict['HRV_Entropy_std'] = HRV_tempe_resp_df['HRV Entropy  (Fitbit Calculation)'].std()
        subject_summary_dict['HRV_Coverage_mean'] = HRV_tempe_resp_df['HRV Coverage Mean'].mean()
        subject_summary_dict['HRV_Coverage_std'] = HRV_tempe_resp_df['HRV Coverage Mean'].std()
        subject_summary_dict['HRV_Low_Frequency_mean'] = HRV_tempe_resp_df['HRV Low Frequency Mean'].mean()
        subject_summary_dict['HRV_Low_Frequency_std'] = HRV_tempe_resp_df['HRV Low Frequency Mean'].std()
        subject_summary_dict['HRV_High_Frequency_mean'] = HRV_tempe_resp_df['HRV High Frequency Mean'].mean()
        subject_summary_dict['HRV_High_Frequency_std'] = HRV_tempe_resp_df['HRV High Frequency Mean'].std()
        subject_summary_dict['Skin_Temperature_mean'] = HRV_tempe_resp_df['Skin Temperature Mean'].mean()
        subject_summary_dict['Skin_Temperature_std'] = HRV_tempe_resp_df['Skin Temperature Mean'].std()
        subject_summary_dict['Skin_Temperature_min'] = HRV_tempe_resp_df['Skin Temperature Min'].mean()
        subject_summary_dict['Skin_Temperature_max'] = HRV_tempe_resp_df['Skin Temperature Max'].mean()
        # subject_summary_dict['NREM_Heart_Rate_mean'] = HRV_tempe_resp_df['NREM Heart Rate  (Fitbit Calculation)'].mean()
        # subject_summary_dict['NREM_Heart_Rate_std'] = HRV_tempe_resp_df['NREM Heart Rate  (Fitbit Calculation)'].std()
        subject_summary_dict['Sleep_Breaths_In_Minute_mean'] = HRV_tempe_resp_df['SleepBreathsInMinute Mean (Fitbit Calculation)'].mean()
        subject_summary_dict['Sleep_Breaths_In_Minute_std'] = HRV_tempe_resp_df['SleepBreathsInMinute Mean (Fitbit Calculation)'].std()
    
        # Append the dictionary to the list
        all_subjects_summary_subjects.append(subject_summary_dict)
        # Concatenate the DataFrame to the all_subjects_details_df
        all_subjects_details_df = pd.concat([all_subjects_details_df, HRV_tempe_resp_df])
    # Create a DataFrame from the list of dictionaries
    all_subjects_summary_subjects_df = pd.DataFrame(all_subjects_summary_subjects)
    all_subjects_summary_subjects_df = concate_to_old('Summary Of HRV Temperature Respiratory At Sleep', AGGREGATED_OUTPUT_PATH, all_subjects_summary_subjects_df)

    all_subjects_details_df = concate_to_old('HRV Temperature Respiratory At Sleep All Subjects', AGGREGATED_OUTPUT_PATH, all_subjects_details_df)

    if all_subjects_summary_subjects_df.empty:
        print('No HRV Temperature Respiratory At Sleep.csv files were found in the subjects folders')
        return
    # Sort by Id column
    all_subjects_summary_subjects_df = all_subjects_summary_subjects_df.sort_values(by=['Id'])
    all_subjects_details_df = all_subjects_details_df.sort_values(by=['Id'])
    
    # Save output CSV file to project aggregated output folder
    all_subjects_summary_subjects_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Summary Of HRV Temperature Respiratory At Sleep.csv'), index=False) 

    # Save output CSV file to project aggregated output folder
    all_subjects_details_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'HRV Temperature Respiratory At Sleep All Subjects.csv'), index=False) 
    
    ut.check_for_duplications(AGGREGATED_OUTPUT_PATH, AGGREGATED_OUTPUT_PATH_HISTORY)

    





  


def is_dst_change(date: datetime.datetime) -> bool:
    jerusalem = pytz.timezone("Asia/Jerusalem")
    
    midnight = jerusalem.localize(datetime.datetime(date.year, date.month, date.day))
    next_midnight = jerusalem.localize(datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(days=1))
    
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

def valid_sleep(subject_sleep_df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function gets a DataFrame with sleep data of a subject and returns a DataFrame column with the valid sleep data.
    Valid sleep data is defined as:
                1. Between 20:00 to 08:00
                2. More than 3 hours of sleep
                3. If there are multiple sleep at night, the function will set True for the longest sleep and False for the rest.
    # the column names that the dataframe must have are:
    SleepStartTime
    DateOfSleepEvening
    If there are multiple sleep at night, the function will set True for the longest sleep and False for the rest.
    It does it by group by date and select the longest sleep at night.
    :param subject_sleep_df: DataFrame with sleep data of a subject
    :return: DataFrame column with the valid sleep column (boolean column)

    '''
    # Create a copy of the original DataFrame within the function and modify the copy.
    subject_sleep_df_copy = subject_sleep_df.copy()
    # Create a boolean mask to identify rows with valid sleep times based on the hour between 20:00 in the evening to 08:00 at the morning.
    mask_valid_rows_by_hour = (subject_sleep_df_copy['SleepStartTime'].dt.hour < 8) | (subject_sleep_df_copy['SleepStartTime'].dt.hour >= 20)
    # Add a new column to the DataFrame indicating whether each row has valid sleep times
    subject_sleep_df_copy['ValidSleep'] = mask_valid_rows_by_hour
    valid_subject_sleep_df_copy = subject_sleep_df_copy.loc[subject_sleep_df_copy['ValidSleep'] == True]
    # group by DateOfSleepEvening and iterate over each group
    for date, group in valid_subject_sleep_df_copy.groupby('DateOfSleepEvening'):
        max_index = group['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'].idxmax()
        # Set False for all rows in the group
        for idx in group.index:
            subject_sleep_df_copy.at[idx, 'ValidSleep'] = False
        # Set True for the row with the longest sleep duration
        subject_sleep_df_copy.at[max_index, 'ValidSleep'] = True
    # Validate that each valid sleep has more than 3 hours of sleep, if not set False in ValidSleep column
    subject_sleep_df_copy.loc[
        (subject_sleep_df_copy['ValidSleep'] == True) & (subject_sleep_df_copy['SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)'] <= 180),'ValidSleep'] = False

    return subject_sleep_df_copy['ValidSleep']


if __name__ == '__main__':
    
    try:
        param = sys.argv[1]
        now = sys.argv[2]
        user_name = sys.argv[3]

    except IndexError:
        param = 'FIBRO_TESTS'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        user_name = 'Unknown'

    main(param, now, user_name)