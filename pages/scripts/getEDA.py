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
sys.path.append(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages')

import UTILS.utils as ut
# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings
warnings.filterwarnings('ignore')





def main(project, now, username):
    FIRST = 0
    LAST = -1

    exeHistory_path = Path(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\ExecutionHis\exeHistory.parquet')

    exeHistory = pl.read_parquet(exeHistory_path)

    paths_json = json.load(open(r"C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\Pconfigs\paths.json", "r"))
    project_path = Path(paths_json[project])



    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

    AGGREGATED_OUTPUT_PATH_HISTORY = ut.output_record(OUTPUT_PATH, 'Aggregated Output',username, now)

    if not AGGREGATED_OUTPUT_PATH_HISTORY.exists():
        os.makedirs(AGGREGATED_OUTPUT_PATH_HISTORY)
        
    PROJECT_CONFIG = json.load(open(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\Pconfigs\paths data.json', 'r'))
        
    SUBJECTS_DATES = METADATA_PATH.joinpath('Subjects Dates.csv')


    subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                try_parse_dates=True)



    subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')

    selected_subjects_path = Path(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_EDA.parquet')

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
    print('\n Process EDA Data')
   
    tqdm_subjects = tqdm(os.listdir(DATA_PATH))
        
    for subject in tqdm_subjects:
        if not re.search(r'\d{3}$', subject):
            continue
        if run_on_specific_subjects and subject not in subjects_to_run_on:
            continue
        tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)
        # if the folder not look like 'sub_203' so skip it
        # if not re.fullmatch(SUBJECT_FOLDER_FORMAT, subject):
        #     continue #@TODO: GET BACK IN
        stress_avgs = {"session_id": [], "total_avg": [], "valid_avg": []}
        subject_stress_folder = DATA_PATH.joinpath(subject).joinpath(r'FITBIT\Stress')

        eda_mindfulness_file_name_pattern = re.compile(r'^Mindfulness Eda Data Sessions.csv')
        eda_mindfulness_file_name_pattern_alternative = re.compile(r'^mindfulness_eda_data_sessions.csv')
        eda_mindfulness_csv_files = [file_name for file_name in os.listdir(subject_stress_folder)
                            if eda_mindfulness_file_name_pattern.search(file_name)]
        eda_mindfulness_csv_files_alternative = [file_name for file_name in os.listdir(subject_stress_folder)
                            if eda_mindfulness_file_name_pattern_alternative.search(file_name)]
        eda_mindfulness_csv_files.extend(eda_mindfulness_csv_files_alternative)
        if len(eda_mindfulness_csv_files) > 0:
            eda_mindfulness_file = eda_mindfulness_csv_files[0] # The validation is in the function: create_valid_database
        else:
            continue
        eda_mindfulness_df = pd.read_csv(subject_stress_folder.joinpath(eda_mindfulness_file),
                                         usecols=['session_id', 'scl_avg', 'valid_data'],
                                         dtype={'session_id': 'category', 'scl_avg': 'float', 'valid_data': 'bool'})

        sessions_file_name_pattern = re.compile(r'^Mindfulness Sessions.csv')
        sessions_file_name_pattern_alternative = re.compile(r'^mindfulness_sessions.csv')
        sessions_csv_files = [file_name for file_name in os.listdir(subject_stress_folder)
                                      if sessions_file_name_pattern.search(file_name)]
        sessions_csv_files_alternative = [file_name for file_name in os.listdir(subject_stress_folder)
                                        if sessions_file_name_pattern_alternative.search(file_name)]
        sessions_csv_files.extend(sessions_csv_files_alternative)
        if len(sessions_csv_files) > 0:
            sessions_file = sessions_csv_files[0] # The validation is in the function: create_valid_database
        else:
            continue
        sessions_df = pd.read_csv(subject_stress_folder.joinpath(sessions_file),
                                  usecols=['session_id', 'average_heart_rate', 'start_heart_rate', 'end_heart_rate',
                                           'start_date_time', 'end_date_time', 'duration'],
                                  dtype={'session_id': 'category', 'average_heart_rate': 'float',
                                         'start_heart_rate': 'float', 'end_heart_rate': 'float'},
                                  parse_dates=['start_date_time', 'end_date_time'],
                                  date_parser=lambda col: pd.to_datetime(col, utc=False, errors='coerce'))
        
        if os.stat(subject_stress_folder.joinpath(sessions_file)).st_size == 155:
            print(f'{subject} is missing {sessions_file}')
            continue
        # Get start and end dates of experiment for current subject
        subject_dates = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
        # Remove timezone ("+00:00") from the dates
        # Convert datetime values to Israel local time by:
        #       Already localized to UTC. so the conversion: "2022-11-07 23:01:00" --> "2022-11-07 23:01:00+00:00" is not needed.
        #       First, converting to Israel timezone - "2022-11-07 23:01:00+00:00" --> "2022-11-08 01:01:00+02:00"
        #       Third, finally removing the timezone information "2022-02-28 10:00:00+02:00" --> "2022-02-28 10:00:00"
        sessions_df['start_date_time'] = pd.to_datetime(sessions_df['start_date_time'][0])
        
        # # Localize to UTC if necessary
        # if sessions_df['start_date_time'].dt.tz is None:
        #     sessions_df['start_date_time'] = sessions_df['start_date_time'].dt.tz_localize('UTC')
        # # Convert to Israel timezone
        # sessions_df['start_date_time'] = sessions_df['start_date_time'].dt.tz_convert('Israel')
        # Remove timezone information
        sessions_df['start_date_time'] = sessions_df['start_date_time'].dt.tz_localize(None)

        # Add 3 hours to the start_date_time to get the correct date
        sessions_df['start_date_time'] = sessions_df['start_date_time'] + pd.Timedelta(hours=3)

        sessions_df['end_date_time'] = pd.to_datetime(sessions_df['end_date_time'][0])


        # if sessions_df['end_date_time'].dt.tz is None:
        #     sessions_df['end_date_time'] = sessions_df['end_date_time'].dt.tz_localize('UTC')
        # sessions_df['end_date_time'] = sessions_df['end_date_time'].dt.tz_convert('Israel')
        sessions_df['end_date_time'] = sessions_df['end_date_time'].dt.tz_localize(None)

        sessions_df['end_date_time'] = sessions_df['end_date_time'] + pd.Timedelta(hours=3)

        # Mask valid rows. which are the dates that between the experiment start date and the experiment end date.
        mask_valid_rows = sessions_df['start_date_time'].between(subject_dates.iloc[0]['ExperimentStartDateTime'], subject_dates.iloc[0]['ExperimentEndDateTime'])
        # Remove all rows that are not valid
        sessions_df = sessions_df[mask_valid_rows]

        for session_id in sessions_df['session_id']:
            # Select from eda_mindfulness_df only the rows that has the same session_id as current sessions_df
            session_eda_mindfulness_df = eda_mindfulness_df[eda_mindfulness_df['session_id'] == session_id]
            total_avg = np.average(session_eda_mindfulness_df['scl_avg'])
            total_avg_1 = session_eda_mindfulness_df['scl_avg'].mean()
            session_eda_mindfulness_df = session_eda_mindfulness_df[session_eda_mindfulness_df['valid_data'] == True]
            valid_avg = np.average(session_eda_mindfulness_df['scl_avg'])
            valid_avg_1 = session_eda_mindfulness_df['scl_avg'].mean()
            stress_avgs["session_id"].append(session_id)
            if np.isnan(total_avg) or np.isnan(valid_avg) or np.isnan(total_avg_1) or np.isnan(valid_avg_1):
                subjects_with_missing_stress_files.append(subject + f' is missing {session_id} in Mindfulness Eda Data Sessions.csv')
                stress_avgs["total_avg"].append(np.NaN)
                stress_avgs["valid_avg"].append(np.NAN)
            elif total_avg != total_avg_1 or valid_avg_1 != valid_avg:
                subjects_with_weird_values.append(subject)
                stress_avgs["total_avg"].append(np.NaN)
                stress_avgs["valid_avg"].append(np.NAN)
            else: # Regular case
                stress_avgs["total_avg"].append(total_avg)
                stress_avgs["valid_avg"].append(valid_avg)

        stress_avgs = pd.DataFrame(stress_avgs)
        mindfulness_output_df = pd.merge(sessions_df, stress_avgs, on='session_id')
        # Rename columns
        mindfulness_output_df = mindfulness_output_df.rename(columns={'session_id': 'SessionId',
                                                                      'start_heart_rate': 'StartHeartRate',
                                                                      'end_heart_rate': 'EndHeartRate',
                                                                      'duration': 'MeasureInSeconds',
                                                                      'start_date_time': 'StartMeasureTime',
                                                                      'end_date_time': 'EndMeasureTime',
                                                                      'total_avg': '2MinutesMeanEDA',
                                                                      'valid_avg': 'ValidMeanEDA'})
        # Remove timezone information "2022-02-28 10:00:00+02:00" --> "2022-02-28 10:00:00"
        mindfulness_output_df['StartMeasureTime'] = mindfulness_output_df['StartMeasureTime'].dt.tz_localize(None)
        mindfulness_output_df['EndMeasureTime'] = mindfulness_output_df['EndMeasureTime'].dt.tz_localize(None)
        mindfulness_output_df = mindfulness_output_df.set_index('SessionId', drop=True)
        mindfulness_output_df = mindfulness_output_df.rename(columns={'average_heart_rate': 'AverageHeartRate'})


        # Add DateOfMeasure column
        mindfulness_output_df['DateOfMeasure'] = mindfulness_output_df['StartMeasureTime'].dt.date
        # Convert 'DateOfMeasure' column to Datetime object
        mindfulness_output_df['DateOfMeasure'] = pd.to_datetime(mindfulness_output_df['DateOfMeasure'])
        # If the measure start time is between 00 AM to 5 AM, the date of the sleep should be the previous day
        mindfulness_output_df.loc[mindfulness_output_df['StartMeasureTime'].dt.hour < 5, 'DateOfMeasure'] -= datetime.timedelta(days=1)

        # create a DatetimeIndex with the range of experiment dates
        subject_experiment_dates = pd.date_range(subject_dates['ExperimentStartDate'].values[0],
                                                 subject_dates['ExperimentEndDate'].values[0])

        # Convert the DatetimeIndex (subject_experiment_dates) to a DataFrame with a column named 'ExperimentStartDate'
        subject_experiment_dates_df = pd.DataFrame({'ExperimentDates': subject_experiment_dates})

        # Merge subject's mindfulness_output_df dataframe with the range of experiment dates.
        # Dropping the dates of sleep data that is not in dates of experiment.
        # Using 'right' join: "right: use only keys from right frame, similar to a SQL right outer join; preserve key order."
        # Creating nan rows with dates that are missing by doing a right join with the experiment dates.
        mindfulness_output_df = mindfulness_output_df.merge(subject_experiment_dates_df, how='right',
                                                  left_on='DateOfMeasure', right_on='ExperimentDates')
        # drop not relevant DateOfMeasure column
        mindfulness_output_df = mindfulness_output_df.drop(columns=['DateOfMeasure'])
        # Create a new column that contains the day of the Experiment.
        # For each unique date of 'ExperimentDates' column a consecutive rank value is assigned.
        # 'dense' = 'consecutive values'
        mindfulness_output_df['DayOfExperiment'] = mindfulness_output_df['ExperimentDates'].rank(method='dense')
        # sort by DayOfExperiment, StartMeasureTime
        mindfulness_output_df = mindfulness_output_df.sort_values(by=['DayOfExperiment', 'StartMeasureTime'])

        # Add Id column
        mindfulness_output_df['Id'] = subject
        # Save output CSV file to each subject output folder
        output_path = OUTPUT_PATH.joinpath(subject)
        
        output_path_history = ut.output_record(OUTPUT_PATH, subject, user_name, now )
        
        # Check that output_path is exists
        if not output_path.exists():
            output_path.mkdir()
        if not output_path_history.exists():
            output_path_history.mkdir()
        
        # Remove duplicates
        mindfulness_output_df = mindfulness_output_df.drop_duplicates(
            subset=['DayOfExperiment', 'StartMeasureTime', '2MinutesMeanEDA', 'ValidMeanEDA'],
            keep='first')

        # Remove rows with nan values in 'StartMeasureTime' and 'EndMeasureTime' columns
        mindfulness_output_df = mindfulness_output_df.dropna(subset=['StartMeasureTime', 'EndMeasureTime'])

        # Select specific columns to reorder the columns and save the resulting DataFrame to a CSV file
        mindfulness_output_df[['Id','DayOfExperiment', 'ExperimentDates',
                                                  'StartMeasureTime', 'EndMeasureTime', 'MeasureInSeconds',
                                                  '2MinutesMeanEDA', 'ValidMeanEDA',
                                                  'AverageHeartRate', 'StartHeartRate', 'EndHeartRate']].to_csv(output_path_history.joinpath(f'{subject} EDA.csv'), index=False) 
        
        ut.check_for_duplications(output_path, output_path_history)

    # If there are any subjects without stress files, logger.info the subject IDs
    if subjects_with_missing_stress_files:
        print(f'Subjects with missing mindfulness_eda_data_sessions or mindfulness_sessions files in FITBIT/Stress folder of each subject:')
        print('\n'.join(subjects_with_missing_stress_files))
    # If there are any subjects without stress files, logger.info the subject IDs
    if subjects_with_weird_values:
        print(f'Subjects with subjects with weird values in mindfulness_eda_data_sessions or mindfulness_sessions files inside FITBIT/Stress folder of each subject:')
        print('\n'.join(set(subjects_with_weird_values)))


    ############################## aggregate and summarize data ##############################
    all_subjects_summary_subjects = []
    all_subjects_details_df = pd.DataFrame()
    print('\n Means of EDA During Mindfulness session')
    
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

        # Get latest file of EDA.csv
        mindfulness_file = ut.new_get_latest_file_by_term('EDA', subject=subject, root=OUTPUT_PATH)

        # If there is no EDA.csv file in the subject folder, continue to the next subject
        if not mindfulness_file.exists():
            continue
        # Read the file to a DataFrame
        mindfulness_df = pd.read_csv(mindfulness_file)
        subject_summary_dict = {}
        subject_summary_dict['Id'] = subject
        subject_summary_dict['mean_2_minutes_EDA_mindfulness'] = mindfulness_df['2MinutesMeanEDA'].mean()
        subject_summary_dict['std_2_minutes_EDA_mindfulness'] = mindfulness_df['2MinutesMeanEDA'].std()
        subject_summary_dict['mean_valid_2_minutes_EDA_mindfulness'] = mindfulness_df['ValidMeanEDA'].mean()
        subject_summary_dict['std_valid_2_minutes_EDA_mindfulness'] = mindfulness_df['ValidMeanEDA'].std()

        # Append the dictionary to the list
        all_subjects_summary_subjects.append(subject_summary_dict)

        # Concatenate subject's dataframe with the dataframes of all subjects.
        all_subjects_details_df = pd.concat([all_subjects_details_df, mindfulness_df])


    # Create a DataFrame from the list of dictionaries
    all_subjects_summary_subjects_df = pd.DataFrame(all_subjects_summary_subjects)
    old_all_subjects_summary_subjects_df = ut.new_get_latest_file_by_term('EDA Summary', root=AGGREGATED_OUTPUT_PATH)
    if all_subjects_summary_subjects_df.empty:
        print('No EDA.csv files were found in the subjects folders')
        return
    if old_all_subjects_summary_subjects_df.exists():
        old_all_subjects_summary_subjects_df = pd.read_csv(old_all_subjects_summary_subjects_df)
        for subject in all_subjects_summary_subjects_df['Id']:
            if subject in old_all_subjects_summary_subjects_df['Id'].values:
                old_all_subjects_summary_subjects_df = old_all_subjects_summary_subjects_df[old_all_subjects_summary_subjects_df['Id'] != subject]
        all_subjects_summary_subjects_df = pd.concat([old_all_subjects_summary_subjects_df, all_subjects_summary_subjects_df])
    
    old_all_subjects_details_df = ut.new_get_latest_file_by_term('EDA All Subjects', root=AGGREGATED_OUTPUT_PATH)
    if old_all_subjects_details_df.exists():
        old_all_subjects_details_df = pd.read_csv(old_all_subjects_details_df)
        for subject in all_subjects_details_df['Id']:
            if subject in old_all_subjects_details_df['Id'].values:
                old_all_subjects_details_df = old_all_subjects_details_df[old_all_subjects_details_df['Id'] != subject]
        all_subjects_details_df = pd.concat([old_all_subjects_details_df, all_subjects_details_df])
    
    # Validate that the DataFrame is not empty
    if all_subjects_summary_subjects_df.empty:
        print('No EDA.csv files were found in the subjects folders')
        return
    # sort by id column and reset index
    all_subjects_summary_subjects_df = all_subjects_summary_subjects_df.sort_values(by=['Id']).reset_index(drop=True)
    all_subjects_details_df = all_subjects_details_df.sort_values(by=['Id', 'StartMeasureTime']).reset_index(drop=True)
    # Save output CSV file to project aggregated output folder
    all_subjects_summary_subjects_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'EDA Summary.csv'), index=False) 
    all_subjects_details_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'EDA All Subjects.csv'), index=False) 

    ut.check_for_duplications(AGGREGATED_OUTPUT_PATH, AGGREGATED_OUTPUT_PATH_HISTORY)
    





# Function to get the last Friday of March and last Sunday of October for a given year
def get_dst_change_dates(year):
    # Last Friday of March
    last_day_march = datetime.datetime(year, 3, 31)
    last_friday_march = last_day_march - datetime.timedelta(days=(last_day_march.weekday() + 3) % 7)

    # Last Sunday of October
    last_day_october = datetime.datetime(year, 10, 31)
    last_sunday_october = last_day_october - datetime.timedelta(days=(last_day_october.weekday() + 1) % 7)

    return last_friday_march, last_sunday_october

# Function to check if a date is a DST change date
def is_dst_change(date):
    year = date.year
    last_friday_march, last_sunday_october = get_dst_change_dates(year)
    return date.date() == last_friday_march.date() or date.date() == last_sunday_october.date()


                    


                    


            

            



            
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

    main(param, now, user_name)