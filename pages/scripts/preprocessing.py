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
if os.path.exists(rf'C:\Users\PsyLab-6028'):
    sys.path.append(r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages')
else:
    sys.path.append(r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages')

import UTILS.utils as ut
# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings
warnings.filterwarnings('ignore')





def main(project, now, username):
    try:
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

        try:
            subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                        try_parse_dates=True)
        except:
            subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                        parse_dates=True,
                                        encoding='utf-8')

        subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')
        if os.path.exists(rf'C:\Users\PsyLab-6028'):
            selected_subjects = pl.read_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_Preprocessing.parquet')
        else:
            selected_subjects = pl.read_parquet(rf'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_Preprocessing.parquet') 
            
    
        run_on_specific_subjects = True
        subjects_to_run_on = selected_subjects['subject'].to_list()

        subjects_with_missing_heart_rate_files = []
        print('Process Heart Rate Data')
        tqdm_subjects = tqdm(os.listdir(DATA_PATH))
        for subject in tqdm_subjects:
            if not re.search(r'\d{3}$', subject):
                continue
            if run_on_specific_subjects and subject not in subjects_to_run_on:
                continue
            tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)
            # if the folder not look like 'sub_xxx' so skip it
            # if not re.fullmatch(SUBJECT_FOLDER_FORMAT, subject):
            #     continue #@TODO: GET BACK IN
            # Set up the path to the subject's sleep folder
            if not os.path.exists(OUTPUT_PATH.joinpath(subject)):
                os.mkdir(OUTPUT_PATH.joinpath(subject))
            heart_rate_json_directory = DATA_PATH.joinpath(f'{subject}\FITBIT\Physical Activity')
            try:
                # Read the subject_HR_json_data from the pickle file
                with open(OUTPUT_PATH.joinpath(subject).joinpath('from_code_subject_HR_df.pkl'), 'rb') as f:
                    subject_HR_df = pickle.load(f)
            except:
                # Find all "heart_rate-YYYY-MM-DD.json" files in the Physical Activity folder
                heart_rate_file_name_pattern = re.compile(r'^heart_rate-\d{4}-\d{2}-\d{2}.json')
                heart_rate_api_file_name_pattern = re.compile(r'^api-heart_rate-\d{4}-\d{2}-\d{2}.json')
                heart_rate_json_files = [file_name for file_name in os.listdir(heart_rate_json_directory)
                                        if heart_rate_file_name_pattern.search(file_name)]
                heart_rate_api_files = [file_name for file_name in os.listdir(heart_rate_json_directory)
                                        if heart_rate_api_file_name_pattern.search(file_name)]
                # If the subject is missing heart rate files, skip them and add their name to the list
                if not heart_rate_json_files:
                    if not heart_rate_api_files:
                        subjects_with_missing_heart_rate_files.append(subject)
                        continue  # skipping current subject
                if heart_rate_json_files:
                    # Sort heart rate files
                    heart_rate_json_files = sorted(heart_rate_json_files)
                    # Merge all heart rate files to one list
                    all_files_df = pl.DataFrame()
                    for json_hr_file in heart_rate_json_files:
                        file_df = pl.read_json(os.path.join(heart_rate_json_directory, json_hr_file))
                        all_files_df = pl.concat([all_files_df, file_df])

                    subject_HR_df = (
                        all_files_df
                        .with_columns(
                            pl.col('dateTime')
                            .str.strptime(pl.Datetime, '%m/%d/%y %H:%M:%S')
                            .dt.convert_time_zone('Israel')
                            .dt.replace_time_zone(None)
                            .dt.cast_time_unit('ns')
                        )  
                        .unnest('value')
                        .to_pandas()
                    )                

                    with open(OUTPUT_PATH.joinpath(subject).joinpath('from_code_subject_HR_df.pkl'), 'wb') as f:
                        pickle.dump(subject_HR_df, f)
                elif heart_rate_api_files:
                    # Sort heart rate files
                    heart_rate_api_files = sorted(heart_rate_api_files)
                    # Merge all heart rate files to one list
                    subject_HR_json_data = []
                    for file_name in heart_rate_api_files:
                        with open(heart_rate_json_directory.joinpath(file_name)) as f:
                            subject_HR_json_data.extend(json.load(f))

                    ########## Preprocess data ##########

                    # Convert to dataframe with the columns: 'dateTime', 'value', 'confidence'.
                    subject_HR_df = pd.json_normalize(subject_HR_json_data)  # TODO: small bottleneck
                    # Convert dates columns to datetime object: '2021-11-16T00:16:30.000' --> '2021-11-16 00:16:30'
                    subject_HR_df['dateTime'] = pd.to_datetime(subject_HR_df['dateTime'])  # TODO: big bottleneck

                    with open(OUTPUT_PATH.joinpath(subject).joinpath('from_code_subject_HR_df.pkl'), 'wb') as f:
                        pickle.dump(subject_HR_df, f)        
            # Rename columns
            if 'value.bpm' in subject_HR_df.columns:
                subject_HR_df = subject_HR_df.rename(columns={'value.bpm': 'bpm',
                                                            'value.confidence': 'confidence'})
                
            # Get start and end dates of experiment for current subject
            subject_dates = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
            try:
                # Mask valid rows. which are the dates that between the experiment start date and the experiment end date.
                subject_HR_df = subject_HR_df[
                    subject_HR_df['dateTime'].between(subject_dates.iloc[0]['ExperimentStartDateTime'],
                                                    subject_dates.iloc[0]['ExperimentEndDateTime'])]
                # if subject_dates['NotInIsrael'].values[0]:
                #     relevant_subject_steps_df_before = subject_HR_df[subject_HR_df['dateTime'] < subject_dates['NotInIsraelStartDate'].values[0]]
                #     relevant_subject_steps_df_after = subject_HR_df[subject_HR_df['dateTime'] > pd.to_datetime(subject_dates['NotInIsraelEndDate'].values[0]) + datetime.timedelta(days=1)]
                #     subject_HR_df = pd.concat([relevant_subject_steps_df_before, relevant_subject_steps_df_after])
                
                # if subject_dates['NotInIsrael_1'].values[0]:
                #     relevant_subject_steps_df_before = subject_HR_df[subject_HR_df['dateTime'] < subject_dates['NotInIsraelStartDate_1'].values[0]]
                #     relevant_subject_steps_df_after = subject_HR_df[subject_HR_df['dateTime'] > pd.to_datetime(subject_dates['NotInIsraelEndDate_1'].values[0]) + datetime.timedelta(days=1)]
                #     subject_HR_df = pd.concat([relevant_subject_steps_df_before, relevant_subject_steps_df_after])

            except:
                subjects_with_missing_heart_rate_files.append(subject)
                continue
            # # Remove all rows that are not valid
            # subject_HR_df = subject_HR_df[mask_valid_rows]

            ########## Create Dataframe 1: Select Valid data, resample the 'subject_HR_df' DataFrame by minute, backward-filling missing data ##########

            # Add a new column to the DataFrame named 'Valid'. Initialize it with 0 temporarily.
            subject_HR_df['valid'] = 0
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$old filter$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # Set 1 at 'Valid' column where confidence is above 1 and heart rate bpm is between 40 to 180.
            # subject_HR_df.loc[(subject_HR_df['confidence'] > 1) &
            #                            (subject_HR_df['bpm'] >= 40) &
            #                            (subject_HR_df['bpm'] <= 180), 'valid'] = 1
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$old filter$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

            # Set 1 at 'Valid' column where confidence is above 1 and heart rate bpm is between 40 to 180.
            subject_HR_df.loc[(subject_HR_df['confidence'] > 0) &
                            (subject_HR_df['bpm'] >= 40) &
                            (subject_HR_df['bpm'] <= 180), 'valid'] = 1

            # Group by minute and count the number of bpm samples per minute.
            num_of_all_samples_df = subject_HR_df[['dateTime', 'bpm']].groupby(pd.Grouper(key='dateTime', freq='1Min')).agg(['count'])
            num_of_all_samples_df.columns = ['NumOfAllSamples']

            # if NumOfAllSamples is 0, set "unknown" value to Mode column
            num_of_all_samples_df.loc[num_of_all_samples_df['NumOfAllSamples'] == 0, 'Mode'] = 'unknown'
            # Select valid rows
            valid_heart_rate_merged = subject_HR_df.loc[subject_HR_df['valid'] == 1]
            # Group by minute and count the number of bpm samples per minute. Create data with one column named "bpm/count"
            num_of_valid_samples_df = valid_heart_rate_merged[['dateTime', 'bpm']].groupby(pd.Grouper(key='dateTime', freq='1Min')).agg(['count'])
            # Rename column "bpm/count" to "NumOfValidSamples"
            num_of_valid_samples_df.columns = ['NumOfValidSamples']

            # Add Mode column that equals to num_of_valid_samples_df
            num_of_valid_samples_df['Mode'] = num_of_all_samples_df['Mode']
            num_of_valid_samples_df['NumOfAllSamples'] = num_of_all_samples_df['NumOfAllSamples']
            ########## Create Dataframe 2: Resample the 'subject_HR_df' DataFrame by second, backward-filling missing data ##########

            # Set the index of subject_HR_df to dateTime. It is a must for upsampling by seconds.
            subject_HR_df = subject_HR_df.set_index('dateTime')
            # Resample 'subject_HR_df' DataFrame by second (without filter the valid rows), backward-filling missing data of the 4 (all) columns: 'dateTime', 'bpm', 'confidence', 'valid'.
            # Each row correspond to a second.
            # bfill(): Use next observation to fill the rows before / Fills missing values in the DataFrame with the value of the next row that is not nan.
            bpm_confidence_valid_resample_by_seconds = subject_HR_df.resample('S').bfill().reset_index()
            # Select rows from the DataFrame where the 'valid' column equals 1, and only keep the 'dateTime' and 'bpm' columns
            bpm_valid_resample_by_seconds_filtered = bpm_confidence_valid_resample_by_seconds.loc[bpm_confidence_valid_resample_by_seconds['valid'] == 1, ['dateTime', 'bpm']]
            # Group the filtered DataFrame by minute intervals on the 'dateTime' column and compute the mean and count of the 'bpm' column for each minute interval.
            # The count represents the number of seconds where the samples are valid (after up-sampling to minutes and backward fill) and also the number of valid samples after bfill.
            bpm_group_by_minute_after_bfill = bpm_valid_resample_by_seconds_filtered.groupby(pd.Grouper(key='dateTime', freq='1Min')).agg(['mean', 'count'])
            # Rename columns
            bpm_group_by_minute_after_bfill.columns = ['BpmMean', 'NumOfValidSamplesAfterBfill']

            ########## Merge Dataframe 1 ('BpmMean') and Dataframe 2 ('NumOfValidSamples', 'NumOfValidSamplesAfterBfill') by date ##########
            # creating dataframe with 3 columns: 'NumOfValidSamples', 'BpmMean', 'NumOfValidSamplesAfterBfill'.
            # 'outer': Use dateTime index from both dataframes. meaning if there is missing dateTime in one of them - it will appear with missing values.
            merged_heart_rate_df = num_of_valid_samples_df.merge(bpm_group_by_minute_after_bfill, how='outer', left_index=True, right_index=True)
            # Set nan in rows with less than 4 heart rate samples or in rows with less than 30 valid samples after bfill.
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$OLD FILTER$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            # merged_heart_rate_df.loc[(merged_heart_rate_df['NumOfValidSamples'] < 4) | (merged_heart_rate_df['NumOfValidSamplesAfterBfill'] < 30), 'BpmMean'] = np.NaN
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$OLD FILTER$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            merged_heart_rate_df.loc[(merged_heart_rate_df['NumOfValidSamples'] <= 1) | (merged_heart_rate_df['NumOfValidSamplesAfterBfill'] <= 1), 'BpmMean'] = np.NaN

            # Rename index column
            merged_heart_rate_df.index.name = 'DateTime'

            ########## Save output CSV file to each subject output folder ##########

            # Set up the path to the subject's output folder
            subject_output_path_history = ut.output_record(OUTPUT_PATH, subject, user_name, now)
            subject_output_path = OUTPUT_PATH.joinpath(subject)
            
            if not subject_output_path_history.exists():
                subject_output_path_history.mkdir(parents=True)
                
                
            # Select specific columns to reorder the columns and save the resulting DataFrame to a CSV file
            merged_heart_rate_df[['BpmMean',
                                'NumOfValidSamples',
                                'NumOfValidSamplesAfterBfill',
                                'NumOfAllSamples',
                                'Mode']].to_csv(subject_output_path_history.joinpath(f'{subject} Heart Rate.csv')) 
            
            ut.check_for_duplications(subject_output_path, subject_output_path_history)
            
        # If there are any missing heart_rate files for any subjects, logger.info the subject IDs
        if subjects_with_missing_heart_rate_files:
            print(f'Subjects with missing heart rate files inside FITBIT/Physical Activity folder of each subject:')
            print(subjects_with_missing_heart_rate_files)
        # Create an empty list to hold any subjects that are missing HRV or respiratory files
        subjects_with_missing_steps_files = []
        full_subjects_steps_list = []
        all_subjects_steps_df = pd.DataFrame()

        print('\n Process and aggregate steps Data')
        tqdm_subjects = tqdm(os.listdir(DATA_PATH))
        for subject in tqdm_subjects:

            if run_on_specific_subjects and subject not in subjects_to_run_on:
                continue
            tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)
            
            if not re.search(r'\d{3}$', subject):
                continue
            # Set up the path to the subject's sleep folder
            steps_json_directory = DATA_PATH.joinpath(f'{subject}\FITBIT\Physical Activity')
            # Find all "steps-YYYY-MM-DD.json" files in the Physical Activity folder
            steps_file_name_pattern = re.compile(r'^steps-\d{4}-\d{2}-\d{2}.json')

            steps_api_file_name_pattern = re.compile(r'^api-steps-\d{4}-\d{2}-\d{2}.json')

            steps_json_files = [file_name for file_name in os.listdir(steps_json_directory)
                                if steps_file_name_pattern.search(file_name)]

            steps_api_files = [file_name for file_name in os.listdir(steps_json_directory)
                                if steps_api_file_name_pattern.search(file_name)]
            # If the subject is missing steps files, skip them and add their name to the list
            if not steps_json_files:
                if not steps_api_files:
                    subjects_with_missing_steps_files.append(subject)
                    continue # skipping current subject

            # Sort the files by date (the date is in the file name)
            steps_json_files = sorted(steps_json_files, reverse=True)

            steps_api_files = sorted(steps_api_files, reverse=True)

            try:
                with open(OUTPUT_PATH.joinpath(subject).joinpath('steps_df_after_datetime_conversion.pkl'), 'rb') as f:
                    subject_steps_df = pickle.load(f)
            except:
                steps_file_name_pattern = re.compile(r'^steps-\d{4}-\d{2}-\d{2}.json')
                steps_api_name_pattern = re.compile(r'^api-steps-\d{4}-\d{2}-\d{2}.json')
                steps_json_files = [file_name for file_name in os.listdir(steps_json_directory)
                                    if steps_file_name_pattern.search(file_name)]
                steps_api_files = [file_name for file_name in os.listdir(steps_json_directory)
                                    if steps_api_name_pattern.search(file_name)]
                if not steps_json_files:
                    if not steps_api_files:
                        subjects_with_missing_steps_files.append(subject)
                        continue
                if steps_json_files:
                    # Iterate over the files and add them to the full list steps data of all subjects
                    steps_json_files = sorted(steps_json_files, reverse=True)
                    all_steps_files_df = pl.DataFrame()
                    for json_steps_file in steps_json_files:
                        file_df = pl.read_json(os.path.join(steps_json_directory, json_steps_file))
                        all_steps_files_df = pl.concat([all_steps_files_df, file_df])
                    
                    # convert the 'dateTime' column to datetime
                    subject_steps_df = (
                        all_steps_files_df
                        .with_columns(
                            pl.col('dateTime')
                            .str.strptime(pl.Datetime, '%m/%d/%y %H:%M:%S')
                            .dt.convert_time_zone('Israel')
                            .dt.replace_time_zone(None)
                            .dt.cast_time_unit('ns'),
                            value=pl.col('value').cast(pl.Int64)
                        )
                        .to_pandas()
                    )

                    with open(OUTPUT_PATH.joinpath(subject).joinpath('steps_df_after_datetime_conversion.pkl'), 'wb') as f:
                        pickle.dump(subject_steps_df, f)
                elif steps_api_files:
                    steps_api_files = sorted(steps_api_files, reverse=True)
                    subject_steps_list = []
                    for json_file in steps_api_files:
                        with open(steps_json_directory.joinpath(json_file)) as f:
                            # json to dict
                            subject_steps_list.extend(json.load(f))
                    # Convert the list of dictionaries to a dataframe
                    subject_steps_df = pd.json_normalize(subject_steps_list)

                    # convert the 'dateTime' column to datetime
                    subject_steps_df['dateTime'] = pd.to_datetime(subject_steps_df['dateTime'])

                    with open(OUTPUT_PATH.joinpath(subject).joinpath('steps_df_after_datetime_conversion.pkl'), 'wb') as f:
                        pickle.dump(subject_steps_df, f)
                

            tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)
            relevant_subject_steps_df = subject_steps_df
            if relevant_subject_steps_df.shape[0] == 0:
                subjects_with_missing_steps_files.append(f'{subject} is missing steps files')
                continue
            # Remove dates that are not in the range of the experiment
            subject_dates_of_experiment = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
            # Select only the dates that are in the range of the experiment dates
            relevant_subject_steps_df = relevant_subject_steps_df[
                relevant_subject_steps_df['dateTime'].between(subject_dates_of_experiment.iloc[0]['ExperimentStartDateTime'],
                                                            subject_dates_of_experiment.iloc[0]['ExperimentEndDateTime'])]
            
            # If the subject is missing steps data/files, add their name to the missing files list
            if len(relevant_subject_steps_df) == 0:
                subjects_with_missing_steps_files.append(subject)
            # add index to the list of valid indexes
            # valid_indexes.extend(index_of_relevant_subject_steps_df)

            # subject_steps_df = subject_steps_df.loc[valid_indexes]
            
            relevant_subject_steps_df = relevant_subject_steps_df.rename(columns={'dateTime': 'DateAndMinute',
                                                                'value': 'StepsInMinute'})
            
            relevant_subject_steps_df = relevant_subject_steps_df[['DateAndMinute', 'StepsInMinute']].sort_values('DateAndMinute')
            
            # Save output CSV file to Output folder
            subject_output_path_history = ut.output_record(OUTPUT_PATH, subject, user_name, now)
            subject_output_path = OUTPUT_PATH.joinpath(subject)
            
            if not subject_output_path_history.exists():
                subject_output_path_history.mkdir(parents=True, exist_ok=True)
                
            relevant_subject_steps_df.to_csv(subject_output_path_history.joinpath(f'{subject} Steps.csv'))
            
            ut.check_for_duplications(subject_output_path, subject_output_path_history)
            
        if subjects_with_missing_steps_files:
            print(f'Subjects with missing or not relevant dates of FITBIT/Physical Activity/steps.json file:')
            print('\n'.join(subjects_with_missing_steps_files))
            print('Change/Delete the missing files and run the script again')
        
        subjects_with_missing_steps_files = []
        full_subjects_steps_list = []
        subject_steps_list = []
        all_subjects_steps_df = pd.DataFrame()
        
        print('\n Aggregate steps Data')
        tqdm_subjects = tqdm(os.listdir(DATA_PATH))
        for subject in tqdm_subjects:
            if run_on_specific_subjects and subject not in subjects_to_run_on:
                continue
            tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)
            if not re.search(r'\d{3}$', subject):
                continue
            
            subject_steps_path = OUTPUT_PATH.joinpath(subject).joinpath(f'{subject} Steps.csv')
            if not subject_steps_path.exists():
                subjects_with_missing_steps_files.append(subject)
                continue
            subject_steps_df = pd.read_csv(subject_steps_path)
            subject_steps_df['Id'] = [subject] * len(subject_steps_df)
            
            all_subjects_steps_df = pd.concat([subject_steps_df, all_subjects_steps_df])
            
            
        # Change the order of columns and sort
        all_subjects_steps_df = all_subjects_steps_df[['Id', 'DateAndMinute', 'StepsInMinute']].sort_values(['Id', 'DateAndMinute'])
        
        all_subjects_steps_df = concate_to_old('Steps', AGGREGATED_OUTPUT_PATH, all_subjects_steps_df)
        # Save output CSV file to Aggregated Output folder
        all_subjects_steps_df.to_csv(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Steps Aggregated.csv'), index=False) 
        all_subjects_steps_df.to_parquet(AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'Steps Aggregated.parquet'), index=False)
        ut.check_for_duplications(AGGREGATED_OUTPUT_PATH, AGGREGATED_OUTPUT_PATH_HISTORY)
        
        if subjects_with_missing_steps_files:
            print(f'Subjects with missing steps.csv file:')
            print('\n'.join(subjects_with_missing_steps_files))
            print('Change/Delete the missing files and run the script again')

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
        param = 'FIBRO_TESTS'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        user_name = 'Unknown'

    main(param, now, user_name)