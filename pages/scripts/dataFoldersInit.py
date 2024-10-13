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
sys.path.append(r'G:\Shared drives\AdmonPsy - Code\Idanko\Scripts')

import UTILS.utils as ut
import warnings
warnings.filterwarnings('ignore')





def main(project, now, username):
    # try:
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

    # Create and Validation the file that will write information about the missing files.
    missing_files_list = []



    # Load the source data path to copy raw files that downloaded from the fitbit cloud.
    source_data_path = Path(PROJECT_CONFIG[project])
    # source_data_path = Path('H:\Shared drives\Sigal Lab - Psychotherapy HR MDD Project\Experiment\Processed Data T2\P_Data')


    # Load the subjects dates of experiment file.
    subjects_dates_df = pl.read_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_init.parquet').sort(by='Id').unique('Id').drop_nulls('Id')
    
    subjects_dates_df = (
        subjects_dates_df
        .select(
            pl.selectors.exclude('Date')
        )
        .join(
            subjects_dates,
            on='Id',
            how='inner'
        )

    )


    
    tqdm_subjects = tqdm(subjects_dates_df.to_pandas().iterrows())

    for index, subject_row in tqdm_subjects:
        try:
            
            tqdm_subjects.set_description(f'Subject: {subject_row.Id}', refresh=True)
            # Get folder of current subject / row
            subject_output_folder = OUTPUT_PATH.joinpath(subject_row['Id'])
            # Create Output folder for current subject
            if not subject_output_folder.exists():
                os.makedirs(subject_output_folder)

            subjects_with_takeout = []
            # Create a range of dates that incriments by one day from the subject['ExperimentStartDate'] to the subject['ExperimentEndDate']
            subject_dates_range = pd.date_range(start=subject_row['ExperimentStartDate'], end=subject_row['ExperimentEndDate'])
            
            # Retrieve Sleep, Stress and Physical Activity paths of the source folder of the current subject
            source_fitbit_folder = 'FITBIT'
            source_subject_data_folder = source_data_path.joinpath(subject_row['Id'])
            source_subject_sleep_folder = source_subject_data_folder.joinpath(source_fitbit_folder).joinpath('Sleep')
            source_subject_stress_folder = source_subject_data_folder.joinpath(source_fitbit_folder).joinpath('Stress')
            source_subject_physical_activity_folder = source_subject_data_folder.joinpath(source_fitbit_folder).joinpath('Physical Activity')
            source_subject_takeout_folder = source_subject_data_folder.joinpath('Takeout')

            # Retrieve Sleep, Stress and Physical Activity paths of the target folder of the current subject
            target_fitbit_folder = 'FITBIT'
            target_subject_data_folder = target_data_path.joinpath(subject_row['Id'])
            target_subject_sleep_folder = target_subject_data_folder.joinpath(target_fitbit_folder).joinpath('Sleep')
            target_subject_stress_folder = target_subject_data_folder.joinpath(target_fitbit_folder).joinpath('Stress')
            target_subject_physical_activity_folder = target_subject_data_folder.joinpath(target_fitbit_folder).joinpath('Physical Activity')

            # Create Target Data folders for current subject
            if not target_subject_data_folder.joinpath(target_fitbit_folder).exists():
                # Create Subject folder
                os.makedirs(target_subject_data_folder.joinpath(target_fitbit_folder))
                # Create Sleep folder
                os.makedirs(target_subject_sleep_folder)
                # Create Stress folder
                os.makedirs(target_subject_stress_folder)
                # Create Physical Activity folder
                os.makedirs(target_subject_physical_activity_folder)

            
            if source_subject_takeout_folder.exists():
                subjects_with_takeout.append(subject_row['Id'])

                source_fitbit_folder = source_subject_data_folder.joinpath('FITBIT')
                if not source_fitbit_folder.exists():
                    source_fitbit_folder = source_subject_data_folder.joinpath('FITBIT')
                    os.makedirs(source_fitbit_folder)

                if not source_subject_sleep_folder.exists():
                    source_subject_sleep_folder = source_fitbit_folder.joinpath('Sleep')
                    os.makedirs(source_subject_sleep_folder)

                if not source_subject_stress_folder.exists():
                    source_subject_stress_folder = source_fitbit_folder.joinpath('Stress')
                    os.makedirs(source_subject_stress_folder)

                if not source_subject_physical_activity_folder.exists():
                    source_subject_physical_activity_folder = source_fitbit_folder.joinpath('Physical Activity')
                    os.makedirs(source_subject_physical_activity_folder)

                global_export_folder = source_subject_takeout_folder.joinpath('Fitbit').joinpath('Global Export Data')

                if not global_export_folder.exists():
                    continue

                # Copy the files from the takeout folder to the subject source folders

                for file in os.listdir(global_export_folder):
                    if file.startswith('heart_rate-'):
                        shutil.copy(global_export_folder.joinpath(file), source_subject_physical_activity_folder.joinpath(file))
                    elif file.startswith('sleep-'):
                        shutil.copy(global_export_folder.joinpath(file), source_subject_sleep_folder.joinpath(file))
                    elif file.startswith('steps-'):
                        shutil.copy(global_export_folder.joinpath(file), source_subject_physical_activity_folder.joinpath(file))
                
                heart_rate_variability_folder = source_subject_takeout_folder.joinpath('Fitbit').joinpath('Heart Rate Variability')

                if not heart_rate_variability_folder.exists():
                    print(f"Subject {subject_row['Id']} does not have heart rate variability folder")

                if heart_rate_variability_folder.exists():
                    for file in os.listdir(heart_rate_variability_folder):
                        shutil.copy(heart_rate_variability_folder.joinpath(file), source_subject_sleep_folder.joinpath(file))
                
                temperature_folder = source_subject_takeout_folder.joinpath('Fitbit').joinpath('Temperature')

                if not temperature_folder.exists():
                    print(f"Subject {subject_row['Id']} does not have temperature folder")

                if temperature_folder.exists():
                    for file in os.listdir(temperature_folder):
                        shutil.copy(temperature_folder.joinpath(file), source_subject_sleep_folder.joinpath(file))

                
                mindfulness_folder = source_subject_takeout_folder.joinpath('Fitbit').joinpath('Mindfulness')

                if not mindfulness_folder.exists():
                    print(f"Subject {subject_row['Id']} does not have mindfulness folder")

                if mindfulness_folder.exists():
                    for file in os.listdir(mindfulness_folder):
                        shutil.copy(mindfulness_folder.joinpath(file), source_subject_stress_folder.joinpath(file))

                print(f"Subject {subject_row['Id']} takeout files copied successfully")

            if os.path.exists(source_subject_stress_folder):

                ### Copy Stress/Mindfulness/EDA files ###
                # Find the files that contains the EDA data of the mindfulness sessions in FITBIT/Stress folder.
                eda_mindfulness_file_name_pattern = re.compile(r'^Mindfulness Eda Data Sessions.csv')
                eda_mindfulness_file_name_pattern_alternative = re.compile(r'^mindfulness_eda_data_sessions.csv')
                eda_mindfulness_csv_files = [file_name for file_name in os.listdir(source_subject_stress_folder) if eda_mindfulness_file_name_pattern.search(file_name)]
                eda_mindfulness_csv_files_alternative = [file_name for file_name in os.listdir(source_subject_stress_folder) if eda_mindfulness_file_name_pattern_alternative.search(file_name)]
                eda_mindfulness_csv_files.extend(eda_mindfulness_csv_files_alternative)
                eda_mindfulness_found = False
                sessions_found = False
                # Validate that there is only 1 file that that called "mindfulness_eda_data_sessions".
                if len(eda_mindfulness_csv_files) == 1:
                    eda_mindfulness_file = eda_mindfulness_csv_files[0]
                    eda_mindfulness_found = True
                else:
                    missing_files_list.append({'subject': subject_row['Id'], 'missing_file': 'FITBIT/Stress/Mindfulness Eda Data Sessions.csv'})

                sessions_file_name_pattern = re.compile(r'^Mindfulness Sessions.csv')
                sessions_file_name_pattern_alternative = re.compile(r'^mindfulness_sessions.csv')
                sessions_csv_files = [file_name for file_name in os.listdir(source_subject_stress_folder)
                                    if sessions_file_name_pattern.search(file_name)]
                sessions_csv_files_alternative = [file_name for file_name in os.listdir(source_subject_stress_folder)
                                    if sessions_file_name_pattern_alternative.search(file_name)]
                sessions_csv_files.extend(sessions_csv_files_alternative)
                # Validate that there is only 1 file that that called "mindfulness_sessions".
                if len(sessions_csv_files) == 1:
                    sessions_file = sessions_csv_files[0]
                    sessions_found = True
                else:
                    missing_files_list.append({'subject': subject_row['Id'], 'missing_file': 'FITBIT/Stress/Mindfulness Sessions.csv'})

                if eda_mindfulness_found and sessions_found:
                    # Retrieve the paths of the eda files
                    source_eda_mindfulness_path = source_subject_stress_folder.joinpath(eda_mindfulness_file)
                    source_sessions_path = source_subject_stress_folder.joinpath(sessions_file)
                    # Create the target paths of the eda files
                    target_eda_mindfulness_path = target_subject_stress_folder.joinpath(eda_mindfulness_file)
                    target_sessions_path = target_subject_stress_folder.joinpath(sessions_file)
                    # No need to check the dates of the mindfulness_eda_data_sessions file because if the mindfulness_sessions file is not valid,
                    # so there are no sessions id to find in the mindfulness_eda_data_sessions file.
                    shutil.copy(source_eda_mindfulness_path, target_eda_mindfulness_path)
                    shutil.copy(source_sessions_path, target_sessions_path)
            else:
                missing_files_list.append({'subject': subject_row['Id'], 'missing_file': 'FITBIT/Stress/Mindfulness Eda Data Sessions.csv'})
                missing_files_list.append({'subject': subject_row['Id'], 'missing_file': 'FITBIT/Stress/Mindfulness Sessions.csv'})
                
            
            
            date_regex = re.compile(r'\d{4}-\d{2}-\d{2}')

            ### Heart Rate files ###
            # Find all heart rate files in the FITBIT/Physical Activity folder
            heart_rate_file_name_pattern = re.compile(r'heart_rate-\d{4}-\d{2}-\d{2}.json$')
            heart_rate_json_files = [file_name for file_name in os.listdir(source_subject_physical_activity_folder)
                                        if heart_rate_file_name_pattern.search(file_name)]
            heart_rate_files_to_copy = []


            for day in subject_dates_range:
                file_name = f'heart_rate-{day.strftime("%Y-%m-%d")}.json'
                file_name_api = f'api_heart_rate-{day.strftime("%Y-%m-%d")}.json'
                if file_name in heart_rate_json_files:
                    heart_rate_files_to_copy.append(file_name)
                elif file_name_api in heart_rate_json_files:
                    heart_rate_files_to_copy.append(file_name_api)
                else:
                    missing_files_list.append({'subject': subject_row['Id'], 'missing_file': f'FITBIT/Physical Activity/{file_name}'})
            
            


            # Copy only heart rate files which dates are valid
        
            for file in heart_rate_files_to_copy:
                shutil.copy(source_subject_physical_activity_folder.joinpath(file),
                            target_subject_physical_activity_folder.joinpath(file))


            ### Heart Rate Variability Details files ###
            # Find all "Heart Rate Variability Details - YYYY-MM-DD.csv" files in the sleep folder
            HRV_details_file_name_pattern = re.compile(r"^Heart Rate Variability Details - \d{4}-\d{2}-\d{2}.csv")
            HRV_details_files = [file_name for file_name in os.listdir(source_subject_sleep_folder)
                                    if HRV_details_file_name_pattern.search(file_name)]
            HRV_details_files_to_copy = []
            
            
            for day in subject_dates_range:
                file_name = f'Heart Rate Variability Details - {day.strftime("%Y-%m-%d")}.csv'
                if file_name in HRV_details_files:
                    HRV_details_files_to_copy.append(file_name)
                else:
                    missing_files_list.append({'subject': subject_row['Id'], 'missing_file': f'FITBIT/Sleep/{file_name}'})
            

            # Copy Heart Rate Variability Details files
            for file in HRV_details_files_to_copy:
                shutil.copy(source_subject_sleep_folder.joinpath(file),
                            target_subject_sleep_folder.joinpath(file))

            ### Computed Temperature files ### not used anymore -TODO
            # Find the file "Computed Temperature - YYYY-MM-DD.csv" in the sleep folder
            computed_temperature_file_name_pattern = re.compile(r"^Computed Temperature - \d{4}-\d{2}-\d{2}\.csv")
            computed_temperature_files = [file_name for file_name in os.listdir(source_subject_sleep_folder)
                                            if computed_temperature_file_name_pattern.search(file_name)]
            if len(computed_temperature_files) == 0:
                missing_files_list.append({'subject': subject_row['Id'], 'missing_file': f'FITBIT/Sleep/Computed Temperature - YYYY-MM-DD.csv'})


            # Copy Heart Rate Variability Details files
            for file in computed_temperature_files:
                shutil.copy(source_subject_sleep_folder.joinpath(file),
                            target_subject_sleep_folder.joinpath(file))

            ### Device Temperature files ###
            # Find all "Device Temperature - YYYY-MM-DD.csv" files in the sleep folder
            device_temperature_file_name_pattern = re.compile(r"^Device Temperature - \d{4}-\d{2}-\d{2}")
            device_temperature_files = [file_name for file_name in os.listdir(source_subject_sleep_folder)
                                        if device_temperature_file_name_pattern.search(file_name)]
            device_temperature_files_to_copy = []
            
            for day in subject_dates_range:
                file_name = f'Device Temperature - {day.strftime("%Y-%m-%d")}.csv'
                if file_name in device_temperature_files:
                    device_temperature_files_to_copy.append(file_name)
                else:
                    missing_files_list.append({'subject': subject_row['Id'], 'missing_file': f'FITBIT/Sleep/{file_name}'})
            

            # Copy Device Temperature files

            for file in device_temperature_files_to_copy:
                shutil.copy(source_subject_sleep_folder.joinpath(file),target_subject_sleep_folder.joinpath(file))

            ### Daily Respiratory Rate Summary files ###
            # Find all "Daily Respiratory Rate Summary - YYYY-MM-DD.csv" files in the sleep folder
            respiratory_file_name_pattern = re.compile(r"^Daily Respiratory Rate Summary - \d{4}-\d{2}-\d{2}\.csv")
            respiratory_files = [file_name for file_name in os.listdir(source_subject_sleep_folder)
                                    if respiratory_file_name_pattern.search(file_name)]
            respiratory_files_to_copy = []
            
            for day in subject_dates_range:
                file_name = f'Daily Respiratory Rate Summary - {day.strftime("%Y-%m-%d")}.csv'
                if file_name in respiratory_files:
                    respiratory_files_to_copy.append(file_name)
                else:
                    missing_files_list.append({'subject': subject_row['Id'], 'missing_file': f'FITBIT/Sleep/{file_name}'})
            

            # Copy Daily Respiratory Rate Summary files
            for file in respiratory_files_to_copy:
                shutil.copy(source_subject_sleep_folder.joinpath(file), target_subject_sleep_folder.joinpath(file))


            ### Sleep files ###
            # The validation of sleep dates are in process_and_aggregate_sleep function. so we don't need to check the dates here.
            # only copy all the files.
            
            pattern = re.compile(r'^sleep-(\d{4}-\d{2}-\d{2})\.json')
            sleep_files = [file_name for file_name in os.listdir(source_subject_sleep_folder) if
                            pattern.search(file_name)]
            # Copy sleep files
            for file in sleep_files:
                shutil.copy(source_subject_sleep_folder.joinpath(file),
                            target_subject_sleep_folder.joinpath(file))

            ### Steps files ###
            # The validation of steps dates are in process_and_aggregate_steps function.
            steps_file_name_pattern = re.compile(r'steps-\d{4}-\d{2}-\d{2}.json$')
            steps_json_files = [file_name for file_name in os.listdir(source_subject_physical_activity_folder)
                                if steps_file_name_pattern.search(file_name)]
            # Copy steps files
            for file in steps_json_files:
                shutil.copy(source_subject_physical_activity_folder.joinpath(file),
                            target_subject_physical_activity_folder.joinpath(file))

        except Exception as e:
            missing_files_list.append({'subject': subject_row['Id'], 'missing_file': 'Couldn\'t read its files from the source folder.'})

        try:
            if isinstance(now, pd.Timestamp):
                now = now.strftime('%Y-%m-%d %H-%M-%S')
            
            # Replace underscores with spaces
            now = now.replace("_", " ")
            
            # Replace hyphens in the time part with colons
            now = now.replace("-", ":")
            
            # Convert to datetime
            now = pd.to_datetime(now)

        except Exception as e:
            print(e)
            time.sleep(20)

        exeHistoryUpdate = pl.DataFrame({
            "Project": [project],
            "Subject": [subject_row['Id']],
            "User": [username],
            'Page': ['Copy Files'],
            "Datetime": [now],
            "Action": ['Copy Files'],
        })
        exeHistory = pl.concat([exeHistory, exeHistoryUpdate])


    exeHistory.write_parquet(exeHistory_path)

    for subject in os.listdir(DATA_PATH):
        subject_output_folder = OUTPUT_PATH.joinpath(subject)
        if not subject_output_folder.exists():
            os.makedirs(subject_output_folder)
    # except Exception as e:
    #     print(e)
    #     time.sleep(20)






if __name__ == '__main__':
    
    try:
        param = sys.argv[1]
        now = sys.argv[2]
        print(now)
        user_name = sys.argv[3]
    except IndexError:
        param = 'NOVA_TESTS'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        # now = '2024-08-26_13:48:25'
        user_name = 'Unknown'

    main(param, now, user_name)