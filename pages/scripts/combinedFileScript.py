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

        print('Starting script')
        exeHistory_path = Path(r'.\pages\ExecutionHis\exeHistory.parquet')

        exeHistory = pl.read_parquet(exeHistory_path)
        print('Reading exeHistory')
        paths_json = json.load(open(r".\pages\Pconfigs\paths.json", "r"))

        project_path = Path(paths_json[project])



        DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

        AGGREGATED_OUTPUT_PATH_HISTORY = ut.output_record(OUTPUT_PATH, 'Aggregated Output',username, now)

        if not AGGREGATED_OUTPUT_PATH_HISTORY.exists():
            os.makedirs(AGGREGATED_OUTPUT_PATH_HISTORY)
        
        PROJECT_CONFIG = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))

        SUBJECTS_DATES = METADATA_PATH.joinpath('Subjects Dates.csv')


        subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                    try_parse_dates=True)

        print('Subjects Dates readed')

        subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')

        selected_subjects_path = Path(rf'.\pages\sub_selection\{project}_sub_selection_refresh_Combined.parquet')


        subjects_to_run_on = []
        refresh = False

        if not selected_subjects_path.exists():
            subjects_to_run_on = subjects_dates_df['Id'].to_list()
            refresh = False
        else:
            selected_subjects = pl.read_parquet(selected_subjects_path)

            if selected_subjects.shape[0] == 0:
                subjects_to_run_on = subjects_dates_df['Id'].to_list()
                refresh = False
            else:
                subjects_to_run_on = selected_subjects['Subject'].to_list()
                refresh = True

        run_on_specific_subjects = True

        

        sleep_all_subjects_path = OUTPUT_PATH.joinpath('Aggregated Output','Sleep All Subjects.csv')
        steps_all_subjects_path = OUTPUT_PATH.joinpath('Aggregated Output', 'Steps Aggregated.csv')

        if not sleep_all_subjects_path.exists():
            print(f'Sleep All Subjects file does not exist for {project}')
            time.sleep(10)
            quit()

        if not steps_all_subjects_path.exists():
            print(f'Steps All Subjects file does not exist for {project}')
            time.sleep(10)
            quit()


        sleep_all_subjects = pl.read_csv(sleep_all_subjects_path, try_parse_dates=True)
        steps_all_subjects = pl.read_csv(steps_all_subjects_path, try_parse_dates=True)

        missing_values_df = pl.DataFrame()


        subjects_with_missing_heart_rate_files = []
        print('Upload and basic files stats')
        tqdm_subjects = tqdm(os.listdir(DATA_PATH))
        for subject in tqdm_subjects:
            if not re.search(r'\d{3}$', subject):
                continue
            if run_on_specific_subjects and subject not in subjects_to_run_on:
                continue
            tqdm_subjects.set_description(f'Subject: {subject}', refresh=True)

            if not os.path.exists(OUTPUT_PATH.joinpath(subject)):
                os.mkdir(OUTPUT_PATH.joinpath(subject))

            subjects_dates = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)

            experiment_start_datetime = subjects_dates['ExperimentStartDateTime'].values[0]
            experiment_end_datetime = subjects_dates['ExperimentEndDateTime'].values[0]

            experiment_start_date = subjects_dates['ExperimentStartDate'].values[0]
            experiment_end_date = subjects_dates['ExperimentEndDate'].values[0]

            experiment_date_range = (
                pl.DataFrame({
                    'FullDate': pl.date_range(experiment_start_date, experiment_end_date, interval='1d', eager=True)
                    })
            )

            experiment_datetimes_range = (
                pl.DataFrame({
                    'FullDateTime': pl.datetime_range(experiment_start_datetime, experiment_end_datetime, interval='1m', eager=True, time_unit='us')
                })
            )

            subject_path = OUTPUT_PATH.joinpath(subject)

            if not subject_path.exists():
                print(f'No output folder for {subject}')
                continue
            mean_missing_percent = np.nan
            for file in subject_path.iterdir():
                
                if file.name.endswith('Heart Rate.csv'):
                    hr_path = file
                    hr_file = pl.read_csv(hr_path, try_parse_dates=True)

                    hr_file = (
                        hr_file
                        .join(
                            experiment_datetimes_range,
                            left_on='DateTime',
                            right_on='FullDateTime',
                            how='outer'
                        )
                        .sort('DateTime')
                        .with_columns(
                            pl.col('DateTime').fill_null(pl.col('FullDateTime'))
                        )
                        .drop('FullDateTime')
                    )

                    df = (
                        hr_file
                        .with_columns(
                            pl.col('BpmMean').is_null().alias('gap'),
                        )
                    )
                    
                    df = (
                        df
                        .with_columns(
                            (pl.col('gap') != pl.col('gap').shift(1)).cum_sum().alias('gap_group')
                        )
                    )

                    gap_sizes = (
                        df
                        .filter(
                            pl.col('gap')
                        )
                        .group_by('gap_group')
                        .agg(
                            pl.count().name.suffix('_count')
                        )
                    )

                    gap_sizes_counts = (
                        gap_sizes
                        .group_by('len_count')
                        .agg(
                            pl.len()
                        )
                        .sort('len')
                        .with_columns(
                            pl.col('len_count').cast(pl.String)
                        )
                    )

                    df = (
                        df
                        .with_columns(
                            pl.col('DateTime').dt.truncate('1d').alias('Date')
                        )
                    )

                    missing_per_day = (
                        df
                        .group_by('Date')
                        .agg(
                            missing_percent = pl.col('BpmMean').is_null().mean() * 100
                        )
                    )

                    mean_missing_percent = (
                        missing_per_day
                        .select(
                            pl.col('missing_percent').mean()
                            )
                            .item()
                    )
                    break
                else:
                    mean_missing_percent = np.nan
                    
            
            sleep_subject = (
                sleep_all_subjects
                .filter(
                    pl.col('Id') == subject
                )
            )

            if sleep_subject.shape[0] == 0:
                mean_sleep_missing_percent = np.nan 
            else:
                sleep_subject = (
                    sleep_subject
                    .join(
                        experiment_date_range,
                        left_on='ExperimentDates',
                        right_on='FullDate',
                        how='outer'
                    )
                    .sort('ExperimentDates')
                    .with_columns(
                        pl.col('ExperimentDates').fill_null(pl.col('FullDate'))
                    )
                    .drop('FullDate')

                )

                sleep_df = (
                    sleep_subject
                    .with_columns(
                        (pl.col('SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)').is_null()).alias('gap')
                    )
                )

                sleep_df = (
                    sleep_df
                    .with_columns(
                        (pl.col('gap') != pl.col('gap').shift(1)).cum_sum().alias('gap_group')
                    )
                )

                gap_sizes = (
                    sleep_df
                    .filter(
                        pl.col('gap')
                    )
                    .group_by('gap_group')
                    .agg(
                        pl.count().name.suffix('_count')
                    )
                )

                gap_sizes_counts = (
                    gap_sizes
                    .group_by('len_count')
                    .agg(
                        pl.len()
                    )
                    .sort('len')
                    .with_columns(
                        pl.col('len_count').cast(pl.String)
                    )
                )

                sleep_df = (
                    sleep_df
                    .with_columns(
                        pl.col('ExperimentDates').dt.truncate('1d').alias('ExperimentDates')
                    )
                )

                missing_per_day = (
                    sleep_df
                    .group_by('ExperimentDates')
                    .agg(
                        missing_percent = pl.col('SleepTimeInMinutes (our Calculation: SleepEndTime-SleepStartTime)').is_null().mean() * 100
                        )
                )

                mean_sleep_missing_percent = (
                    missing_per_day
                    .select(pl.col('missing_percent').mean())
                    .item()
                )

            steps_subject = (
                steps_all_subjects
                .filter(
                    pl.col('Id') == subject
                )
            )

            if steps_subject.shape[0] == 0:
                mean_steps_missing_percent = np.nan
            else:
                steps_subject = (
                    steps_subject
                    .join(
                        experiment_datetimes_range,
                        left_on='DateAndMinute',
                        right_on='FullDateTime',
                        how='outer'
                    )
                    .sort('DateAndMinute')
                    .with_columns(
                        pl.col('DateAndMinute').fill_null(pl.col('FullDateTime'))
                    )
                    .drop('FullDateTime')
                )

                steps_df = (
                    steps_subject
                    .with_columns(
                        pl.col('StepsInMinute').is_null().alias('gap')
                    )
                )

                steps_df = (
                    steps_df
                    .with_columns(
                        gap_group = pl.col('gap') != pl.col('gap').shift(1).cum_sum()
                    )
                )

                gap_sizes = (
                    steps_df
                    .filter(
                        pl.col('gap')
                    )
                    .group_by('gap_group')
                    .agg(
                        pl.count().name.suffix('_count')
                    )
                )

                gap_sizes_counts = (
                    gap_sizes
                    .group_by('len_count')
                    .agg(
                        pl.len()
                    )
                    .sort('len')
                    .with_columns(
                        pl.col('len_count').cast(pl.String)
                    )
                )

                steps_df = (
                    steps_df
                    .with_columns(
                        pl.col('DateAndMinute').dt.truncate('1d').alias('Date')
                    )
                )

                missing_per_day = (
                    steps_df
                    .group_by('Date')
                    .agg(missing_percent = pl.col('StepsInMinute').is_null().sum() * 100)
                )

                mean_steps_missing_percent = (
                    missing_per_day
                    .select(pl.col('missing_percent').mean())
                    .item()
                )

            missing_values_df = (
                pl.concat(
                    [
                        missing_values_df,
                        pl.DataFrame({
                            'Subject': [subject],
                            'Mean Missing HR %': [mean_missing_percent],
                            'Mean Missing Sleep %': [mean_sleep_missing_percent],
                            'Mean Missing Steps %': [mean_steps_missing_percent],
                            'Last Updated': [datetime.datetime.now()]
                        })
                    ]
                )
            )

        
        missing_values_df = missing_values_df.sort(['Subject', 'Last Updated']).unique('Subject', keep='last')
        try:
            combined_subjects_path = rf'.\pages\sub_selection\{project}_sub_selection_Combined.parquet'
        except:
            combined_subjects_path = rf'.\pages\sub_selection\{project}_sub_selection_Combined.parquet'
            

        if not os.path.exists(combined_subjects_path):
            missing_values_df.write_parquet(combined_subjects_path)
        else:
                
            combined_subjects = (
                pl.read_parquet(combined_subjects_path)
                .filter(
                    ~pl.col('Subject').is_in(missing_values_df['Subject'])
                )
            )

            combined_subjects = pl.concat([combined_subjects, missing_values_df])

            combined_subjects.write_parquet(combined_subjects_path)





                    


                    


                

                



            
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
        print(param)
        print(f'Starting initiate with system parameters')
    except IndexError:
        param = 'FIBRO_TESTS'
        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        user_name = 'Unknown'

    main(param, now, user_name)
    time.sleep(15)