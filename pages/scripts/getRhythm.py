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
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.optimize import curve_fit
from CosinorPy import file_parser, cosinor, cosinor1

import dash_ag_grid as dag
import dash
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




try:
    def main(project, now, username, not_in_il, dst_change, window_size, incriment_size, downsample_rate, missing_data_thr, data_interpolation, signal):
        
        FIRST = 0
        LAST = -1
        exeHistory_path = Path(r'.\pages\ExecutionHis\exeHistory.parquet')   

        exeHistory = pl.read_parquet(exeHistory_path)
        paths_json = json.load(open(r".\pages\Pconfigs\paths.json", "r"))

        project_path = Path(paths_json[project])



        # DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)
        DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables_custom(project_path, '_NEW_CODE')

        AGGREGATED_OUTPUT_PATH_HISTORY = ut.output_record(OUTPUT_PATH, 'Aggregated Output',username, now)

        if not AGGREGATED_OUTPUT_PATH_HISTORY.exists():
            os.makedirs(AGGREGATED_OUTPUT_PATH_HISTORY)
        
        PROJECT_CONFIG = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))    
        SUBJECTS_DATES = METADATA_PATH.joinpath('Subjects Dates.csv')


        subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                    try_parse_dates=True)



        subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')
        selected_subjects_path = Path(rf'.\pages\sub_selection\{project}_sub_selection_folders_rhythmic.parquet')
            
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
        print('\nProcess cosinor analysis')
        

        all_subjects_cosinor = pl.DataFrame(strict=False)
        all_subjects_cosinor_preprocessed_data = pl.DataFrame(strict=False)

        all_subjects_visu = pl.DataFrame(strict=False)
        all_subjects_estimates = pl.DataFrame(strict=False)

        all_subjects_json = {}

        # Find relevant HRV files and respiratory files
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

                    
            # Get start and end dates of experiment for current subject
            subject_dates = ut.get_subject_dates_of_experiment(subject, METADATA_PATH)
            # create a DatetimeIndex with the range of experiment dates
            subject_experiment_dates = pd.date_range(subject_dates['ExperimentStartDate'].values[0],
                                                    subject_dates['ExperimentEndDate'].values[0])
            experiment_start_date = subject_experiment_dates[0]

            # Convert the DatetimeIndex (subject_experiment_dates) to a DataFrame with a column named 'ExperimentStartDate'
            subject_experiment_dates_df = pd.DataFrame({'ExperimentDates': subject_experiment_dates})

            aggregated_file = ut.new_get_latest_file_by_term('Heart Rate and Steps and Sleep Aggregated', subject, root=OUTPUT_PATH)
            if aggregated_file.exists():
                aggregated_df = pd.read_csv(aggregated_file)
            else:
                print(f'No aggregated file for {subject}')
                continue

            df = (
                pl.DataFrame(aggregated_df)
                .select([
                    "DateAndMinute",
                    signal,
                    "not_in_israel",
                    "is_dst_change",
                ])
                .with_columns(
                    pl.col("DateAndMinute").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
                )
            )


            if not not_in_il:
                df = df.filter(pl.col("not_in_israel") == False)

            if not dst_change:
                df = df.filter(pl.col("is_dst_change") == False)

            if df.shape[0] == 0:
                print(f'No data for {subject}')
                continue

            start_datetime = df.select(pl.col("DateAndMinute").min()).item()
            end_datetime = df.select(pl.col("DateAndMinute").max()).item()
            if data_interpolation:
                df = decompose_and_interpolate(df, window_size, signal)
            else:
                df = df

            windows = generate_fixed_windows(start_datetime, end_datetime, window_size, incriment_size)


            preprocessed_data = []

            for window in windows:
                window_start = window['start']
                window_end = window['end']
                window_id = window['window_id']
                window_label = window['label']

                data_in_window = df.filter(
                    (pl.col("DateAndMinute") >= window_start),
                    (pl.col("DateAndMinute") <= window_end)
                )

                total_points = data_in_window.shape[0]
                missing_points = data_in_window.filter(pl.col(signal).is_null()).shape[0]
                missing_percentage = (missing_points / total_points) * 100 if total_points > 0 else 100

                if missing_percentage > missing_data_thr:
                    continue

                null_sequences = (
                    data_in_window
                    .select(
                        pl.when(pl.col(signal).is_null())
                        .then(1)
                        .otherwise(0)
                        .alias("is_null")
                    )
                    .with_columns(
                        group_id = pl.col("is_null") != pl.col("is_null").shift(1).cum_sum()
                    )
                    .filter(pl.col("is_null") == 1)
                )

                null_sequences=(
                    null_sequences
                    .group_by("group_id")
                    .agg(pl.col('is_null').count().alias('null_length'))
                )

                max_null_sequence = null_sequences.select(pl.col('null_length').max()).item() if not null_sequences.is_empty() else 0
                total_nulls = null_sequences.select(pl.col('null_length').sum()).item() if not null_sequences.is_empty() else 0



                downsampled_data = downsample_signal(data_in_window, downsample_rate, signal)

                window_df = pl.DataFrame({
                    "test": [window_label] * downsampled_data.shape[0],
                    "x": np.arange(downsampled_data.shape[0]),
                    "y": downsampled_data["downsampled"],
                    "interpolated_y": downsampled_data["interpolated_mean"] if 'interpolated_mean' in downsampled_data.columns else None,
                    "max_null_sequence": [max_null_sequence] * downsampled_data.shape[0],
                    "total_nulls": [total_nulls] * downsampled_data.shape[0],
                })

                preprocessed_data.append(window_df)

            if len(preprocessed_data) == 0:
                print(f'No valid windows for {subject}')
                continue

            period = window_size * 60 / downsample_rate

            result, data = cosinor_analysis(preprocessed_data, signal, period)

            final_sub_df, visu_sub_df, json_visu_sub, estimates_sub_df = generate_final_df(result, preprocessed_data, downsample_rate, period, window_size, subject, signal, experiment_start_date, incriment_size, end_datetime)

            subject_output_path = OUTPUT_PATH.joinpath(subject)
            if not subject_output_path.exists():
                os.makedirs(subject_output_path)

            subject_output_path_history = ut.output_record(OUTPUT_PATH, subject, username, now)

            if not subject_output_path_history.exists():
                subject_output_path_history.mkdir(parents=True)
            
            final_sub_df.write_csv(subject_output_path_history.joinpath(f'{subject} cosinor.csv'))
            
            if len(preprocessed_data) != 0:
                pl.DataFrame(data).write_parquet(subject_output_path_history.joinpath(f'{subject} cosinor preprocessed data ({signal}).parquet'))



            ut.check_for_duplications(subject_output_path, subject_output_path_history)


            if all_subjects_cosinor_preprocessed_data.is_empty():
                all_subjects_cosinor_preprocessed_data = pl.DataFrame(data, strict=False)
            
            else:
                all_subjects_cosinor_preprocessed_data = pl.concat([all_subjects_cosinor_preprocessed_data, pl.DataFrame(data, strict=False)], how='vertical_relaxed')


            if all_subjects_estimates.is_empty():
                all_subjects_estimates = pl.DataFrame(estimates_sub_df, strict=False)

            else:
                all_subjects_estimates = pl.concat([all_subjects_estimates, estimates_sub_df], how='vertical_relaxed')


            if all_subjects_visu.is_empty():
                all_subjects_visu = pl.DataFrame(visu_sub_df, strict=False)

            else:
                all_subjects_visu = pl.concat([all_subjects_visu, visu_sub_df], how='vertical_relaxed')

            if all_subjects_cosinor.is_empty():
                all_subjects_cosinor = pl.DataFrame(final_sub_df, strict=False)

            else:
                all_subjects_cosinor = pl.concat([all_subjects_cosinor, final_sub_df], how='vertical_relaxed')

            all_subjects_json[subject] = json_visu_sub


        all_subjects_cosinor_path_hist = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'all_subjects_cosinor_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.csv')
        all_subjects_cosinor_path = OUTPUT_PATH.joinpath('Aggregated Output').joinpath(f'all_subjects_cosinor_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.csv')
        
        all_subjects_visu_path_hist = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'all_subjects_visu_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.parquet')
        all_subjects_visu_path = OUTPUT_PATH.joinpath('Aggregated Output').joinpath(f'all_subjects_visu_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.parquet')
        
        all_subjects_json_path_hist = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'all_subjects_json_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.json')
        all_subjects_json_path = OUTPUT_PATH.joinpath('Aggregated Output').joinpath(f'all_subjects_json_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.json')

        all_subjects_estimates_path_hist = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'all_subjects_estimates_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.parquet')
        all_subjects_estimates_path = OUTPUT_PATH.joinpath('Aggregated Output').joinpath(f'all_subjects_estimates_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.parquet')

        if not all_subjects_cosinor_path.exists():
            all_subjects_cosinor.sort(['Id', 'ExperimentDay'])
            all_subjects_cosinor.write_csv(all_subjects_cosinor_path_hist)
            all_subjects_visu.sort(['Id', 'test'])
            all_subjects_visu.write_parquet(all_subjects_visu_path_hist)
            all_subjects_estimates.sort(['Id', 'test'])
            all_subjects_estimates.write_parquet(all_subjects_estimates_path_hist)
            with open(all_subjects_json_path_hist, 'w') as f:
                json.dump(all_subjects_json, f, indent=4)

        else:
            old_all_subjects_cosinor = pl.read_csv(all_subjects_cosinor_path)
            old_all_subjects_cosinor = (
                old_all_subjects_cosinor
                .filter(
                    ~pl.col("Id").is_in(all_subjects_cosinor['Id'])
                )
            )

            all_subjects_cosinor = all_subjects_cosinor.sort(['Id', 'ExperimentDay'])

            all_subjects_cosinor = pl.concat([old_all_subjects_cosinor, all_subjects_cosinor], how='vertical_relaxed')
            all_subjects_cosinor.sort(['Id', 'ExperimentDay'])

            all_subjects_cosinor.write_csv(all_subjects_cosinor_path_hist)

            old_all_subjects_visu = pl.read_parquet(all_subjects_visu_path)
            old_all_subjects_visu = (
                old_all_subjects_visu
                .filter(
                    ~pl.col("Id").is_in(all_subjects_visu['Id'])
                )
            )

            all_subjects_visu = pl.concat([old_all_subjects_visu, all_subjects_visu], how='vertical_relaxed')
            all_subjects_visu.sort(['Id', 'test'])

            all_subjects_visu.write_parquet(all_subjects_visu_path_hist)

            old_all_subjects_json = json.load(open(all_subjects_json_path))
            old_all_subjects_json.update(all_subjects_json)

            with open(all_subjects_json_path_hist, 'w') as f:
                json.dump(old_all_subjects_json, f, indent=4)

            old_all_subjects_estimates = pl.read_parquet(all_subjects_estimates_path)
            old_all_subjects_estimates = (
                old_all_subjects_estimates
                .filter(
                    ~pl.col("Id").is_in(all_subjects_estimates['Id'])
                )
            )

            all_subjects_estimates = pl.concat([old_all_subjects_estimates, all_subjects_estimates], how='vertical_relaxed')
            all_subjects_estimates.sort(['Id', 'test'])

            all_subjects_estimates.write_parquet(all_subjects_estimates_path_hist)




        all_subjects_preprocessed_data_path_hist = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath(f'all_subjects_cosinor_preprocessed_data_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.parquet')
        all_subjects_preprocessed_data_path = OUTPUT_PATH.joinpath('Aggregated Output').joinpath(f'all_subjects_cosinor_preprocessed_data_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.parquet')

        if not all_subjects_preprocessed_data_path.exists():
            all_subjects_cosinor_preprocessed_data.write_parquet(all_subjects_preprocessed_data_path_hist)
        else:
            old_all_subjects_cosinor_preprocessed_data = pl.read_parquet(all_subjects_preprocessed_data_path)
            old_all_subjects_cosinor_preprocessed_data = (
                old_all_subjects_cosinor_preprocessed_data
                .filter(
                    ~pl.col("Id").is_in(all_subjects_cosinor_preprocessed_data['Id'])
                )
            )

            all_subjects_cosinor_preprocessed_data = pl.concat([old_all_subjects_cosinor_preprocessed_data, all_subjects_cosinor_preprocessed_data], how='vertical_relaxed')
            all_subjects_cosinor_preprocessed_data.write_parquet(all_subjects_preprocessed_data_path_hist)




        all_subjects_summary = []
        all_subjects_agg = pd.DataFrame()

        print('\n Aggregating results')



        all_subjects_cosinor_agg = (
            all_subjects_cosinor
            .select([
                'Id', 'signal', 'windowSize (minutes)', 
                'incrementSize (minutes)', 'downsampleRate (minutes)',
                'interpolation', 'samples', 'nMissing', 'max_null_sequence (before downsample)', 'nMissing  (before downsample)',
                'period', 'AIC', 'BIC', 'r_squared', 'r_squared_adj', 'mesor',
                'amplitude', 'acrophase (rad)', 'acrophase from midnight (minutes)',
                'acrophase (time)'
            ])
        )

        all_subjects_cosinor_agg = (
            all_subjects_cosinor_agg
            .with_columns(
                pl.col("acrophase (time)").map_elements(lambda x: datetime.datetime.strptime(x, "%H:%M").hour *60 + datetime.datetime.strptime(x, "%H:%M").minute).alias("acrophase_minutes")
            )
            .group_by('Id')
            .agg([
                pl.col('Id').count().alias('nDays'),
                pl.col('signal').first().alias('signal'),
                pl.col('windowSize (minutes)').first().alias('windowSize (minutes)'),
                pl.col('incrementSize (minutes)').first().alias('incrementSize (minutes)'),
                pl.col('downsampleRate (minutes)').first().alias('downsampleRate (minutes)'),
                pl.col('interpolation').first().alias('interpolation'),
                pl.col('samples').mean().alias('samples (mean)'),
                pl.col('samples').min().alias('samples (min)'),
                pl.col('samples').max().alias('samples (max)'),
                pl.col('nMissing  (before downsample)').mean().alias('nMissing (before downsample) (mean)'),
                pl.col('nMissing  (before downsample)').min().alias('nMissing (before downsample) (min)'),
                pl.col('nMissing  (before downsample)').max().alias('nMissing (before downsample) (max)'),
                pl.col('max_null_sequence (before downsample)').mean().alias('max_null_sequence (before downsample) (mean)'),
                pl.col('max_null_sequence (before downsample)').min().alias('max_null_sequence (before downsample) (min)'),
                pl.col('max_null_sequence (before downsample)').max().alias('max_null_sequence (before downsample) (max)'),
                pl.col('period').first().alias('period'),
                pl.col('AIC').mean().alias('AIC (mean)'),
                pl.col('AIC').std().alias('AIC (std)'),
                pl.col('AIC').median().alias('AIC (median)'),
                pl.col('AIC').min().alias('AIC (min)'),
                pl.col('AIC').max().alias('AIC (max)'),
                pl.col('BIC').mean().alias('BIC (mean)'),
                pl.col('BIC').std().alias('BIC (std)'),
                pl.col('BIC').median().alias('BIC (median)'),
                pl.col('BIC').min().alias('BIC (min)'),
                pl.col('BIC').max().alias('BIC (max)'),
                pl.col('r_squared').mean().alias('r_squared (mean)'),
                pl.col('r_squared').std().alias('r_squared (std)'),
                pl.col('r_squared').median().alias('r_squared (median)'),
                pl.col('r_squared').min().alias('r_squared (min)'),
                pl.col('r_squared').max().alias('r_squared (max)'),
                pl.col('r_squared_adj').mean().alias('r_squared_adj (mean)'),
                pl.col('r_squared_adj').std().alias('r_squared_adj (std)'),
                pl.col('r_squared_adj').median().alias('r_squared_adj (median)'),
                pl.col('r_squared_adj').min().alias('r_squared_adj (min)'),
                pl.col('r_squared_adj').max().alias('r_squared_adj (max)'),
                pl.col('mesor').mean().alias('mesor (mean)'),
                pl.col('mesor').std().alias('mesor (std)'),
                pl.col('mesor').median().alias('mesor (median)'),
                pl.col('mesor').min().alias('mesor (min)'),
                pl.col('mesor').max().alias('mesor (max)'),
                pl.col('amplitude').mean().alias('amplitude (mean)'),
                pl.col('amplitude').std().alias('amplitude (std)'),
                pl.col('amplitude').median().alias('amplitude (median)'),
                pl.col('amplitude').min().alias('amplitude (min)'),
                pl.col('amplitude').max().alias('amplitude (max)'),
                pl.col('acrophase (rad)').mean().alias('acrophase (rad) (mean)'),
                pl.col('acrophase (rad)').std().alias('acrophase (rad) (std)'),
                pl.col('acrophase (rad)').median().alias('acrophase (rad) (median)'),
                pl.col('acrophase (rad)').min().alias('acrophase (rad) (min)'),
                pl.col('acrophase (rad)').max().alias('acrophase (rad) (max)'),
                pl.col('acrophase from midnight (minutes)').mean().alias('acrophase from midnight (minutes) (mean)'),
                pl.col('acrophase from midnight (minutes)').std().alias('acrophase from midnight (minutes) (std)'),
                pl.col('acrophase from midnight (minutes)').median().alias('acrophase from midnight (minutes) (median)'),
                pl.col('acrophase from midnight (minutes)').min().alias('acrophase from midnight (minutes) (min)'),
                pl.col('acrophase from midnight (minutes)').max().alias('acrophase from midnight (minutes) (max)'),
                pl.col('acrophase_minutes').mean().alias('acrophase (time) (mean in minutes)'),
                pl.col('acrophase_minutes').std().alias('acrophase (time) (std in minutes)'),
                pl.col('acrophase_minutes').median().alias('acrophase (time) (median in minutes)'),
                pl.col('acrophase_minutes').min().alias('acrophase (time) (min in minutes)'),
                pl.col('acrophase_minutes').max().alias('acrophase (time) (max in minutes)'),
            ])
        )

        all_subjects_cosinor_agg = (
            all_subjects_cosinor_agg
            .with_columns(
                # Convert mean acrophase_minutes back to HH:MM format
                pl.col("acrophase (time) (mean in minutes)")
                .map_elements(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")
                .alias("acrophase (time) (mean)"),

                pl.col("acrophase (time) (median in minutes)")
                .map_elements(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")
                .alias("acrophase (time) (median)"),

                # Convert min acrophase_minutes back to HH:MM format
                pl.col("acrophase (time) (min in minutes)")
                .map_elements(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")
                .alias("acrophase (time) (min)"),

                # Convert max acrophase_minutes back to HH:MM format
                pl.col("acrophase (time) (max in minutes)")
                .map_elements(lambda x: f"{int(x // 60):02}:{int(x % 60):02}")
                .alias("acrophase (time) (max)"),

                # Convert std deviation to HH:MM (if desired; otherwise, keep it numeric)
                pl.col("acrophase (time) (std in minutes)")
                .map_elements(lambda x: f"{int(x // 60):02}:{int(x % 60):02}" if not pd.isnull(x) else None)
                .alias("acrophase (time) (std)"),
            )
        )


        all_subjects_cosinor_agg_path = OUTPUT_PATH.joinpath('Aggregated Output').joinpath(f'all_subjects_cosinor_agg_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.csv')

        if not all_subjects_cosinor_agg_path.exists():
            all_subjects_cosinor_agg.write_csv(all_subjects_cosinor_agg_path)
        else:
            old_all_subjects_cosinor_agg = pl.read_csv(all_subjects_cosinor_agg_path)
            old_all_subjects_cosinor_agg = (
                old_all_subjects_cosinor_agg
                .filter(
                    ~pl.col("Id").is_in(all_subjects_cosinor_agg['Id'])
                )
            )

            all_subjects_cosinor_agg = pl.concat([old_all_subjects_cosinor_agg, all_subjects_cosinor_agg], how='vertical_relaxed')
            all_subjects_cosinor_agg_path = AGGREGATED_OUTPUT_PATH_HISTORY.joinpath('Aggregated Output').joinpath(f'all_subjects_cosinor_agg_w{window_size}_incr{incriment_size}_ds{downsample_rate}_missingThr{missing_data_thr}_interpolation_{intepolation}.csv')
            all_subjects_cosinor_agg.write_csv(all_subjects_cosinor_agg_path)

        ut.check_for_duplications(AGGREGATED_OUTPUT_PATH, AGGREGATED_OUTPUT_PATH_HISTORY)

        exeHistoryUpdate = exeHistory.with_columns(
            pl.DataFrame({
                'Project': [project],
                'Username': [username],
                'Date': [now],
                'Window Size': [window_size],
                'Increment Size': [incriment_size],
                'Downsample Rate': [downsample_rate],
                'Missing Data Threshold': [missing_data_thr],
                'Interpolation': [data_interpolation],
                'Signal': [signal],
                'File': [str(all_subjects_cosinor_agg_path)]
            })

        )

        exeHistory = pl.concat([exeHistory, exeHistoryUpdate], how='diagonal_relaxed')

        logging.info(f'File generation completed')






    def generate_final_df(results, original_data, window_size, period, select_period_size, subject, signal, experiment_start_date, increment_size, end_datetime):

        original_data = pl.concat(original_data, how='vertical_relaxed').to_pandas()
        
        columns = ['Id',  'ExperimentDay', 'signal', 'start_date', 'week_day', 'windowSize (minutes)', 
                    'incrementSize (minutes)', 'downsampleRate (minutes)', 'interpolation', 'samples', 'nMissing', 
                    'max_null_sequence (before downsample)', 'nMissing  (before downsample)', 'period', 'AIC', 'BIC',
                    'r_squared', 'r_squared_adj', 'mesor', 'amplitude', 'acrophase (rad)', 'acrophase from midnight (minutes)',
                    'acrophase (time)', 'acrophase (datetime)', 'F-value', 'F-pvalue', 'df residual ', 'df model', 'amplitude high CI',
                    'amplitude low CI', 'acrophase high CI', 'acrophase low CI', 'mesor high CI', 'mesor low CI',
                    'p amplitude', 'p acrophase', 'p mesor']
                #    'date', 'amplitude','period','acrophase (rad)', 'corrected_acrophase (rad)',
                #     'corrected_acrophase (hours)', 'corrected_acrophase (datetime)', 'corrected_acrophase (degrees)', 
                #     'mesor','AIC', 'BIC','peaks','heights', 'troughs', 'trough_time', 'trough_datetime', 'y_estimated_min_loc', 'length', 
                #     'heights2','max_loc', 'period2', 'p-value', 'p_reject', 'SNR', 'RSS', 
                #     'resid_SE', 'ME','f-pvalue', 't-values const', 't-values x1',
                #     't-values x2','R-squared', 'R-squared adjusted', 'SSR', 'minutes_based']

        visu_colums = ['Id', 'test', 'x', 'y', 'interpolated_y']

        estimated_columns = ['Id', 'test', 'estimated_x', 'estimated_y']


        # Initialize an empty DataFrame with these columns
        results_df = pl.DataFrame({col: [] for col in columns}, strict=False)
        visu_dfs = pl.DataFrame({col: [] for col in visu_colums}, strict=False)
        estimated_dfs = pl.DataFrame({col: [] for col in estimated_columns}, strict=False)

        json_results = {}
        


        for key in results.keys():
            if results[key] is None:
                continue
            
            current_win_start = key.split(" ")[0]
            current_win_end = key.split(" ")[-2]
            incriment_percent = increment_size / select_period_size

            max_null_sequence = original_data[original_data['test'] == key]
            max_null_sequence = max_null_sequence['max_null_sequence'].max()

            total_nulls = original_data[original_data['test'] == key]
            total_nulls = total_nulls['total_nulls'].max()

            model = results[key]
            cosinor_model = model[0]
            stats = model[1]
            params = model[2]
            estimated_x = model[3]
            estimated_y = model[4]
            length = len(estimated_y)
            original_data1 = original_data[original_data['test'] == key]
            
            peak_indices = params['peaks'] if len(params['peaks']) > 0 else [np.nan]
            theta = peak_indices[0]/params['period'] * 2 * np.pi
            corrected_acrophase_deg = quadrant_adjustment(theta, params['acrophase'], radian=False)
            





            corrected_acrophase = np.deg2rad(corrected_acrophase_deg)

            trough_indices = params['troughs'][0] if len(params['troughs']) > 0 else np.nan
            
            trough_loc = trough_indices/params['period'] * period
        
            trough_hours = int(trough_loc) if not np.isnan(trough_loc) else np.nan
            

            trough_minutes = int((trough_loc - trough_hours) * 60) if not np.isnan(trough_loc) else np.nan

            y_estimated_min_loc = np.argmin(estimated_y)

            if not np.isnan(trough_loc):
                if select_period_size > 24:
                    trough_days = int((y_estimated_min_loc / len(estimated_y)) * (period* window_size)) // 1440 
                    trough_hours = trough_hours + increment_size
                    trough_hours = trough_hours % 24
                    trough_time = f"{trough_days} day(s) {trough_hours:02d}:{trough_minutes:02d}"

                else:
                    trough_hours = trough_hours + increment_size
                    trough_hours = trough_hours % 24
                    trough_time = f"{trough_hours:02d}:{trough_minutes:02d}"
            else:
                trough_time = np.nan

            # convert the corrected acrophase degrees to time in format HH:MM
            hours = int((corrected_acrophase_deg/360) * select_period_size)

            # if half_day:
            #     hours = hours + 12

            if select_period_size > 24:
                days = np.abs(hours // 24)
                if incriment_percent != 1.0:
                    hours = np.abs(hours) + increment_size
                    hours = hours % 24
                else:
                    hours = np.abs(hours) % 24
                minutes = np.abs(int((corrected_acrophase_deg/360) * 24 * 60) % 60)
                corrected_acrophase_time = f"{np.abs(days)} day(s) {np.abs(hours):02d}:{np.abs(minutes):02d}"
                accrophs_minutes_from_midnight = hours * 60 + minutes
            else:
                minutes = np.abs(int((corrected_acrophase_deg/360) * 24 * 60) % 60)
                if incriment_percent != 1.0:
                    hours = np.abs(hours) + increment_size
                else:
                    hours = np.abs(hours)
                corrected_acrophase_time = f"{np.abs(hours):02d}:{np.abs(minutes):02d}"
                accrophs_minutes_from_midnight = hours * 60 + minutes
            # minutes = int((corrected_acrophase_deg/360) * 24 * 60) % 60
            # corrected_acrophase_time = f"{hours:02d}:{minutes:02d}"
            accrophs_minutes_from_midnight = hours * 60 + minutes
            window_start_date = key.split(" ")[0]
            # window_start_date_trough = trough_time.split(" ")[0]
            window_end_date = key.split(" ")[-2]

            window_start_time = key.split(" ")[1]
            window_end_time = key.split(" ")[-1]

            if not np.isnan(trough_loc):
                if select_period_size > 24:
                    days = int(corrected_acrophase_time.split(' ')[0])
                    trough_days = int((y_estimated_min_loc / len(estimated_y)) * (period* window_size)) // 1440 
                    window_start_date = datetime.datetime.strptime(window_start_date, "%Y-%m-%d")
                    window_start_date_fixed = window_start_date + datetime.timedelta(days=days)
                    trough_date = window_start_date + datetime.timedelta(days=trough_days)
                    window_start_date = window_start_date_fixed.strftime("%Y-%m-%d")
                    trough_date = trough_date.strftime("%Y-%m-%d")
                    accrophase_datetime = f"{window_start_date} {corrected_acrophase_time.split(' ')[-1]}"
                    trough_datetime = f"{trough_date} {trough_time.split(' ')[-1]}"
                else:
                    accrophase_datetime = f"{window_start_date} {corrected_acrophase_time}"
                    trough_datetime = f"{window_start_date} {trough_time}"
            else:
                if select_period_size > 24:
                    days = int(corrected_acrophase_time.split(' ')[0])
                    window_start_date = datetime.datetime.strptime(window_start_date, "%Y-%m-%d")
                    window_start_date_fixed = window_start_date + datetime.timedelta(days=days)
                    window_start_date = window_start_date_fixed.strftime("%Y-%m-%d")
                    accrophase_datetime = f"{window_start_date} {corrected_acrophase_time.split(' ')[-1]}"
                else:
                    accrophase_datetime = f"{window_start_date} {corrected_acrophase_time}"

            experiment_day_from_start = (datetime.datetime.strptime(window_start_date, "%Y-%m-%d").date() - experiment_start_date.date()).days
            nDays = len(set([(datetime.datetime.strptime(x.split(' ')[0], "%Y-%m-%d").date() - experiment_start_date.date()).days for x in results.keys()]))

            max_visu_len = np.max([500, len(original_data1)])


            visu_params = {
                'Id': [subject] * len(original_data1),
                'test': [key] * len(original_data1),
                'x': [(x*window_size)/60 for x in original_data1['x']],
                'y': original_data1['y'],
                'interpolated_y': original_data1['interpolated_y']
            }

            estimated_ts = {
                'Id': [subject] * len(estimated_x[:500]),
                'test': [key] * len(estimated_x[:500]),
                'estimated_x': [(x*window_size)/60 for x in estimated_x[:500]], #@TODO: why 500?
                'estimated_y': estimated_y[:500]
            }

            visu_df =pl.DataFrame(visu_params, strict=False)
            estimated_df = pl.DataFrame(estimated_ts, strict=False)

            visu_dfs = pl.concat([visu_dfs, visu_df], how='vertical_relaxed')
            estimated_dfs = pl.concat([estimated_dfs, estimated_df], how='vertical_relaxed')

            json_results[key] = {
                'amplitude': params['amplitude'],
                'acrophase': params['acrophase'],
                'mesor': params['mesor'],
                'theta': theta,
                'r_squared': cosinor_model.rsquared,
                'r_squared_adj': cosinor_model.rsquared_adj,
                'CI(amplitude)': params['CI(amplitude)'],
                'CI(acrophase)': params['CI(acrophase)'],
                'CI(mesor)': params['CI(mesor)'],
                'y_estimated_max_loc': estimated_x[np.argmax(estimated_y)],
                'y_estimated_min_loc': estimated_x[np.argmin(estimated_y)],
                'y_estimated_min': np.min(estimated_y),
            }




            cosinor_model_params = {
                'Id': [subject],
                'ExperimentDay': [experiment_day_from_start],
                'signal': [signal],
                'start_date': [window_start_date],
                'week_day': [datetime.datetime.strptime(window_start_date, "%Y-%m-%d").date().strftime('%A')],
                'windowSize (minutes)': [select_period_size * 60],
                'incrementSize (minutes)': [increment_size * 60],
                'downsampleRate (minutes)': [window_size],
                'interpolation': [intepolation],
                'samples': [int(original_data1['y'].notna().sum())],
                'nMissing': [int(original_data1['y'].isnull().sum())],
                'max_null_sequence (before downsample)': [max_null_sequence],
                'nMissing  (before downsample)': [total_nulls],
                'period': [float(params['period'])],
                'AIC': [float(model[0].aic)],  # ensure floats
                'BIC': [float(model[0].bic)],
                'r_squared': [float(cosinor_model.rsquared)],
                'r_squared_adj': [float(cosinor_model.rsquared_adj)],
                'mesor': [float(params['mesor'])],
                'amplitude': [float(params['amplitude'])],
                'acrophase (rad)': [float(params['acrophase'])],
                'acrophase from midnight (minutes)': [accrophs_minutes_from_midnight],
                'acrophase (time)': [corrected_acrophase_time],
                'acrophase (datetime)': [accrophase_datetime],
                'F-value': [float(cosinor_model.fvalue)],
                'F-pvalue': [float(cosinor_model.f_pvalue)],
                'df residual ': [int(cosinor_model.df_resid)],
                'df model': [int(cosinor_model.df_model)],
                'amplitude high CI': [float(params['CI(amplitude)'][1])],
                'amplitude low CI': [float(params['CI(amplitude)'][0])],
                'acrophase high CI': [float(params['CI(acrophase)'][1])],
                'acrophase low CI': [float(params['CI(acrophase)'][0])],
                'mesor high CI': [float(params['CI(mesor)'][1])],
                'mesor low CI': [float(params['CI(mesor)'][0])],
                'p amplitude': [float(params['p(amplitude)'])],
                'p acrophase': [float(params['p(acrophase)'])],
                'p mesor': [float(params['p(mesor)'])],

                # 'amplitude': [float(params['amplitude'])],
                # 'acrophase (rad)': [float(params['acrophase'])],
                # 'corrected_acrophase (rad)': [float(corrected_acrophase)],
                # 'corrected_acrophase (hours)': [corrected_acrophase_time],
                # 'corrected_acrophase (datetime)': [accrophase_datetime],
                # 'corrected_acrophase (degrees)': [float(corrected_acrophase_deg)],
                # 'mesor': [float(params['mesor'])],
                # 'AIC': [float(model[0].aic)],  # ensure floats
                # 'BIC': [float(model[0].bic)],
                # 'peaks': [str(params['peaks'])],  # Convert list to string
                # 'heights': [str(params['heights'])],  # Convert list to string
                # 'troughs': [str(params['troughs'])],
                # 'trough_time': [str(trough_time)],
                # 'trough_datetime': [trough_datetime],
                # 'y_estimated_min_loc': [float(y_estimated_min_loc)],
                # 'length': [int(length)],
                # 'heights2': [str(params['heights2'])],
                # 'max_loc': [float(params['max_loc'])],
                # 'period2': [float(params['period2'])],
                # 'p-value': [float(stats['p'])],
                # 'p_reject': [bool(stats['p_reject'])],
                # 'SNR': [float(stats['SNR'])],
                # 'RSS': [float(stats['RSS'])],
                # 'resid_SE': [float(stats['resid_SE'])],
                # 'ME': [float(stats['ME'])],
                # 'f-pvalue': [float(cosinor_model.f_pvalue)],
                # 't-values const': [float(cosinor_model.tvalues[0])],
                # 't-values x1': [float(cosinor_model.tvalues[1])],
                # 't-values x2': [float(cosinor_model.tvalues[2])],
                # 'R-squared': [float(cosinor_model.rsquared)],
                # 'R-squared adjusted': [float(cosinor_model.rsquared_adj)],
                # 'SSR': [float(cosinor_model.ssr)],
                # 'minutes_based': [int(original_data1['y'].notna().sum())]
            }
            df = pl.DataFrame(cosinor_model_params, strict=False)
            results_df = pl.concat([results_df, df], how='vertical_relaxed')

        



        return results_df, visu_dfs, json_results, estimated_dfs


    def quadrant_adjustment(thta, acrphs, radian=True):
        # Check which quadrant the acrophase falls into
        if 0 <= thta < (np.pi / 2):
            if radian:
                corrected_acrophase = acrphs
            else:
                # First quadrant: no correction needed
                corrected_acrophase = np.rad2deg(acrphs)
        elif (np.pi / 2) <= thta < np.pi:
            # Second quadrant: subtract a constant to realign
            if radian:
                corrected_acrophase = acrphs 
            else:
                corrected_acrophase =  np.rad2deg(acrphs)
        elif np.pi <= thta < (3 * np.pi / 2):
            # Third quadrant: make it negative
            if radian:
                corrected_acrophase = 2 * np.pi - acrphs
            else:
                corrected_acrophase = 360 - np.rad2deg(acrphs)
        elif (3 * np.pi / 2) <= thta < (2 * np.pi):
            if radian:
                corrected_acrophase = 2 * np.pi - acrphs
            else:
                # Fourth quadrant: shift to bring into biological range
                corrected_acrophase = 360 - np.rad2deg(acrphs)
        else:
            # If outside normal bounds, wrap it
            corrected_acrophase = acrphs % (2 * np.pi)

        return corrected_acrophase


    def cosinor_analysis(data: list, signal: str, period: int):

        results = {}
        
        data = pl.concat(data, how='vertical_relaxed').to_pandas()

        labels = data['test'].unique()
        for label in labels:
            data_for_label = data[data['test'] == label]
            data_for_label = data_for_label.dropna(subset=['y'])

            if any(data_for_label['interpolated_y'].notna()):
                data_for_label['y'] = np.where(data_for_label['y'].isnull(), data_for_label['interpolated_y'], data_for_label['y'])
            try:
                results[label] = cosinor.fit_me(data_for_label['x'], data_for_label['y'], n_components=1, period=period, plot=False, return_model=True, params_CI=True)
            except:
                results[label] = None
        return results, data
    

    def downsample_signal(df: pl.DataFrame, window_size: str, signal: str) -> pl.DataFrame:


        if 'interpolated_y' in df.columns:
            grouped = df.group_by_dynamic(
                "DateAndMinute",
                every=str(window_size) + 'm',
                period=str(window_size) + 'm',
                closed='left',
                label='left'
            ).agg([
                pl.col(signal).count().alias("count_non_missing"),
                pl.col(signal).is_null().sum().alias("count_missing"),
                pl.col(signal).mean().alias("mean"),
                pl.col("interpolated_y").mean().alias("interpolated_mean")
            ]
            )

            downsampled = grouped.with_columns(
                pl.when(pl.col("count_missing") > pl.col("count_non_missing"))
                .then(None)
                .otherwise(pl.col("mean"))
                .alias("downsampled")
            ).select([
                pl.col("DateAndMinute"),
                pl.col("downsampled"),
                pl.col("interpolated_mean")
            ])
        else:
            grouped = df.group_by_dynamic(
                "DateAndMinute",
                every=str(window_size) + 'm',
                period=str(window_size) + 'm',
                closed='left',
                label='left'
            ).agg([
                pl.col(signal).count().alias("count_non_missing"),
                pl.col(signal).is_null().sum().alias("count_missing"),
                pl.col(signal).mean().alias("mean"),
            ]
            )
            
            downsampled = grouped.with_columns(
                pl.when(pl.col("count_missing") > pl.col("count_non_missing"))
                .then(None)
                .otherwise(pl.col("mean"))
                .alias("downsampled")
            ).select([
                pl.col("DateAndMinute"),
                pl.col("downsampled")
            ])

        downsampled_df = downsampled.to_pandas()

        return downsampled_df

    def generate_fixed_windows(start_datetime, end_datetime, period_size, shift_size):
        windows = []
        window_id = 0
        delta_window = datetime.timedelta(hours=period_size)
        delta_shift = datetime.timedelta(hours=shift_size)

        current_start = start_datetime.replace(minute=0, second=0, microsecond=0)
        reminder_hours = (current_start.hour % period_size)

        if reminder_hours != 0:
            current_start += datetime.timedelta(hours=shift_size - reminder_hours)

        while current_start <= end_datetime:
            current_end = current_start + delta_window - datetime.timedelta(seconds=1)
            label = f"{current_start.strftime('%Y-%m-%d %H:%M:%S')} to {current_end.strftime('%Y-%m-%d %H:%M:%S')}"
            windows.append({
                'window_id': window_id,
                'start': current_start,
                'end': current_end,
                'label': label
            })
            current_start += delta_shift
            window_id += 1

        return windows

    def is_dst_change(date: datetime.datetime) -> bool:
        jerusalem = pytz.timezone("Asia/Jerusalem")
        
        midnight = jerusalem.localize(datetime.datetime(date.year, date.month, date.day))
        next_midnight = jerusalem.localize(datetime.datetime(date.year, date.month, date.day) + datetime.timedelta(days=1))
        
        return midnight.dst() != next_midnight.dst() 

        # except Exception as e:
        #     print(e)
        #     time.sleep(10)


    def decompose_and_interpolate(data: pl.DataFrame, period_size, signal="BpmMean", method="linear",):


        df = (
            data
            .select([
                "DateAndMinute",
                signal
            ])
            .with_columns(
                pl.col(signal).is_null().alias("missing"),
                pl.col(signal).interpolate(method=method).alias("interpolated")
            )
            .with_columns(
                pl.col('interpolated').fill_null(strategy='mean').alias('interpolated')
            )
        )

        data_stl_pandas = df.select(["DateAndMinute", "interpolated"]).to_pandas().set_index("DateAndMinute")

        stl = seasonal_decompose(data_stl_pandas, period=period_size*60, model='additive')

        stl_trend = pl.Series(stl.trend)
        stl_seasonal = pl.Series(stl.seasonal)
        stl_residual = pl.Series(stl.resid)
        polar_interpolated = pl.Series(df['interpolated'])

        adjusted = []

        for x, y in zip(polar_interpolated, stl_seasonal):
            adjusted.append(x - y)

        seasonal_adjusted = pl.Series(adjusted)

        final_df = (
            df
            .with_columns(
                seasonal=stl_seasonal,
                seasonal_adjusted=seasonal_adjusted
            )
            .with_columns(
                seasonal_adjusted=pl.when(pl.col("missing") == True)
                .then(pl.lit(None))
                .otherwise(pl.col("seasonal_adjusted"))
            )
            .with_columns(
                seasonal_adjusted=pl.col("seasonal_adjusted").interpolate(method=method)
            )
            .with_columns(
                interpolated_y=pl.col("seasonal_adjusted") + pl.col("seasonal")
            )
        )

        return final_df


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
    

except Exception as e:
    logging.info('File generation failed')

    print(f'Error: {e}')
    time.sleep(10)

if __name__ == '__main__':
    try:
        try:
            param = sys.argv[1]
            now = sys.argv[2]
            user_name = sys.argv[3]
            include_not_il = sys.argv[4]
            include_dst = sys.argv[5]
            window_size = int(sys.argv[6])
            increment_size = int(sys.argv[7])
            downsample = int(sys.argv[8])
            missing_data_threshold = int(sys.argv[9])
            intepolation = sys.argv[10]
            signal = sys.argv[11]


        except IndexError:
            param = 'NOVA_TESTS'
            now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            user_name = 'Unknown'
            include_not_il = False
            include_dst = False
            window_size = 24
            increment_size = 100
            
            downsample = 5
            missing_data_threshold = 20
            intepolation = False
            signal = 'BpmMean'
        

        window_size_in_minutes = window_size * 60
        records_per_window = window_size_in_minutes / downsample
        increment_size = int(((increment_size/100 * records_per_window) * downsample)/60)
    except Exception as e:
        print(f'Error: {e}')
        time.sleep(10)
    print(f'param: {param}')
    print(f'now: {now}')
    print(f'user_name: {user_name}')
    print(f'include_not_il: {include_not_il}')
    print(f'include_dst: {include_dst}')
    print(f'window_size: {window_size}')
    print(f'increment_size: {increment_size}')
    print(f'downsample: {downsample}')
    print(f'missing_data_threshold: {missing_data_threshold}')
    print(f'intepolation: {intepolation}')
    print(f'signal: {signal}')

    logging.basicConfig(filename=f'logs/getRhythm_{now}.log', level=logging.INFO)
    logging.info('Starting getRhythm.py')
    main(param, now, user_name, include_not_il, include_dst, window_size, increment_size, downsample, missing_data_threshold, intepolation, signal)
    time.sleep(15)