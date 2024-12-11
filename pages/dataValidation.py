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
import scripts.UTILS.utils as ut

# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings



FIRST, LAST = 0, -1
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # for the output files





dash.register_page(__name__, name='set up', order=2)

pages = {}

Pconfigs = json.load(open(r".\pages\Pconfigs\paths.json", "r"))

for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
    dcc.Store(id='start-time-Data-Validation', data=now),
    dcc.Interval(id='interval-Data-Validation', interval=1000, n_intervals=0, disabled=True),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Initialize Processed Data Page"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-Data-Validation',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname',
                        placeholder='user_name',
                        value=''
                    )
                ]),

                html.Hr()
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='raw-data-subjects-container')
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Initialize data folder for selected subjects',
                        id='initialize-folder-button',
                        n_clicks=0,
                        color='success'
                    ),
                    dbc.Button(
                        'Initialize data folder for all subjects',
                        id='initialize-folder-button-all',
                        n_clicks=0,
                        color='warning'
                    ),
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog',
                                message='Folders initializing started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing Tasks...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-initialize-dialog',
                        message='Error initializing folders'
                    )
                ]),
            ])
        ])
])

### fINISH THE LAYOUT
@callback(
    Output('raw-data-subjects-container', 'children'),
    Input('load-fitbit-button', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Data-Validation', 'value')
)
def load_fitbit_data(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    

    
    paths_json = json.load(open(r".\pages\Pconfigs\paths.json", "r"))
    project_path = Path(paths_json[project])



    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)


        
    PROJECT_CONFIG = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))
        
    SUBJECTS_DATES = METADATA_PATH.joinpath('Subjects Dates.csv')

    try:
        subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                    try_parse_dates=True)
    except:
        subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                    try_parse_dates=True,
                                    encoding='utf8')

    subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')

    available_raw_data_folders = pl.DataFrame({
        'Id': pl.Series([], dtype=pl.Utf8),
        'Sleep jsons': pl.Series([], dtype=pl.Int64),
        'Device Temperature': pl.Series([], dtype=pl.Int64),
        'Computed Temperature': pl.Series([], dtype=pl.Int64),
        'Daily Heart Rate Variability Summary': pl.Series([], dtype=pl.Int64),
        'Daily Heart Rate Variability Detail': pl.Series([], dtype=pl.Int64),
        'Daily Respiratory Rate Summary': pl.Series([], dtype=pl.Int64),
        'Heart Rate': pl.Series([], dtype=pl.Int64),
        'Steps': pl.Series([], dtype=pl.Int64),
        'Mindfulness EDA Data Sessions': pl.Series([], dtype=pl.Int64),
        'Mindfulness Sessions': pl.Series([], dtype=pl.Int64),
        'Subject dates': pl.Series([], dtype=pl.Boolean)
    })



    for folder in os.listdir(PROJECT_CONFIG[project]):
        sub_pattern = r'[0-9]{3}$'
        if re.search(sub_pattern, folder):
            sub_name = folder
            if not os.path.exists(Path(PROJECT_CONFIG[project]).joinpath(sub_name, 'FITBIT')):
                print(f'No FITBIT folder for {sub_name}')
                continue
            folders_to_search = ['Sleep', 'Physical Activity', 'Stress']
            if not all([folder_ in os.listdir(Path(PROJECT_CONFIG[project]).joinpath(sub_name, 'FITBIT')) for folder_ in folders_to_search]):
                print(f'Not all folders for {sub_name} are present')
                continue
            for folder_ in os.listdir(Path(PROJECT_CONFIG[project]).joinpath(sub_name, 'FITBIT')):
                if 'Sleep' in folder_:
                    sleep_json_pattern = r'^sleep-'
                    device_temperature_pattern = r'^Device Temperature -'
                    computed_temperature_pattern = r'^Computed Temperature -'
                    daily_heart_rate_variability_summary_pattern = r'^Daily Heart Rate Variability Summary -'
                    daily_heat_rate_variability_detail_pattern = r'^Daily Heart Rate Variability Details -'
                    daily_respiratory_rate_summary_pattern = r'^Daily Respiratory Rate Summary -'
                    sleep_jsons_amount = 0
                    device_temperature_amount = 0
                    computed_temperature_amount = 0
                    daily_heart_rate_variability_summary_amount = 0
                    daily_heat_rate_variability_detail_amount = 0
                    daily_respiratory_rate_summary_amount = 0
                    for file in os.listdir(Path(PROJECT_CONFIG[project]).joinpath(sub_name, 'FITBIT', folder_)):
                        if re.search(sleep_json_pattern, file):
                            sleep_jsons_amount += 1
                        elif re.search(device_temperature_pattern, file):
                            device_temperature_amount += 1
                        elif re.search(computed_temperature_pattern, file):
                            computed_temperature_amount += 1
                        elif re.search(daily_heart_rate_variability_summary_pattern, file):
                            daily_heart_rate_variability_summary_amount += 1
                        elif re.search(daily_heat_rate_variability_detail_pattern, file):
                            daily_heat_rate_variability_detail_amount += 1
                        elif re.search(daily_respiratory_rate_summary_pattern, file):
                            daily_respiratory_rate_summary_amount += 1
                elif 'Physical Activity' in folder_:
                    heart_rate_pattern = r'^heart_rate-'
                    steps_pattern = r'^steps-'
                    heart_rate_amount = 0
                    steps_amount = 0
                    for file in os.listdir(Path(PROJECT_CONFIG[project]).joinpath(sub_name, 'FITBIT', folder_)):
                        if re.search(heart_rate_pattern, file):
                            heart_rate_amount += 1
                        elif re.search(steps_pattern, file):
                            steps_amount += 1

                elif 'Stress' in folder_:
                    mindfulness_eda_data_pattern = r'^Mindfulness Eda Data Sessions'
                    mindfulness_sessions_pattern = r'^Mindfulness Sessions'
                    mindfulness_eda_data_amount = 0
                    mindfulness_sessions_amount = 0
                    for file in os.listdir(Path(PROJECT_CONFIG[project]).joinpath(sub_name, 'FITBIT', folder_)):
                        if re.search(mindfulness_eda_data_pattern, file):
                            mindfulness_eda_data_amount += 1
                        elif re.search(mindfulness_sessions_pattern, file):
                            mindfulness_sessions_amount += 1

            if sub_name in subjects_dates_df['Id'].to_list():
                available_raw_data_folders_update = pl.DataFrame({
                    'Id': [sub_name],
                    'Sleep jsons': [sleep_jsons_amount],
                    'Device Temperature': [device_temperature_amount],
                    'Computed Temperature': [computed_temperature_amount],
                    'Daily Heart Rate Variability Summary': [daily_heart_rate_variability_summary_amount],
                    'Daily Heart Rate Variability Detail': [daily_heat_rate_variability_detail_amount],
                    'Daily Respiratory Rate Summary': [daily_respiratory_rate_summary_amount],
                    'Heart Rate': [heart_rate_amount],
                    'Steps': [steps_amount],
                    'Mindfulness EDA Data Sessions': [mindfulness_eda_data_amount],
                    'Mindfulness Sessions': [mindfulness_sessions_amount],
                    'Subject dates': [True]
                })
                available_raw_data_folders = pl.concat([available_raw_data_folders,available_raw_data_folders_update])

            else:
                available_raw_data_folders_update = pl.DataFrame({
                    'Id': [sub_name],
                    'Sleep jsons': [sleep_jsons_amount],
                    'Device Temperature': [device_temperature_amount],
                    'Computed Temperature': [computed_temperature_amount],
                    'Daily Heart Rate Variability Summary': [daily_heart_rate_variability_summary_amount],
                    'Daily Heart Rate Variability Detail': [daily_heat_rate_variability_detail_amount],
                    'Daily Respiratory Rate Summary': [daily_respiratory_rate_summary_amount],
                    'Heart Rate': [heart_rate_amount],
                    'Steps': [steps_amount],
                    'Mindfulness EDA Data Sessions': [mindfulness_eda_data_amount],
                    'Mindfulness Sessions': [mindfulness_sessions_amount],
                    'Subject dates': [False]
                })
                available_raw_data_folders = pl.concat([available_raw_data_folders,available_raw_data_folders_update])
                
    rows = available_raw_data_folders.with_row_index().to_pandas().to_dict('records')

    columns_def = [
        {'headerName': 'Index', 'field': 'index', 'sortable': True, 'filter': True, 'checkboxSelection': True},
        {'headerName': 'Id', 'field': 'Id', 'sortable': True, 'filter': True},
        {'headerName': 'Sleep jsons', 'field': 'Sleep jsons', 'sortable': True, 'filter': True},
        {'headerName': 'Device Temperature', 'field': 'Device Temperature', 'sortable': True, 'filter': True},
        {'headerName': 'Computed Temperature', 'field': 'Computed Temperature', 'sortable': True, 'filter': True},
        {'headerName': 'Daily Heart Rate Variability Summary', 'field': 'Daily Heart Rate Variability Summary', 'sortable': True, 'filter': True},
        {'headerName': 'Daily Heart Rate Variability Detail', 'field': 'Daily Heart Rate Variability Detail', 'sortable': True, 'filter': True},
        {'headerName': 'Daily Respiratory Rate Summary', 'field': 'Daily Respiratory Rate Summary', 'sortable': True, 'filter': True},
        {'headerName': 'Heart Rate', 'field': 'Heart Rate', 'sortable': True, 'filter': True},
        {'headerName': 'Steps', 'field': 'Steps', 'sortable': True, 'filter': True},
        {'headerName': 'Mindfulness EDA Data Sessions', 'field': 'Mindfulness EDA Data Sessions', 'sortable': True, 'filter': True},
        {'headerName': 'Mindfulness Sessions', 'field': 'Mindfulness Sessions', 'sortable': True, 'filter': True},
        {'headerName': 'Subject dates', 'field': 'Subject dates', 'sortable': True, 'filter': True}
    ]

    sub_table = dag.AgGrid(
        id={
            'type': 'raw-data-subjects-table',
            'index': 1
        },
        columnDefs=columns_def,
        rowData=rows,
        defaultColDef={
            'resizable': True,
            'editable': False,
            'sortable': True,
            'filter': True
        },
        columnSize = 'autoSize',
        dashGridOptions={'pagination': True, 'paginationPageSize': 10, 'rowSelection': 'multiple'}
    )

    return html.Div([
        sub_table
    ])        


@callback(
    Output('confirm-dialog', 'displayed'),
    Output('error-initialize-dialog', 'displayed'),
    Output('error-initialize-dialog', 'message'),
    Output('interval-Data-Validation', 'disabled'),
    Output('start-time-Data-Validation', 'data'),
    Input('initialize-folder-button', 'n_clicks'),
    State({'type': 'raw-data-subjects-table', 'index': ALL}, 'selectedRows'),
    State('project-selection-dropdown-FitBit-Data-Validation', 'value'),
    State('usenname', 'value')
)
def initialize_folders(n_clicks, selected_rows, project, username):
    if n_clicks == 0:
        raise PreventUpdate

    paths_json = json.load(open(r".\pages\Pconfigs\paths.json", "r"))
    project_path = Path(paths_json[project])

    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

    PROJECT_CONFIG = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))

    if not selected_rows:
        raise PreventUpdate

    selected_rows_df = pl.DataFrame({
        'Id': [row['Id'] for row in selected_rows[0]],
        'Date': [datetime.datetime.now().strftime('%Y-%m-%d') for row in selected_rows[0]]
    })
    
    selected_rows_df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_folders_init.parquet')

    if username == '':
        return False, True, 'Please enter your name', True, ''
    
    



    try:

        param = project
        param2 = now
        param3 = username
        # Define the command to run the script
        script_path = r'.\pages\scripts\dataFoldersInit.py'  

        if platform.system() == "Windows":
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3}'  # Adjust this for non-Windows systems if needed
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3}'  # Adjust this for non-Windows systems if needed

        # Run the script in a new CMD window
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        # Optionally wait for the process to complete and capture output
        # stdout, stderr = process.communicate()


        return False, False, '', False, now
    except Exception as e:
        return False, True, str(e), True, ''
    



                    


@callback(
    Output('confirm-dialog', 'displayed', allow_duplicate=True),
    Output('error-initialize-dialog', 'displayed', allow_duplicate=True),
    Output('error-initialize-dialog', 'message', allow_duplicate=True),
    Input('initialize-folder-button-all', 'n_clicks'),
    State({'type': 'raw-data-subjects-table', 'index': ALL}, 'rowData'),
    State('project-selection-dropdown-FitBit-Data-Validation', 'value'),
    State('usenname', 'value'),
    prevent_initial_call=True
)
def initialize_folders(n_clicks, rows, project, username):
    if n_clicks == 0:
        raise PreventUpdate

    paths_json = json.load(open(r".\pages\Pconfigs\paths.json", "r"))
    project_path = Path(paths_json[project])

    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

    PROJECT_CONFIG = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))

    if not rows:
        raise PreventUpdate

    selected_rows_df = pl.DataFrame({
        'Id': [row['Id'] for row in rows[0]],
        'Date': [datetime.datetime.now().strftime('%Y-%m-%d') for row in rows[0]]
    })
    
    selected_rows_df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_folders_init.parquet')

    if username == '':
        return False, True, 'Please enter your name', True, ''
    
    



    try:

        param = project
        param2 = now
        param3 = username
        # Define the command to run the script
        script_path = r'.\pages\scripts\dataFoldersInit.py'
            
        if platform.system() == "Windows":
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3}'  # Adjust this for non-Windows systems if needed
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3}'  # Adjust this for non-Windows systems if needed

        # Run the script in a new CMD window
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        # Optionally wait for the process to complete and capture output
        # stdout, stderr = process.communicate()


        return False, False, '', False, now
    except Exception as e:
        return False, True, str(e)
    



                    


                
@callback(
    Output('interval-Data-Validation', 'disabled', allow_duplicate=True),
    Output('confirm-dialog', 'displayed', allow_duplicate=True),
    Output('confirm-dialog', 'message', allow_duplicate=True),
    Input('interval-Data-Validation', 'n_intervals'),
    State('start-time-Data-Validation', 'data'),
    prevent_initial_call=True   
)
def check_file_generation(n_intervals, start_time):
    if n_intervals == 0:
        raise PreventUpdate
    
    print(f'Checking file generation: {n_intervals}')
    if n_intervals > 0:
        print(f'Checking file generation: {n_intervals}')
        # C:\Users\PsyLab-6028\Desktop\FitbitDash\logs\sleepAllSubjectsScript_2024-12-11_18-35-35.log
        log_path = Path(rf'.\logs\Data-Validation_{start_time}.log')
        if os.path.exists(log_path):
            print(f'Checking file generation: {n_intervals}')
            with open(log_path, 'r') as f:
                log = f.read()
                print(f'lOG: {log}')
            if 'File generation completed' in log:
                with open(log_path, 'a') as f:
                    f.write(log + '\n' + 'File generation confirmed')
                return True, True, 'File generation completed'
            elif 'File generation failed' in log:
                with open(log_path, 'a') as f:
                    f.write(log + '\n' + 'File generation failed')
                    message = 'File generation failed' + '\n' + f'{log}'

                return True, True, message
            else:
                return False, False, ''
        else:
            return False, False, ''
        
    return False, False, ''
    
    