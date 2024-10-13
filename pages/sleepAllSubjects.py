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
# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings



FIRST, LAST = 0, -1
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # for the output files





dash.register_page(__name__, name='extract sleep data', order=3)

pages = {}

Pconfigs = json.load(open(r"G:\Shared drives\AdmonPsy - Lab Resources\Projects\FitbitDash\pages\Pconfigs\paths.json", "r"))

for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Sleep All Subjects Gen Page"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-Sleep-All-Subjects',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load', id='load-fitbit-button-Sleep-All-Subjects', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-Sleep-All-Subjects',
                        placeholder='user_name',
                        value=''
                    )
                ]),

                html.Hr()
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id='raw-sleep-subjects-container')
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Generate Sleep All Subjects',
                        id='Generate-file-button',
                        n_clicks=0,
                        color='success'
                    ),
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-Sleep-All-Subjects',
                                message='File Generation Started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog',
                        message='Error generating file'
                    )
                ]),
            ])
        ])
])

### fINISH THE LAYOUT
@callback(
    Output('raw-sleep-subjects-container', 'children'),
    Input('load-fitbit-button-Sleep-All-Subjects', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Sleep-All-Subjects', 'value')
)
def load_fitbit_data(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    

    
    paths_json = json.load(open(r"G:\Shared drives\AdmonPsy - Lab Resources\Projects\FitbitDash\pages\Pconfigs\paths.json", "r"))
    project_path = Path(paths_json[project])



    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

    PROJECT_CONFIG = json.load(open(r'G:\Shared drives\AdmonPsy - Lab Resources\Projects\FitbitDash\pages\Pconfigs\paths data.json', 'r'))
        
    SUBJECTS_DATES = METADATA_PATH.joinpath('Subjects Dates.csv')

    try:
        subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                    try_parse_dates=True)
    except:
        subjects_dates = pl.read_csv(SUBJECTS_DATES,
                                    parse_dates=True,
                                    encoding='utf-8')

    subjects_dates_df = subjects_dates.sort(by='Id').unique('Id').drop_nulls('Id')

    available_raw_data_folders = pl.DataFrame({
        'Subject': pl.Series([], dtype=pl.Utf8),
        'Sleep jsons': pl.Series([], dtype=pl.Int64),
        'Subject dates': pl.Series([], dtype=pl.Boolean)
    })



    for folder in os.listdir(DATA_PATH):
        sub_pattern = r'[0-9]{3}$'
        if re.search(sub_pattern, folder):
            sub_name = folder
            if not os.path.exists(Path(DATA_PATH).joinpath(sub_name, 'FITBIT')):
                print(f'No FITBIT folder for {sub_name}')
                continue
            for folder_ in os.listdir(Path(DATA_PATH).joinpath(sub_name, 'FITBIT')):
                if 'Sleep' in folder_:
                    sleep_json_pattern = r'^sleep-'
                    sleep_jsons_amount = 0
                    for file in os.listdir(Path(DATA_PATH).joinpath(sub_name, 'FITBIT', folder_)):
                        if re.search(sleep_json_pattern, file):
                            sleep_jsons_amount += 1


            if sub_name in subjects_dates_df['Id'].to_list():
                available_raw_data_folders_update = pl.DataFrame({
                    'Subject': [sub_name],
                    'Sleep jsons': [sleep_jsons_amount],
                    'Subject dates': [True]
                })
                available_raw_data_folders = pl.concat([available_raw_data_folders,available_raw_data_folders_update])

            else:
                available_raw_data_folders_update = pl.DataFrame({
                    'Subject': [sub_name],
                    'Sleep jsons': [sleep_jsons_amount],
                    'Subject dates': [False]
                })
                available_raw_data_folders = pl.concat([available_raw_data_folders,available_raw_data_folders_update])
    
    available_raw_data_folders = (
        available_raw_data_folders
        .with_columns(
            run = True
        )
    )

    rows = available_raw_data_folders.with_row_index().to_pandas().to_dict('records')

    columns_def = [
        {'headerName': 'Index', 'field': 'index', 'sortable': True, 'filter': True},
        {'headerName': 'Subject', 'field': 'Subject', 'sortable': True, 'filter': True},
        {'headerName': 'Sleep jsons', 'field': 'Sleep jsons', 'sortable': True, 'filter': True},
        {'headerName': 'Subject dates', 'field': 'Subject dates', 'sortable': True, 'filter': True},
        {'headerName': 'run', 'field': 'run', 'sortable': True, 'filter': True, 'editable': True}
    ]

    sub_table = dag.AgGrid(
        id={
            'type': 'sleep-all-subjects-table',
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
    Output('confirm-dialog-Sleep-All-Subjects', 'displayed'),
    Output('error-gen-dialog', 'displayed'),
    Output('error-gen-dialog', 'message'),
    Input('Generate-file-button', 'n_clicks'),
    State({'type': 'sleep-all-subjects-table', 'index': ALL}, 'rowData'),
    State('project-selection-dropdown-FitBit-Sleep-All-Subjects', 'value'),
    State('usenname-Sleep-All-Subjects', 'value'),
)
def initialize_folders(n_clicks, selected_rows, project, username):
    if n_clicks == 0:
        raise PreventUpdate

    paths_json = json.load(open(r"G:\Shared drives\AdmonPsy - Lab Resources\Projects\FitbitDash\pages\Pconfigs\paths.json", "r"))
    project_path = Path(paths_json[project])

    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

    PROJECT_CONFIG = json.load(open(r'G:\Shared drives\AdmonPsy - Lab Resources\Projects\FitbitDash\pages\Pconfigs\paths data.json', 'r'))

    if not selected_rows:
        raise PreventUpdate

    print(selected_rows)

    selected_rows_df = (
        pl.DataFrame(selected_rows[0])
        .filter(
            pl.col('run') == True,
            pl.col('Sleep jsons') > 0
        )
        .drop('run')
    )
            

    selected_rows_df = pl.DataFrame({
        'Id': selected_rows_df['Subject'],
        'Date': [datetime.datetime.now().strftime('%Y-%m-%d') for row in selected_rows_df['Subject']]
    })
    
    selected_rows_df.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_sleep_all_subjects.parquet')
    

    
    if username == '':
        return False, True, 'Please enter your name'


    try:

        param = project
        param2 = now
        param3 = username
        # Define the command to run the script
        script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\sleepAllSubjectsScript.py'
    
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
        

        return True, False, ''
    except Exception as e:
        return False, True, str(e)
    



                    

