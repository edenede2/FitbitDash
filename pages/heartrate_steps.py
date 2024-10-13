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





dash.register_page(__name__, name='Preprocessing', order=4)

pages = {}

Pconfigs = json.load(open(r"C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\Pconfigs\paths.json", "r"))

for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Preprocessing Heart Rate and Steps Data"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-Preprocessing',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-Preprocessing', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-Preprocessing',
                        placeholder='user_name',
                        value=''
                    )
                ]),

                html.Hr()
            ]),
        ]),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div(id='raw-data-subjects-container-Preprocessing')
                ]),
                dbc.Col([
                    html.Div(id='available-data-container-Preprocessing')
                ])
            ])
        ]),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-Preprocessing',
                                message='File Generation Started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-Preprocessing',
                        message='Error generating file'
                    )
                ]),
            ])
])

@callback(
    Output('raw-data-subjects-container-Preprocessing', 'children'),
    Input('load-fitbit-button-Preprocessing', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Preprocessing', 'value')
)
def load_raw_data(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate

    data_path = Pconfigs[project]
    print(f'Project proccesed data path: {project}')

    processed_data_path = Path(data_path) / 'Data'
    print(f'Processed data path: {processed_data_path}')

    if not processed_data_path.exists():
        return html.Div('No processed data found, please go to the data validation page')
        # print('No processed data found, please go to the data validation page')

    subject_pattern = r'\d{3}$'
    subjects = [x for x in processed_data_path.iterdir() if x.is_dir() and re.search(subject_pattern, x.name)]
    print(f'Subjects in processed data: {subjects}')



    if not subjects:
        return html.Div('No subjects found in processed data')
        # print('No subjects found in processed data')
    raw_data_df = pl.DataFrame()

    for subject in subjects:
        print(f'Searching for data in the folder of subject: {subject}')

        files_path = processed_data_path / subject / 'FITBIT' / 'Physical Activity'

        hr_files = []
        steps_files = []
        
        for file in files_path.iterdir():
            print(f'File: {file}')
            if file.name.startswith('heart_rate-'):
                hr_files.append(file.name)
            elif file.name.startswith('steps-'):
                steps_files.append(file.name)
            elif file.name.startswith('api-heart_rate-'):
                hr_files.append(file.name)
            elif file.name.startswith('api-steps-'):
                steps_files.append(file.name)

        # df_height = np.max([len(hr_files), len(steps_files)])
        # add the name of the files to the dataframe
        subject_found_files_df = (
            pl.DataFrame({
                'subject': pl.Series([subject.name]),
                'hr_files': pl.Series([len(hr_files)]),
                'steps_files': pl.Series([len(steps_files)])
            })
        )

        raw_data_df = (
            pl.concat([raw_data_df, subject_found_files_df])
        )


    raw_data_df = (
        raw_data_df
        .sort('subject')
        .with_columns(
            run = True
        )
    )

    defcolumns = [
        {'field': 'subject', 'headerName': 'subject', 'sortable': True, 'filter': True, 'checkboxSelection': True},
        {'field': 'hr_files', 'headerName': 'hr_files', 'sortable': True, 'filter': True},
        {'field': 'steps_files', 'headerName': 'steps_files', 'sortable': True, 'filter': True},
        {'field': 'run', 'headerName': 'run', 'sortable': True, 'filter': True, 'editable': True}
    ]

    rows = raw_data_df.to_pandas().to_dict('records')
    print(f'Raw data df: {raw_data_df}')

    grid = dag.AgGrid(
        id={
            'type': 'raw-data-table',
            'index': 1
        },
        columnDefs=defcolumns,
        rowData=rows,
        defaultColDef={
            'resizable': True,
            'editable': True,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 10,
            'rowSelection': 'single',
        }
    )

    run_button = dbc.Button('Run Preprocessing Step', id={
        'type': 'run-preprocessing-button',
        'index': 1
    }, n_clicks=0, color='success')

    run_selected_button = dbc.Button('Run only on selected subjects', id={
        'type': 'run-selected-preprocessing-button',
        'index': 1
    }, n_clicks=0, color='warning')

    show_button = dbc.Button('Show available data', id={
        'type': 'show-available-data-button',
        'index': 1
    }, n_clicks=0, color='warning')


    return [
        html.H4('Raw data:'),
        grid,
        run_button,
        show_button
    ]


        
@callback(
    Output('available-data-container-Preprocessing', 'children'),
    Input({'type': 'show-available-data-button','index': ALL}, 'n_clicks'),
    State({'type': 'raw-data-table', 'index': ALL}, 'selectedRows'),
    State('project-selection-dropdown-FitBit-Preprocessing', 'value')
)
def show_available_data(n_clicks, selected_rows, project):
    if n_clicks == 0:
        raise PreventUpdate

    data_path = Pconfigs[project]
    print(f'Project proccesed data path: {project}')

    processed_data_path = Path(data_path) / 'Data'
    print(f'Processed data path: {processed_data_path}')

    if not processed_data_path.exists():
        return html.Div('No processed data found, please go to the data validation page')
        # print('No processed data found, please go to the data validation page')

    print(f'Selected rows: {selected_rows}')
    selected_subject = selected_rows[0][0]['subject']
        
    subject_path = processed_data_path / selected_subject / 'FITBIT' / 'Physical Activity'
    print(f'Selected subject: {selected_subject}')

    hr_files = []
    steps_files = []

    for file in subject_path.iterdir():
        if file.name.startswith('heart_rate-'):
            hr_files.append(file.name)
        elif file.name.startswith('steps-'):
            steps_files.append(file.name)
        elif file.name.startswith('api-heart_rate-'):
            hr_files.append(file.name)
        elif file.name.startswith('api-steps-'):
            steps_files.append(file.name)

    hr_files = sorted(hr_files)
    steps_files = sorted(steps_files)

    hr_files_df = pl.DataFrame({
        'hr_files': pl.Series(hr_files)
    })

    steps_files_df = pl.DataFrame({
        'steps_files': pl.Series(steps_files)
    })

    defcolumns_hr = [
        {'field': 'hr_files', 'headerName': 'hr_files', 'sortable': True, 'filter': True}
    ]

    defcolumns_steps = [
        {'field': 'steps_files', 'headerName': 'steps_files', 'sortable': True, 'filter': True}
    ]

    rows_hr = hr_files_df.to_pandas().to_dict('records')

    grid_hr = dag.AgGrid(
        id={
            'type': 'hr-files-table',
            'index': 1
        },
        columnDefs=defcolumns_hr,
        rowData=rows_hr,
        defaultColDef={
            'resizable': True,
            'editable': True,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 10,
            'rowSelection': 'single',
        }
    )

    rows_steps = steps_files_df.to_pandas().to_dict('records')

    grid_steps = dag.AgGrid(
        id={
            'type': 'steps-files-table',
            'index': 1
        },
        columnDefs=defcolumns_steps,
        rowData=rows_steps,
        defaultColDef={
            'resizable': True,
            'editable': True,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 10,
            'rowSelection': 'single',
        }
    )

    return [
        html.H4(f'Available data for {selected_subject}:'),
        html.H5('Heart Rate:'),
        grid_hr,
        html.H5('Steps:'),
        grid_steps
    ]

        
@callback(
    Output('confirm-dialog-Preprocessing', 'displayed'),
    Output('error-gen-dialog-Preprocessing', 'displayed'),
    Output('error-gen-dialog-Preprocessing', 'message'),
    Input({'type': 'run-preprocessing-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-data-table', 'index': ALL}, 'rowData'),
    State('usenname-Preprocessing', 'value'),
    State('project-selection-dropdown-FitBit-Preprocessing', 'value')
)
def run_preprocessing(n_clicks, raw_data, username, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    if n_clicks == []:
        return False, False, ''
    
    if n_clicks[0] == 0:
        return False, False, ''
    
    print(f'n_clicks: {n_clicks}')

    df = pl.DataFrame(raw_data[0])
    print(f'Raw data: {df}')

    if not df['run'].any():
        return False, True, 'No subjects selected to run preprocessing'
    
    df = (
        df
        .filter(
            pl.col('hr_files') > 0,
            pl.col('steps_files') > 0,
        )
    )
    df.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_Preprocessing.parquet')

    if username == '':
        return False, True, 'Please enter your name before running the preprocessing'
        

    try:

        param = project
        param2 = now
        param3 = username

        script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\preprocessing.py'

        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3}'
            print(command)
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3}'
            print(command)

        process = subprocess.Popen(command, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return True, False, ''
    except Exception as e:
        print(e)
        return False, True, str(e)


    

@callback(
    Output('confirm-dialog-Preprocessing', 'displayed', allow_duplicate=True),
    Output('error-gen-dialog-Preprocessing', 'displayed', allow_duplicate=True),
    Output('error-gen-dialog-Preprocessing', 'message', allow_duplicate=True),
    Input({'type': 'run-selected-preprocessing-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-data-table', 'index': ALL}, 'selectedRows'),
    State('usenname-Preprocessing', 'value'),
    State('project-selection-dropdown-FitBit-Preprocessing', 'value'),
    prevent_initial_call=True
)
def run_preprocessing(n_clicks, selected_rows, username, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    if n_clicks == []:
        return False, False, ''
    
    if n_clicks[0] == 0:
        return False, False, ''
    
    print(f'n_clicks: {n_clicks}')

    df = pl.DataFrame(selected_rows[0])
    print(f'Raw data: {df}')

    if not df['run'].any():
        return False, True, 'No subjects selected to run preprocessing'
    
    df = (
        df
        .filter(
            pl.col('hr_files') > 0,
            pl.col('steps_files') > 0,
        )
    )
    df.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_folders_Preprocessing.parquet')

    if username == '':
        return False, True, 'Please enter your name before running the preprocessing'
        

    try:

        param = project
        param2 = now
        param3 = username

        script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\preprocessing.py'

        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3}'
            print(command)
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3}'
            print(command)

        process = subprocess.Popen(command, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return True, False, ''
    except Exception as e:
        print(e)
        return False, True, str(e)


    