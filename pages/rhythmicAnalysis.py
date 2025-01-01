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
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import configparser
import numpy as np
import polars as pl
import datetime as dt
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
now = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # for the output files





dash.register_page(__name__, name='Rhythmic Analysis', order=8)

pages = {}

Pconfigs = json.load(open(r".\pages\Pconfigs\paths.json", "r"))


for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dcc.Store(id='file-data-store-rhythmic', storage_type='memory'),
        dcc.Store(id='output-file-path-rhythmic', storage_type='memory'),
        dcc.Store(id='start-time-rhythmic', storage_type='memory'),
        dcc.Interval(id='interval-rhythmic', interval=3000, n_intervals=0, disabled=True),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Rhythmic Analysis"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-rhythmic',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-rhythmic', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-rhythmic',
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
                    html.Div(id='raw-data-subjects-container-rhythmic')
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='available-data-container-rhythmic')
                ])
            ])
        ]),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-rhythmic',
                                message='File Generation Started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-rhythmic',
                        message='Error generating file'
                    )
                ]),
            dbc.Row([
                    dbc.Col([
                        dbc.Button('Load outputs', id='load-outputs-button-rhythmic', n_clicks=0, color='primary')
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='outputs-rhythmic'),
                    ]),
                    dbc.Col([
                        html.Div(id='outputs-dropdowns-rhythmic'),
                    ])
                ])
            ]),
            html.Div([
                html.Hr(),
                html.Div(id='outputs-plots-rhythmic'),
            ], style={'margin-top': '50px'})
])

@callback(
    Output('raw-data-subjects-container-rhythmic', 'children'),
    Input('load-fitbit-button-rhythmic', 'n_clicks'),
    State('project-selection-dropdown-FitBit-rhythmic', 'value')
)
def load_raw_data(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    data_path = Pconfigs[project]
    print(f'Project proccesed data path: {project}')

    output_path = Path(data_path) / 'Outputs'
    print(f'Output data path: {output_path}')

    if not output_path.exists():
        return html.Div('No processed data found, please go to the data validation page and check for eda files')
        # print('No processed data found, please go to the data validation page')

    subject_pattern = r'\d{3}$'
    subjects = [x for x in output_path.iterdir() if x.is_dir() and re.search(subject_pattern, x.name)]
    print(f'Subjects in processed data: {subjects}')



    if not subjects:
        return html.Div('No subjects found in processed data')
        # print('No subjects found in processed data')
    raw_data_df = pl.DataFrame()

    
    subjects_tqdm = tqdm(subjects, desc='Searching for data in the folder of subjects', position=0, leave=True)
    for subject in subjects_tqdm:
        subjects_tqdm.set_description(f'Searching for data in the folder of subject: {subject}')
        print(f'Searching for data in the folder of subject: {subject}')
        relevant_files = []
        creation_dates = []
        n_dates = 0
        n_samples = 0
        files_path = output_path / subject
        
        for file in files_path.iterdir():
            print(f'File: {file.name}')
            if file.name.endswith(' Heart Rate and Steps and Sleep Aggregated.csv'):
                relevant_files.append(file)
                creation_dates.append(dt.datetime.fromtimestamp(file.stat().st_ctime))
                n_dates = len(pl.read_csv(file, try_parse_dates=True).select(pl.col('DateAndMinute').cast(pl.Date)).unique())
                n_samples = len(pl.read_csv(file, try_parse_dates=True))
            
            



        # df_height = np.max([len(hr_files), len(steps_files)])
        # add the name of the files to the dataframe
        subject_found_files_df = (
            pl.DataFrame({
                'subject': pl.Series([subject.name]),
                'n_files': pl.Series([len(relevant_files)]),
                'creation_date': pl.Series([creation_dates]),
                'n_dates': pl.Series([n_dates]),
                'n_samples': pl.Series([n_samples])
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
        {'field': 'n_files', 'headerName': 'n_files', 'sortable': True, 'filter': True},
        {'field': 'creation_date', 'headerName': 'creation_date', 'sortable': True, 'filter': True},
        {'field': 'n_dates', 'headerName': 'n_dates', 'sortable': True, 'filter': True},
        {'field': 'n_samples', 'headerName': 'n_samples', 'sortable': True, 'filter': True},
        {'field': 'run', 'headerName': 'run', 'sortable': True, 'filter': True, 'editable': True}
    ]

    rows = raw_data_df.to_pandas().to_dict('records')
    print(f'Raw data df: {raw_data_df}')

    grid = dag.AgGrid(
        id={
            'type': 'raw-rhythmic-data-table',
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
            'rowSelection': 'multiple',
        }
    )

    
    inclusions_params_card = dbc.Card([
            dbc.CardHeader('Inclusions parameters', style={'font-size': '20px'}),
            dbc.CardBody([
                html.H5('Include dates when the subject was not in Israel?'),
                html.P('By the not_in_israel xlsx file'),
                dbc.Checkbox(id={'type':'include-dates-checkbox', 'index': 1}, value=False),
                html.Br(),
                html.H5('Include dates of dst changes?'),
                dbc.Checkbox(id= { 'type': 'include-dst-checkbox', 'index': 1}, value=False)
            ])
        ])
    
    window_params_card = dbc.Card([
            dbc.CardHeader('Window parameters', style={'font-size': '20px'}),
            dbc.CardBody([
                html.H5('Window size (in hours)'),
                dcc.Slider(
                    id={'type': 'window-size-slider', 'index': 1},
                    min = 12,
                    max = 168,
                    step = 12,
                    value = 24,
                    marks = {
                        12: '12',
                        24: '24',
                        36: '36',
                        48: '48',
                        60: '60',
                        72: '72',
                        84: '84',
                        96: '96',
                        108: '108',
                        120: '120',
                        132: '132',
                        144: '144',
                        156: '156',
                        168: '168'
                    }
                ),
                html.Br(),
                html.H5('Increment size (in percent)'),
                html.P('The increment size is the percentage of the window size that the window will move each time'),
                dcc.Slider(
                    id={'type': 'increment-size-slider', 'index': 1},
                    min = 0,
                    max = 100,
                    step = 10,
                    value = 50,
                    marks = {
                        0: '0',
                        10: '10',
                        20: '20',
                        30: '30',
                        40: '40',
                        50: '50',
                        60: '60',
                        70: '70',
                        80: '80',
                        90: '90',
                        100: '100'
                    }
                )
            ]),
        ])

    analysis_params_card = dbc.Card([
            dbc.CardHeader('Analysis parameters', style={'font-size': '20px'}),
            dbc.CardBody([
                html.H5('Downsample rate'),
                html.P('The downsample rate is the rate at which the data will be downsampled'),
                dcc.Slider(
                    id={'type': 'downsample-rate-slider', 'index': 1},
                    min = 1,
                    max = 120,
                    value = 5,
                    marks = {
                        1: {'label': '1 m', 'style': {'font-size': '10px'}},
                        5: {'label': '5 m', 'style': {'font-size': '10px'}},
                        10: {'label': '10 m', 'style': {'font-size': '10px'}},
                        15: {'label': '15 m', 'style': {'font-size': '10px'}},
                        30: {'label': '30 m', 'style': {'font-size': '10px'}},
                        60: {'label': '1 H', 'style': {'font-size': '10px'}},
                        120: {'label': '2 H', 'style': {'font-size': '10px'}}
                    }
                ),
                html.Br(),
                html.H5('Missing data threshold'),
                html.P('The missing data threshold is the percentage of missing data per window that will be tolerated'),
                dcc.Slider(
                    id={'type': 'missing-data-threshold-slider', 'index': 1},
                    min = 0,
                    max = 100,
                    value = 10,
                    marks = {
                        0: '0',
                        10: '10',
                        20: '20',
                        30: '30',
                        40: '40',
                        50: '50',
                        60: '60',
                        70: '70',
                        80: '80',
                        90: '90',
                        100: '100'
                    }
                ),
                html.Br(),
                html.H5('Data interpolation'),
                html.P('To interpolate the data with sessional component decomposition and linear interpolation'),
                dbc.Checkbox(id={'type': 'data-interpolation-checkbox', 'index': 1}, value=False),
                html.Br(),
                html.H5('Signal to analyze'),
                html.P('Select the signal to analyze'),
                dcc.Dropdown(
                    id={'type': 'signal-dropdown', 'index': 1},
                    options=[
                        {'label': 'Heart Rate', 'value': 'BpmMean'},
                        {'label': 'Steps', 'value': 'StepsInMinute'},
                        {'label': 'Temperature', 'value': 'Temperature'}
                    ],
                    value='BpmMean'
                )

            ])
        ])
                        




    run_button = dbc.Button('Run rhythmic analysis', id={
        'type': 'run-rhythmic-button',
        'index': 1
    }, n_clicks=0, color='success')

    run_selected_button = dbc.Button('Run selected subjects', id={
        'type': 'run-selected-rhythmic-button',
        'index': 1
    }, n_clicks=0, color='danger')

    show_button = dbc.Button('Show available data', id={
        'type': 'show-available-rhythmic-data-button',
        'index': 1
    }, n_clicks=0, color='warning')



    return [
        dbc.Container([    
            dbc.Row([
                dbc.Col([
                    html.H4('Raw data:'),
                    grid,
                ], width = 8),
                dbc.Col([
                    inclusions_params_card
                ], width = 4),
            ]),
            dbc.Row([
                window_params_card
            ]),
            dbc.Row([
                analysis_params_card
            ]),
            dbc.Row([
                dbc.Col([
                    run_button,
                    show_button
                ])
            ])
        ])
    ]


        
@callback(
    Output('available-data-container-rhythmic', 'children'),
    Input({'type': 'show-available-rhythmic-data-button','index': ALL}, 'n_clicks'),
    State({'type': 'raw-rhythmic-data-table', 'index': ALL}, 'selectedRows'),
    State('project-selection-dropdown-FitBit-rhythmic', 'value')
)
def show_available_data(n_clicks, selected_rows, project):
    if n_clicks == 0:
        raise PreventUpdate

    project_path = Path(paths_json[project])
    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables_custom(project_path, '_NEW_CODE')
    data_path = DATA_PATH
    print(f'Project proccesed data path: {project}')

    output_path = OUTPUT_PATH
    print(f'Processed data path: {output_path}')

    if not output_path.exists():
        return html.Div('No processed data found, please go to the data validation page')
        # print('No processed data found, please go to the data validation page')

    print(f'Selected rows: {selected_rows}')
    selected_subject = selected_rows[0][0]['subject']
        
    subject_path = output_path / selected_subject 
    print(f'Selected subject: {selected_subject}')

    relevant_file_df = pl.DataFrame()

    for file in subject_path.iterdir():
        print(f'File: {file}')
        if file.name.endswith('Heart Rate and Steps and Sleep Aggregated.csv'):
            relevant_file_df = (
                pl.read_csv(file)
            )
        else:
            print('No relevant files found')
            return html.Div('No relevant files found')
    


    defcolumns_relevant = [
        {'field': col, 'headerName': col, 'sortable': True, 'filter': True} for col in relevant_file_df.columns
    ]


    rows_relevant_data = relevant_file_df.to_pandas().to_dict('records')

    grid_relevant_data = dag.AgGrid(
        id={
            'type': 'rhythmic-data-files-table',
            'index': 1
        },
        columnDefs=defcolumns_relevant,
        rowData=rows_relevant_data,
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
        html.H5('Heart Rate and Steps and Sleep Aggregated:'),
        grid_relevant_data
    ]

        
@callback(
    Output('confirm-dialog-rhythmic', 'displayed'),
    Output('error-gen-dialog-rhythmic', 'displayed'),
    Output('error-gen-dialog-rhythmic', 'message'),
    Output('interval-rhythmic', 'disabled'),
    Output('start-time-rhythmic', 'data'),
    Input({'type': 'run-rhythmic-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-rhythmic-data-table', 'index': ALL}, 'rowData'),
    State('usenname-rhythmic', 'value'),
    State('project-selection-dropdown-FitBit-rhythmic', 'value'),
    State({'type': 'include-dates-checkbox', 'index': ALL}, 'value'),
    State({'type': 'include-dst-checkbox', 'index': ALL}, 'value'),
    State({'type': 'window-size-slider', 'index': ALL}, 'value'),
    State({'type': 'increment-size-slider', 'index': ALL}, 'value'),
    State({'type': 'downsample-rate-slider', 'index': ALL}, 'value'),
    State({'type': 'missing-data-threshold-slider', 'index': ALL}, 'value'),
    State({'type': 'data-interpolation-checkbox', 'index': ALL}, 'value'),
    State({'type': 'signal-dropdown', 'index': ALL}, 'value')
)
def run_preprocessing(n_clicks, raw_data, username, project, include_not_in_il, include_dst, window_size, increment_size, downsample_rate, missing_data_threshold, data_interpolation, signal):
    if n_clicks == 0:
        raise PreventUpdate
    
    if n_clicks == []:
        return False, False, '', False, ''
    
    if n_clicks[0] == 0:
        return False, False, '', False, ''
    
    print(f'n_clicks: {n_clicks}')

    df = pl.DataFrame(raw_data[0])
    print(f'Raw data: {df}')

    if not df['run'].any():
        return False, True, 'No subjects selected to run Rhythmic', True, ''
    

    df = (
        df
        .filter(
            pl.col('n_files') > 0
        )
    )

    df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_folders_rhythmic.parquet')

    if username == '':
        return False, True, 'Please enter your name before running the rhythmic analysis', True, ''
    # ls 
        

    try:

        param = project
        param2 = now
        param3 = username
        param4 = include_not_in_il[0]
        param5 = include_dst[0]
        param6 = window_size[0]
        param7 = increment_size[0]
        param8 = downsample_rate[0]
        param9 = missing_data_threshold[0]
        param10 = data_interpolation[0]
        param11 = signal[0]

        
        script_path = r'.\pages\scripts\getRhythm.py'

        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3} {param4} {param5} {param6} {param7} {param8} {param9} {param10} {param11}'
            print(command)
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3} {param4} {param5} {param6} {param7} {param8} {param9} {param10} {param11}'
            print(command)

        process = subprocess.Popen(command, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return False, False, '', False, now
    except Exception as e:
        print(e)
        return False, True, str(e), True, ''
    pass

    

        
@callback(
    Output('confirm-dialog-rhythmic', 'displayed', allow_duplicate=True),
    Output('error-gen-dialog-rhythmic', 'displayed', allow_duplicate=True),
    Output('error-gen-dialog-rhythmic', 'message', allow_duplicate=True),
    Output('interval-rhythmic', 'disabled', allow_duplicate=True),
    Output('start-time-rhythmic', 'data', allow_duplicate=True),
    Input({'type': 'run-selected-rhythmic-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-rhythmic-data-table', 'index': ALL}, 'selectedRows'),
    State('usenname-rhythmic', 'value'),
    State('project-selection-dropdown-FitBit-rhythmic', 'value'),
    State({'type': 'include-dates-checkbox', 'index': ALL}, 'value'),
    State({'type': 'include-dst-checkbox', 'index': ALL}, 'value'),
    State({'type': 'window-size-slider', 'index': ALL}, 'value'),
    State({'type': 'increment-size-slider', 'index': ALL}, 'value'),
    State({'type': 'downsample-rate-slider', 'index': ALL}, 'value'),
    State({'type': 'missing-data-threshold-slider', 'index': ALL}, 'value'),
    State({'type': 'data-interpolation-checkbox', 'index': ALL}, 'value'),
    State({'type': 'signal-dropdown', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def run_preprocessings(n_clicks, raw_data, username, project, include_not_in_il, include_dst, window_size, increment_size, downsample_rate, missing_data_threshold, data_interpolation, signal):
    if n_clicks == 0:
        raise PreventUpdate
    
    if n_clicks == []:
        return False, False, '', False, ''
    
    if n_clicks[0] == 0:
        return False, False, '', False, ''
    
    print(f'n_clicks: {n_clicks}')

    df = pl.DataFrame(raw_data[0])
    print(f'Raw data: {df}')

    if not df['run'].any():
        return False, True, 'No subjects selected to run Rhythmic', True, ''
    

    df = (
        df
        .filter(
            pl.col('n_files') > 0
        )
    )

    df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_folders_rhythmic.parquet')

    if username == '':
        return False, True, 'Please enter your name before running the rhythmic analysis', True, ''
    # ls 
        

    try:

        param = project
        param2 = now
        param3 = username
        param4 = include_not_in_il[0]
        param5 = include_dst[0]
        param6 = window_size[0]
        param7 = increment_size[0]
        param8 = downsample_rate[0]
        param9 = missing_data_threshold[0]
        param10 = data_interpolation[0]
        param11 = signal[0]

        
        script_path = r'.\pages\scripts\getRhythm.py'

        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3} {param4} {param5} {param6} {param7} {param8} {param9} {param10} {param11}'
            print(command)
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3} {param4} {param5} {param6} {param7} {param8} {param9} {param10} {param11}'
            print(command)

        process = subprocess.Popen(command, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return False, False, '', False, now
    except Exception as e:
        print(e)
        return False, True, str(e), True, ''
    pass
                
@callback(
    Output('interval-rhythmic', 'disabled', allow_duplicate=True),
    Output('confirm-dialog-rhythmic', 'displayed', allow_duplicate=True),
    Output('confirm-dialog-rhythmic', 'message', allow_duplicate=True),
    Input('interval-rhythmic', 'n_intervals'),
    State('start-time-rhythmic', 'data'),
    prevent_initial_call=True   
)
def check_file_generation(n_intervals, start_time):
    if n_intervals == 0:
        raise PreventUpdate
    
    print(f'Checking file generation: {n_intervals}')
    if n_intervals > 0:
        print(f'Checking file generation: {n_intervals}')
        # C:\Users\PsyLab-6028\Desktop\FitbitDash\logs\sleepAllSubjectsScript_2024-12-11_18-35-35.log
        log_path = Path(rf'.\logs\Rhythmic_{start_time}.log')
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
    
    

     
@callback(
    Output('outputs-rhythmic', 'children'),
    Input('load-outputs-button-rhythmic', 'n_clicks'),
    State('project-selection-dropdown-FitBit-rhythmic', 'value')
)
def load_outputs(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate

    project_path = Path(Pconfigs[project])

    outputs_path = project_path.joinpath('Outputs')

    if not os.path.exists(outputs_path):
        return html.Div('No outputs yet, please generate at least one file')
    
    outputs_folders = [folder for folder in os.listdir(outputs_path) if os.path.isdir(outputs_path.joinpath(folder))]

    outputs_folders.sort()

    
    dropdown_options = [{'label': folder, 'value': folder} for folder in outputs_folders]

    

    return html.Div([
        dcc.Dropdown(
            id={
                'type': 'rhythmic-outputs-dropdown',
                'index': 1
            },
            options=dropdown_options,
            value=outputs_folders[0]
        ),
        dbc.Button('Load', id={
            'type': 'load-rhythmic-outputs-button',
            'index': 1
        }, n_clicks=0, color='primary'),

        dbc.Container(id={
            'type': 'rhythmic-outputs-container',
            'index': 1
        })
    ])

@callback(
    Output({'type': 'rhythmic-outputs-container', 'index': ALL}, 'children'),
    Input({'type': 'load-rhythmic-outputs-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'rhythmic-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-rhythmic', 'value')
)
def load_output(n_clicks, selected_folders, project):
    print(f'load_output: {n_clicks}')
    if n_clicks[0] == 0:
        raise PreventUpdate
    
    project_path = Path(Pconfigs[project])

    outputs_path = project_path.joinpath('Outputs')

    if not os.path.exists(outputs_path):
        return html.Div('No outputs yet, please generate at least one file')
    
    outputs_folders = [folder for folder in os.listdir(outputs_path) if os.path.isdir(outputs_path.joinpath(folder))]

    outputs_folders.sort()

    if not selected_folders:
        raise PreventUpdate
    
    selected_folder = selected_folders[0]

    selected_folder_path = outputs_path.joinpath(selected_folder)

    folder_name = selected_folder

    files_dates = {}

    for file in os.listdir(selected_folder_path):
        files_dates[file] = os.path.getmtime(selected_folder_path.joinpath(file))

    files_dates = {k: v for k, v in sorted(files_dates.items(), key=lambda item: item[1], reverse=True)}

    files = list(files_dates.keys())
    files = [file for file in files if file.endswith('.csv') or file.endswith('.parquet')]

    files.sort()

    files_df = pd.DataFrame({
        'File': files,
        'Creation Date': [datetime.datetime.fromtimestamp(files_dates[file]).strftime('%Y-%m-%d %H:%M:%S') for file in files]
    })

    rows = files_df.to_dict('records')

    columns_def = [
        {'headerName': 'File', 'field': 'File', 'sortable': True, 'filter': True, 'checkboxSelection': True},
        {'headerName': 'Creation Date', 'field': 'Creation Date', 'sortable': True, 'filter': True}
    ]

    files_table = dag.AgGrid(
        id={
            'type': 'rhythmic-files-table',
            'index': 1
        },
        columnDefs=columns_def,
        rowData=rows,
        defaultColDef={
            'resizable': True,
            'sortable': True,
            'filter': True
        },
        columnSize = 'autoSize',
        dashGridOptions={'pagination': True, 'paginationPageSize': 10, 'rowSelection': 'multiple'}
    )

    show_button = dbc.Button('Preview selected file', id={
        'type': 'show-preview-button-rhythmic',
        'index': 1
    }, n_clicks=0, color='primary')

    

    file_cotent = html.Div(id={
        'type': 'file-content-rhythmic',
        'index': 1
    })

    return [dbc.Row(
        dbc.Col([
        html.Hr(),
        files_table,
        show_button,
        file_cotent
    ]))]



@callback(
    Output('output-file-path-rhythmic', 'data'),
    Output('outputs-dropdowns-rhythmic', 'children'),
    Input({'type': 'show-preview-button-rhythmic', 'index': ALL}, 'n_clicks'),
    State({'type': 'rhythmic-files-table', 'index': ALL}, 'selectedRows'),
    State({'type': 'rhythmic-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-rhythmic', 'value')
)
def show_file(n_clicks, selected_rows, selected_folder, project):
    if n_clicks[0] == 0:
        raise PreventUpdate

    project_path = Path(Pconfigs[project])
    outputs_path = project_path.joinpath('Outputs')

    if not os.path.exists(outputs_path):
        return html.Div('No outputs yet'), None, []

    if not selected_folder:
        raise PreventUpdate
    
    selected_folder = selected_folder[0]
    selected_folder_path = outputs_path.joinpath(selected_folder)

    if not selected_rows or len(selected_rows[0]) == 0:
        raise PreventUpdate
    
    selected_file = selected_rows[0][0]['File']
    selected_file_path = selected_folder_path.joinpath(selected_file)

    
    if selected_file_path.suffix == '.csv':
        df_columns = pl.read_csv(selected_file_path).columns
        df_size = len(pl.read_csv(selected_file_path))
    elif selected_file_path.suffix == '.parquet':
        df_columns = pl.read_parquet(selected_file_path).columns
        df_size = len(pl.read_parquet(selected_file_path))
    else:
        return html.Div('File type not supported'), None, []
    
    column_options = [{'label': c, 'value': c} for c in df_columns]


    # # if Id column is present,  let the user to select specific values from the unique values in the column
    if 'Id' in df_columns:
        if selected_file_path.suffix == '.csv':
            df = pl.read_csv(selected_file_path).select('Id').unique()
        elif selected_file_path.suffix == '.parquet':
            df = pl.read_parquet(selected_file_path).select('Id').unique()
        unique_values = df['Id'].to_list()

        if len(unique_values) <= 1:
            id_options = [{'label': 'No unique values', 'value': 'No unique values'}]
            id_dropdown_disabled = True
        else:
            if df_size < 2000:
                id_options = [{'label': str(val), 'value': str(val)} for val in unique_values + ['All']]
                id_dropdown_disabled = False
            else:
                id_options = [{'label': str(val), 'value': str(val)} for val in unique_values]
                id_dropdown_disabled = False
    else:
        id_options = [{'label': c, 'value': c} for c in df_columns]
        id_dropdown_disabled = True

    # Store the data in JSON form for plotting
    # data_json = df.to_dict(orient='records')
    # column_options = [{'label': c, 'value': c} for c in df.columns]

    select_id_dropdown = dcc.Dropdown(
        id={
            'type': 'select-id-Dropdown-rhythmic',
            'index': 1
        },
        options=id_options,
        placeholder='Select an ID',
        disabled=id_dropdown_disabled
    )

    column_selection_dropdown = dcc.Dropdown(
        id={
            'type': 'column-selection-dropdown-rhythmic',
            'index': 1
        },
        options=column_options,
        placeholder='Select a column'
    )

    load_content_button = dbc.Button('Load content', id={
        'type': 'load-content-button-rhythmic',
        'index': 1
    }, n_clicks=0, color='primary')

    return str(selected_file_path), [column_selection_dropdown, select_id_dropdown, load_content_button]

@callback(
    Output('outputs-plots-rhythmic', 'children'),
    Output({'type': 'file-content-rhythmic', 'index': ALL}, 'children'),
    Input({'type': 'load-content-button-rhythmic', 'index': ALL}, 'n_clicks'),
    State({'type': 'select-id-Dropdown-rhythmic', 'index': ALL}, 'value'),
    State({'type': 'column-selection-dropdown-rhythmic', 'index': ALL}, 'value'),
    State('output-file-path-rhythmic', 'data'),
    State({'type': 'select-id-Dropdown-rhythmic', 'index': ALL}, 'disabled')
)
def update_column_distribution(n_clicks, selected_id, selected_column, selected_file, id_disabled):
    if n_clicks[0] == 0:
        raise PreventUpdate
    print(f'selected_id: {selected_id}')
    print(f'selected_column: {selected_column}')
    print(f'selected_file: {selected_file}')
    print(f'id_disabled: {id_disabled}')
    id_disabled = id_disabled[0]
    selected_column = selected_column[0]
    selected_id = selected_id[0]
    selected_file = Path(rf'{selected_file}')
    if selected_file.suffix == '.csv':
        if id_disabled:
            df = pl.read_csv(selected_file).select(selected_column)
            df_table = pl.read_csv(selected_file)
        else:
            if selected_id == 'All':
                df = pl.read_csv(selected_file).select(selected_column)
                df_table = pl.read_csv(selected_file)
            else:
                df = pl.read_csv(selected_file).filter(pl.col('Id') == selected_id).select(selected_column)
                df_table = pl.read_csv(selected_file).filter(pl.col('Id') == selected_id)
    elif selected_file.suffix == '.parquet':
        if id_disabled:
            df = pl.read_parquet(selected_file).select(selected_column)
            df_table = pl.read_parquet(selected_file)
        else:
            if selected_id == 'All':
                df = pl.read_parquet(selected_file).select(selected_column)
                df_table = pl.read_parquet(selected_file)
            else:
                df = pl.read_parquet(selected_file).filter(pl.col('Id') == selected_id).select(selected_column)
                df_table = pl.read_parquet(selected_file).filter(pl.col('Id') == selected_id)
    else:
        raise PreventUpdate

    df = df.to_pandas()
    if len(df) > 2000:
        df_table = df_table.to_pandas().head(2000)
        note = 'Note: Only first 2000 rows are shown'
    else:
        df_table = df_table.to_pandas()
    rows = df_table.to_dict('records')
    columns_def = [{'headerName': col, 'field': col, 'sortable': True, 'filter': True} for col in df_table.columns]

    table = dag.AgGrid(
        id={
            'type': 'rhythmic-file-table',
            'index': 1
        },
        columnDefs=columns_def,
        rowData=rows,
        defaultColDef={
            'resizable': True,
            'sortable': True,
            'filter': True
        },
        columnSize = 'autoSize',
        dashGridOptions={'pagination': True, 'paginationPageSize': 10}
    )


    # Check if column is numeric
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        # Create a figure with a line plot and a histogram
        fig_line = go.Figure()
        # Line plot of column values over index
        fig_line.add_trace(go.Scatter(y=df[selected_column], mode='lines', name='Line Plot'))
        # Histogram of the column values
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[selected_column], name='Histogram'))
        fig.update_layout(title=f"Distribution of {selected_column}", barmode='overlay')
        fig.update_traces(opacity=0.75)

        return html.Div([dcc.Graph(figure=fig), dcc.Graph(figure=fig_line)]), [table]
    else:
        fig_line = go.Figure()

    # For non-numeric, show a bar chart of value counts
    counts = df[selected_column].value_counts()
    fig = go.Figure([go.Bar(x=counts.index.astype(str), y=counts.values, name='Count')])
    fig.update_layout(title=f"Value Counts of {selected_column}")
    return html.Div(dcc.Graph(figure=fig)), [table]