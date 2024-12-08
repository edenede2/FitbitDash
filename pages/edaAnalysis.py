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





dash.register_page(__name__, name='EDA', order=11)

pages = {}

Pconfigs = json.load(open(r".\pages\Pconfigs\paths.json", "r"))


for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Preprocessing EDA Data"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-EDA',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-EDA', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-EDA',
                        placeholder='user_name',
                        value=''
                    )
                ]),

                html.Hr()
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Scan data folder',
                        id='scan-folder-button-EDA',
                        n_clicks=0,
                        color='primary'
                    )
                ]),
            ]),
        ]),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div(id='raw-data-subjects-container-EDA')
                ]),
                dbc.Col([
                    html.Div(id='available-data-container-EDA')
                ])
            ])
        ]),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-EDA',
                                message='File Generation Started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-EDA',
                        message='Error generating file'
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button('Load outputs', id='load-outputs-button-EDA', n_clicks=0, color='primary')
                ])
            ]),
            dbc.Row([
                    dbc.Col([
                        html.Div(id='outputs-EDA')
                    ])
                ])
])

@callback(
    Output('raw-data-subjects-container-EDA', 'children'),
    Input('load-fitbit-button-EDA', 'n_clicks'),
    State('project-selection-dropdown-FitBit-EDA', 'value')
)
def load_raw_data(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate

    data_path = Pconfigs[project]
    print(f'Project proccesed data path: {project}')

    processed_data_path = Path(data_path) / 'Data'
    print(f'Processed data path: {processed_data_path}')

    if not processed_data_path.exists():
        return html.Div('No processed data found, please go to the data validation page and check for eda files')
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

        files_path = processed_data_path / subject / 'FITBIT' / 'Stress'

        eda_data_files = []
        eda_sessions_files = []
        
        for file in files_path.iterdir():
            print(f'File: {file}')
            if file.name.startswith('mindfulness_eda_data_sessions'):
                eda_data_files.append(file.name)
            elif file.name.startswith('Mindfulness Eda Data Sessions'):
                eda_data_files.append(file.name)
            elif file.name.startswith('mindfulness_sessions'):
                eda_sessions_files.append(file.name)
            elif file.name.startswith('Mindfulness Sessions'):
                eda_sessions_files.append(file.name)
            

        # df_height = np.max([len(hr_files), len(steps_files)])
        # add the name of the files to the dataframe
        subject_found_files_df = (
            pl.DataFrame({
                'subject': pl.Series([subject.name]),
                'eda_data_files': pl.Series([len(eda_data_files)]),
                'eda_sessions_files': pl.Series([len(eda_sessions_files)])
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
        {'field': 'eda_data_files', 'headerName': 'eda_data_files', 'sortable': True, 'filter': True},
        {'field': 'eda_sessions_files', 'headerName': 'eda_sessions_files', 'sortable': True, 'filter': True},
        {'field': 'run', 'headerName': 'run', 'sortable': True, 'filter': True, 'editable': True}
    ]

    rows = raw_data_df.to_pandas().to_dict('records')
    print(f'Raw data df: {raw_data_df}')

    grid = dag.AgGrid(
        id={
            'type': 'raw-EDA-data-table',
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

    run_button = dbc.Button('Run EDA Proccesing', id={
        'type': 'run-EDA-button',
        'index': 1
    }, n_clicks=0, color='success')

    show_button = dbc.Button('Show available data', id={
        'type': 'show-available-EDA-data-button',
        'index': 1
    }, n_clicks=0, color='warning')


    return [
        html.H4('Raw data:'),
        grid,
        run_button,
        show_button
    ]


        
@callback(
    Output('available-data-container-EDA', 'children'),
    Input({'type': 'show-available-EDA-data-button','index': ALL}, 'n_clicks'),
    State({'type': 'raw-EDA-data-table', 'index': ALL}, 'selectedRows'),
    State('project-selection-dropdown-FitBit-EDA', 'value')
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
        
    subject_path = processed_data_path / selected_subject / 'FITBIT' / 'Stress'
    print(f'Selected subject: {selected_subject}')

    eda_data_files = []
    eda_sessions_files = []

    for file in subject_path.iterdir():
        print(f'File: {file}')
        if file.name.startswith('mindfulness_eda_data_sessions'):
            eda_data_files.append(file.name)
        elif file.name.startswith('Mindfulness Eda Data Sessions'):
            eda_data_files.append(file.name)
        elif file.name.startswith('mindfulness_sessions'):
            eda_sessions_files.append(file.name)
        elif file.name.startswith('Mindfulness Sessions'):
            eda_sessions_files.append(file.name)
    

    eda_data_files = sorted(eda_data_files)
    eda_sessions_files = sorted(eda_sessions_files)

    eda_data_files_df = pl.DataFrame({
        'eda_data_files': pl.Series(eda_data_files)
    })

    eda_sessions_files_files_df = pl.DataFrame({
        'eda_sessions_files': pl.Series(eda_sessions_files)
    })

    defcolumns_eda_data = [
        {'field': 'eda_data_files', 'headerName': 'eda_data_files', 'sortable': True, 'filter': True}
    ]

    defcolumns_eda_sesions = [
        {'field': 'eda_sessions_files', 'headerName': 'eda_sessions_files', 'sortable': True, 'filter': True}
    ]

    rows_eda_data = eda_data_files_df.to_pandas().to_dict('records')

    grid_eda_data = dag.AgGrid(
        id={
            'type': 'eda-data-files-table',
            'index': 1
        },
        columnDefs=defcolumns_eda_data,
        rowData=rows_eda_data,
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

    rows_eda_sessions = eda_sessions_files_files_df.to_pandas().to_dict('records')

    grid_eda_sessions = dag.AgGrid(
        id={
            'type': 'eda-sessions-files-table',
            'index': 1
        },
        columnDefs=defcolumns_eda_sesions,
        rowData=rows_eda_sessions,
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
        html.H5('mindfulness_eda_data_sessions files:'),
        grid_eda_data,
        html.H5('mindfulness_sessions files:'),
        grid_eda_sessions
    ]

        
@callback(
    Output('confirm-dialog-EDA', 'displayed'),
    Output('error-gen-dialog-EDA', 'displayed'),
    Output('error-gen-dialog-EDA', 'message'),
    Input({'type': 'run-EDA-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-EDA-data-table', 'index': ALL}, 'rowData'),
    State('usenname-EDA', 'value'),
    State('project-selection-dropdown-FitBit-EDA', 'value')
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
        return False, True, 'No subjects selected to run EDA'
    
    df = (
        df
        .filter(
            pl.col('eda_data_files') > 0,
            pl.col('eda_sessions_files') > 0,
        )
    )
    
    df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_folders_EDA.parquet')

    if username == '':
        return False, True, 'Please enter your name before running the EDA'
        

    try:

        param = project
        param2 = now
        param3 = username
        
        script_path = r'.\pages\scripts\getEDA.py'

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
    Output('outputs-EDA', 'children'),
    Input('load-outputs-button-EDA', 'n_clicks'),
    State('project-selection-dropdown-FitBit-EDA', 'value')
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
                'type': 'EDA-outputs-dropdown',
                'index': 1
            },
            options=dropdown_options,
            value=outputs_folders[0]
        ),
        dbc.Button('Load', id={
            'type': 'load-EDA-outputs-button',
            'index': 1
        }, n_clicks=0, color='primary'),

        html.Div(id={
            'type': 'EDA-outputs-container',
            'index': 1
        })
    ])

@callback(
    Output({'type': 'EDA-outputs-container', 'index': ALL}, 'children'),
    Input({'type': 'load-EDA-outputs-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'EDA-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-EDA', 'value')
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
            'type': 'sleep-files-table',
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
        'type': 'show-preview-button-EDA',
        'index': 1
    }, n_clicks=0, color='primary')

    file_cotent = html.Div(id={
        'type': 'file-content-EDA',
        'index': 1
    })

    return [html.Div([
        files_table,
        show_button,
        file_cotent
    ])]


@callback(
    Output({'type': 'file-content-EDA', 'index': ALL}, 'children'),
    Input({'type': 'show-preview-button-EDA', 'index': ALL}, 'n_clicks'),
    State({'type': 'EDA-files-table', 'index': ALL}, 'selectedRows'),
    State({'type': 'EDA-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-EDA', 'value')
)
def show_file(n_clicks, selected_rows, selected_folder, project):
    print(f'preview_output: {n_clicks}')
    if n_clicks[0] == 0:
        raise PreventUpdate

    print(selected_rows)

    project_path = Path(Pconfigs[project])

    outputs_path = project_path.joinpath('Outputs')

    if not os.path.exists(outputs_path):
        return html.Div('No outputs yet, please generate at least one file')
    
    outputs_folders = [folder for folder in os.listdir(outputs_path) if os.path.isdir(outputs_path.joinpath(folder))]

    outputs_folders.sort()

    if not selected_folder:
        raise PreventUpdate
    
    selected_folder = selected_folder[0]

    selected_folder_path = outputs_path.joinpath(selected_folder)

    folder_name = selected_folder

    if not selected_rows:
        raise PreventUpdate
    
    selected_file = selected_rows[0][0]['File']

    selected_file_path = selected_folder_path.joinpath(selected_file)

    if selected_file_path.suffix == '.csv':
        df = pd.read_csv(selected_file_path)

        note = ''
        if len(df) > 2000:
            df = df.head(2000)
            note = 'Note: Only first 2000 rows are shown'

        rows = df.to_dict('records')

        columns_def = [{'headerName': col, 'field': col, 'sortable': True, 'filter': True} for col in df.columns]

        table = dag.AgGrid(
            id={
                'type': 'EDA-file-table',
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

        return [html.Div(
            [note,
            table]
        )]
    
    elif selected_file_path.suffix == '.parquet':

        df = pd.read_parquet(selected_file_path)

        note = ''

        if len(df) > 2000:
            df = df.head(2000)
            note = 'Note: Only first 2000 rows are shown'

        rows = df.to_dict('records')

        columns_def = [{'headerName': col, 'field': col, 'sortable': True, 'filter': True} for col in df.columns]

        table = dag.AgGrid(
            id={
                'type': 'EDA-file-table',
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

        return html.Div([
            note,
            table
        ])
    
    else:
        return html.Div('File type not supported')
    
