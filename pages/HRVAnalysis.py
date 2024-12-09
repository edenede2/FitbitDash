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





dash.register_page(__name__, name='HRV', order=10)

pages = {}

Pconfigs = json.load(open(r".\pages\Pconfigs\paths.json", "r"))


for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dcc.Store(id='file-data-store-HRV', storage_type='memory'),
        dcc.Store(id='output-file-path-HRV', storage_type='memory'),

        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Preprocessing HRV Data"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-HRV',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-HRV', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-HRV',
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
                        id='scan-folder-button-HRV',
                        n_clicks=0,
                        color='primary'
                    )
                ]),
            ]),
        ]),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div(id='raw-data-subjects-container-HRV')
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='available-data-container-HRV'),
                    
                ])
            ])
        ]),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-HRV',
                                message='File Generation Started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-HRV',
                        message='Error generating file'
                    )
                ]),
            ]),
            
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Load outputs', id='load-outputs-button-HRV', n_clicks=0, color='primary')
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='outputs-HRV'),
                    ]),
                    dbc.Col([
                        html.Div(id='outputs-dropdowns-HRV'),
                    ])
                ])
            ]),
            html.Div([
                html.Hr(),
                html.Div(id='outputs-plots-HRV'),
            ], style={'margin-top': '50px'})
])

@callback(
    Output('raw-data-subjects-container-HRV', 'children'),
    Input('load-fitbit-button-HRV', 'n_clicks'),
    State('project-selection-dropdown-FitBit-HRV', 'value')
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

        files_path = processed_data_path / subject / 'FITBIT' / 'Sleep'

        sleep_data_files = []
        computed_temperature_files = []
        daily_respiratory_rate_summary_files = []
        heart_rate_variability_details_files = []
        device_temperature_files = []
        
        for file in files_path.iterdir():
            print(f'File: {file}')
            if file.name.startswith('sleep-'):
                sleep_data_files.append(file.name)
            elif file.name.startswith('Computed Temperature'):
                computed_temperature_files.append(file.name)
            elif file.name.startswith('Daily Respiratory Rate Summary'):
                daily_respiratory_rate_summary_files.append(file.name)
            elif file.name.startswith('Heart Rate Variability Details'):
                heart_rate_variability_details_files.append(file.name)
            elif file.name.startswith('Device Temperature'):
                device_temperature_files.append(file.name)



        # df_height = np.max([len(hr_files), len(steps_files)])
        # add the name of the files to the dataframe
        subject_found_files_df = (
            pl.DataFrame({
                'subject': pl.Series([subject.name]),
                'sleep_data_files': pl.Series([len(sleep_data_files)]),
                'computed_temperature_files': pl.Series([len(computed_temperature_files)]),
                'daily_respiratory_rate_summary_files': pl.Series([len(daily_respiratory_rate_summary_files)]),
                'heart_rate_variability_details_files': pl.Series([len(heart_rate_variability_details_files)]),
                'device_temperature_files': pl.Series([len(device_temperature_files)])
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
        {'field': 'sleep_data_files', 'headerName': 'sleep_data_files', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'computed_temperature_files', 'headerName': 'computed_temperature_files', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'daily_respiratory_rate_summary_files', 'headerName': 'daily_respiratory_rate_summary_files', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'heart_rate_variability_details_files', 'headerName': 'heart_rate_variability_details_files', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'device_temperature_files', 'headerName': 'device_temperature_files', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'run', 'headerName': 'run', 'sortable': True, 'filter': True, 'editable': True}
    ]

    rows = raw_data_df.to_pandas().to_dict('records')
    print(f'Raw data df: {raw_data_df}')

    grid = dag.AgGrid(
        id={
            'type': 'raw-HRV-data-table',
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

    run_button = dbc.Button('Run HRV Proccesing', id={
        'type': 'run-HRV-button',
        'index': 1
    }, n_clicks=0, color='success')

    show_button = dbc.Button('Show available data', id={
        'type': 'show-available-HRV-data-button',
        'index': 1
    }, n_clicks=0, color='warning')


    return [
        html.H4('Raw data:'),
        grid,
        run_button,
        show_button
    ]


        
@callback(
    Output('available-data-container-HRV', 'children'),
    Input({'type': 'show-available-HRV-data-button','index': ALL}, 'n_clicks'),
    State({'type': 'raw-HRV-data-table', 'index': ALL}, 'selectedRows'),
    State('project-selection-dropdown-FitBit-HRV', 'value')
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
        
    subject_path = processed_data_path / selected_subject / 'FITBIT' / 'Sleep'
    print(f'Selected subject: {selected_subject}')

    sleep_data_files = []
    computed_temperature_files = []
    daily_respiratory_rate_summary_files = []
    heart_rate_variability_details_files = []
    device_temperature_files = []
    
    for file in subject_path.iterdir():
        print(f'File: {file}')
        if file.name.startswith('sleep-'):
            sleep_data_files.append(file.name)
        elif file.name.startswith('Computed Temperature'):
            computed_temperature_files.append(file.name)
        elif file.name.startswith('Daily Respiratory Rate Summary'):
            daily_respiratory_rate_summary_files.append(file.name)
        elif file.name.startswith('Heart Rate Variability Details'):
            heart_rate_variability_details_files.append(file.name)
        elif file.name.startswith('Device Temperature'):
            device_temperature_files.append(file.name)
        

    sleep_data_files = sorted(sleep_data_files)
    computed_temperature_files = sorted(computed_temperature_files)
    daily_respiratory_rate_summary_files = sorted(daily_respiratory_rate_summary_files)
    heart_rate_variability_details_files = sorted(heart_rate_variability_details_files)
    device_temperature_files = sorted(device_temperature_files)


    sleep_data_files_df = pl.DataFrame({
        'sleep_data_files': pl.Series(sleep_data_files)
    })

    computed_temperature_files_df = pl.DataFrame({
        'computed_temperature_files': pl.Series(computed_temperature_files)
    })

    daily_respiratory_rate_summary_files_df = pl.DataFrame({
        'daily_respiratory_rate_summary_files': pl.Series(daily_respiratory_rate_summary_files)
    })

    heart_rate_variability_details_files_df = pl.DataFrame({
        'heart_rate_variability_details_files': pl.Series(heart_rate_variability_details_files)
    })

    device_temperature_files_df = pl.DataFrame({
        'device_temperature_files': pl.Series(device_temperature_files)
    })


    defcolumns_sleep_data = [
        {'field': 'sleep_data_files', 'headerName': 'sleep_data_files', 'sortable': True, 'filter': True}
    ]

    defcolumns_computed_temperature = [
        {'field': 'computed_temperature_files', 'headerName': 'computed_temperature_files', 'sortable': True, 'filter': True}
    ]

    defcolumns_daily_respiratory_rate_summary = [
        {'field': 'daily_respiratory_rate_summary_files', 'headerName': 'daily_respiratory_rate_summary_files', 'sortable': True, 'filter': True}
    ]

    defcolumns_heart_rate_variability_details = [
        {'field': 'heart_rate_variability_details_files', 'headerName': 'heart_rate_variability_details_files', 'sortable': True, 'filter': True}
    ]

    defcolumns_device_temperature = [
        {'field': 'device_temperature_files', 'headerName': 'device_temperature_files', 'sortable': True, 'filter': True}
    ]

    rows_sleep_data = sleep_data_files_df.to_pandas().to_dict('records')

    grid_sleep_data = dag.AgGrid(
        id={
            'type': 'sleep-data-files-table',
            'index': 1
        },
        columnDefs=defcolumns_sleep_data,
        rowData=rows_sleep_data,
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

    rows_computed_temperature = computed_temperature_files_df.to_pandas().to_dict('records')

    grid_computed_temperature = dag.AgGrid(
        id={
            'type': 'computed-temperature-files-table',
            'index': 1
        },
        columnDefs=defcolumns_computed_temperature,
        rowData=rows_computed_temperature,
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

    rows_daily_respiratory_rate_summary = daily_respiratory_rate_summary_files_df.to_pandas().to_dict('records')

    grid_daily_respiratory_rate_summary = dag.AgGrid(
        id={
            'type': 'daily-respiratory-rate-summary-files-table',
            'index': 1
        },
        columnDefs=defcolumns_daily_respiratory_rate_summary,
        rowData=rows_daily_respiratory_rate_summary,
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

    rows_heart_rate_variability_details = heart_rate_variability_details_files_df.to_pandas().to_dict('records')

    grid_heart_rate_variability_details = dag.AgGrid(
        id={
            'type': 'heart-rate-variability-details-files-table',
            'index': 1
        },
        columnDefs=defcolumns_heart_rate_variability_details,
        rowData=rows_heart_rate_variability_details,
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

    rows_device_temperature = device_temperature_files_df.to_pandas().to_dict('records')

    grid_device_temperature = dag.AgGrid(
        id={
            'type': 'device-temperature-files-table',
            'index': 1
        },
        columnDefs=defcolumns_device_temperature,
        rowData=rows_device_temperature,
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
        html.H5('Sleep data files:'),
        grid_sleep_data,
        html.H5('Computed Temperature files:'),
        grid_computed_temperature,
        html.H5('Daily Respiratory Rate Summary files:'),
        grid_daily_respiratory_rate_summary,
        html.H5('Heart Rate Variability Details files:'),
        grid_heart_rate_variability_details,
        html.H5('Device Temperature files:'),
        grid_device_temperature
    ]



        
@callback(
    Output('confirm-dialog-HRV', 'displayed'),
    Output('error-gen-dialog-HRV', 'displayed'),
    Output('error-gen-dialog-HRV', 'message'),
    Input({'type': 'run-HRV-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-HRV-data-table', 'index': ALL}, 'rowData'),
    State('usenname-HRV', 'value'),
    State('project-selection-dropdown-FitBit-HRV', 'value')
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
        return False, True, 'No subjects selected to run HRV'
    
    df = (
        df
        .filter(
            pl.col('sleep_data_files') > 0,
            pl.col('computed_temperature_files') > 0,
            pl.col('daily_respiratory_rate_summary_files') > 0,
            pl.col('heart_rate_variability_details_files') > 0,
            pl.col('device_temperature_files') > 0
        )
    )
    df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_folders_HRV.parquet')

    if username == '':
        return False, True, 'Please enter your name before running the HRV processing script'
        

    try:

        param = project
        param2 = now
        param3 = username
        script_path = r'.\pages\scripts\getHRV.py'
            
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
    Output('outputs-HRV', 'children'),
    Input('load-outputs-button-HRV', 'n_clicks'),
    State('project-selection-dropdown-FitBit-HRV', 'value')
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
                'type': 'HRV-outputs-dropdown',
                'index': 1
            },
            options=dropdown_options,
            value=outputs_folders[0]
        ),
        dbc.Button('Load', id={
            'type': 'load-HRV-outputs-button',
            'index': 1
        }, n_clicks=0, color='primary'),

        dbc.Container(id={
            'type': 'HRV-outputs-container',
            'index': 1
        })
    ])

@callback(
    Output({'type': 'HRV-outputs-container', 'index': ALL}, 'children'),
    Input({'type': 'load-HRV-outputs-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'HRV-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-HRV', 'value')
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
            'type': 'HRV-files-table',
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
        'type': 'show-preview-button-HRV',
        'index': 1
    }, n_clicks=0, color='primary')

    

    file_cotent = html.Div(id={
        'type': 'file-content-HRV',
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
    Output('output-file-path-HRV', 'data'),
    Output('outputs-dropdowns-HRV', 'children'),
    Input({'type': 'show-preview-button-HRV', 'index': ALL}, 'n_clicks'),
    State({'type': 'HRV-files-table', 'index': ALL}, 'selectedRows'),
    State({'type': 'HRV-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-HRV', 'value')
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
            'type': 'select-id-Dropdown-HRV',
            'index': 1
        },
        options=id_options,
        placeholder='Select an ID',
        disabled=id_dropdown_disabled
    )

    column_selection_dropdown = dcc.Dropdown(
        id={
            'type': 'column-selection-dropdown-HRV',
            'index': 1
        },
        options=column_options,
        placeholder='Select a column'
    )

    load_content_button = dbc.Button('Load content', id={
        'type': 'load-content-button-HRV',
        'index': 1
    }, n_clicks=0, color='primary')

    return str(selected_file_path), [column_selection_dropdown, select_id_dropdown, load_content_button]

@callback(
    Output('outputs-plots-HRV', 'children'),
    Output({'type': 'file-content-HRV', 'index': ALL}, 'children'),
    Input({'type': 'load-content-button-HRV', 'index': ALL}, 'n_clicks'),
    State({'type': 'select-id-Dropdown-HRV', 'index': ALL}, 'value'),
    State({'type': 'column-selection-dropdown-HRV', 'index': ALL}, 'value'),
    State('output-file-path-HRV', 'data'),
    State({'type': 'select-id-Dropdown-HRV', 'index': ALL}, 'disabled')
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
            'type': 'HRV-file-table',
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
        