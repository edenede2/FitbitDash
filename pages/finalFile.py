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





dash.register_page(__name__, name='Final File', order=12)

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
                    html.H1("Generate Final File"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-Final',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-Final', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-Final',
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
                    html.Div(id='raw-data-subjects-container-Final')
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='available-data-container-Final')
                ])
            ])
        ]),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-Final',
                                message='File Generation Started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-Final',
                        message='Error generating file'
                    )
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Load outputs', id='load-outputs-button-Final', n_clicks=0, color='primary')
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='outputs-Final')
                    ])
                ])
            ])
])

@callback(
    Output('raw-data-subjects-container-Final', 'children'),
    Input('load-fitbit-button-Final', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Final', 'value')
)
def load_raw_data(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate

    data_path = Pconfigs[project]
    print(f'Project proccesed data path: {project}')

    processed_outputs_path = Path(data_path) / 'Outputs' / 'Aggregated Output'
    print(f'Processed Output path: {processed_outputs_path}')

    if not processed_outputs_path.exists():
        return html.Div('No processed subjects found, please generate at least one aggregated output')
        # print('No processed data found, please go to the data validation page')

    if (processed_outputs_path / 'Sleep Daily Summary Full Week.csv').exists():
        sleep_daily_summary_full_week_df = pl.read_csv(processed_outputs_path / 'Sleep Daily Summary Full Week.csv')
    else:
        sleep_daily_summary_full_week_df = pl.DataFrame({
            'Id': pl.Series(''),
        })

    if (processed_outputs_path / 'Full Week Summary of Heart Rate Metrics By Activity.csv').exists():
        full_week_heart_rate_of_sleep_summary_means_df = pl.read_csv(processed_outputs_path / 'Full Week Summary of Heart Rate Metrics By Activity.csv')
    else:
        full_week_heart_rate_of_sleep_summary_means_df = pl.DataFrame({
            'Id': pl.Series(''),
        })


    if (processed_outputs_path / 'Summary Of HRV Temperature Respiratory At Sleep.csv').exists():
        HRV_tempe_resp_summary_df = pl.read_csv(processed_outputs_path / 'Summary Of HRV Temperature Respiratory At Sleep.csv')
    else:
        HRV_tempe_resp_summary_df = pl.DataFrame({
            'Id': pl.Series(''),
        })

    if (processed_outputs_path / 'EDA Summary.csv').exists():
        mindfulness_summary_df = pl.read_csv(processed_outputs_path / 'EDA Summary.csv')
    else:
        mindfulness_summary_df = pl.DataFrame({
            'Id': pl.Series(''),
        })


    sleep_daily_summary_full_week_subjects = sleep_daily_summary_full_week_df['Id'].unique().sort().to_list()
    full_week_heart_rate_of_sleep_summary_means_subjects = full_week_heart_rate_of_sleep_summary_means_df['Id'].unique().sort().to_list()
    HRV_tempe_resp_summary_subjects = HRV_tempe_resp_summary_df['Id'].unique().sort().to_list()
    mindfulness_summary_subjects = mindfulness_summary_df['Id'].unique().sort().to_list()

   

    all_subjects = list(set(sleep_daily_summary_full_week_subjects + full_week_heart_rate_of_sleep_summary_means_subjects + HRV_tempe_resp_summary_subjects + mindfulness_summary_subjects))

    raw_data_df = (
            pl.DataFrame({
            'subject': all_subjects,
        })
        .with_columns(
            sleep_daily_summary = pl.when(pl.col('subject').is_in(sleep_daily_summary_full_week_subjects))
            .then(True)
            .otherwise(False),
            full_week_heart_rate_of_sleep_summary_means = pl.when(pl.col('subject').is_in(full_week_heart_rate_of_sleep_summary_means_subjects))
            .then(True)
            .otherwise(False),
            HRV_tempe_resp_summary = pl.when(pl.col('subject').is_in(HRV_tempe_resp_summary_subjects))
            .then(True)
            .otherwise(False),
            mindfulness_summary = pl.when(pl.col('subject').is_in(mindfulness_summary_subjects))
            .then(True)
            .otherwise(False)
        )
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
        {'field': 'sleep_daily_summary', 'headerName': 'sleep_daily_summary', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'full_week_heart_rate_of_sleep_summary_means', 'headerName': 'full_week_heart_rate_of_sleep_summary_means', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'HRV_tempe_resp_summary', 'headerName': 'HRV_tempe_resp_summary', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'mindfulness_summary', 'headerName': 'mindfulness_summary', 'sortable': True, 'filter': True, 'editable': False},
        {'field': 'run', 'headerName': 'run', 'sortable': True, 'filter': True, 'editable': True}
    ]

    rows = raw_data_df.to_pandas().to_dict('records')
    print(f'Raw data df: {raw_data_df}')

    grid = dag.AgGrid(
        id={
            'type': 'raw-Final-data-table',
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

    run_button = dbc.Button('Run Final Proccesing', id={
        'type': 'run-Final-button',
        'index': 1
    }, n_clicks=0, color='success')

    show_button = dbc.Button('Show available data', id={
        'type': 'show-available-Final-data-button',
        'index': 1
    }, n_clicks=0, color='warning')


    return [
        html.H4('Raw data:'),
        grid,
        run_button,
        show_button
    ]


        
@callback(
    Output('available-data-container-Final', 'children'),
    Input({'type': 'show-available-Final-data-button','index': ALL}, 'n_clicks'),
    State({'type': 'raw-Final-data-table', 'index': ALL}, 'selectedRows'),
    State('project-selection-dropdown-FitBit-Final', 'value')
)
def show_available_data(n_clicks, selected_rows, project):
    if n_clicks == 0:
        raise PreventUpdate

    data_path = Pconfigs[project]
    print(f'Project proccesed data path: {project}')

    processed_output_path = Path(data_path) / 'Outputs'
    print(f'Processed output path: {processed_output_path}')

    if not processed_output_path.exists():
        return html.Div('No output path found, please generate at least one aggregated output')
        # print('No processed data found, please go to the data validation page')

    print(f'Selected rows: {selected_rows}')
    selected_subject = [row['subject'] for row in selected_rows[0]]
        
    aggregated_outputs = processed_output_path / 'Aggregated Output' 
    print(f'Selected subject: {selected_subject}')

    if (aggregated_outputs / 'Sleep Daily Summary Full Week.csv').exists():
        sleep_daily_summary_full_week_df = pl.read_csv(aggregated_outputs / 'Sleep Daily Summary Full Week.csv').filter(pl.col('Id').is_in(selected_subject))
    else:
        sleep_daily_summary_full_week_df = pl.DataFrame({
            'Id': pl.Series(''),
        })

    if (aggregated_outputs / 'Full Week Summary of Heart Rate Metrics By Activity.csv').exists():
        full_week_heart_rate_of_sleep_summary_means_df = pl.read_csv(aggregated_outputs / 'Full Week Summary of Heart Rate Metrics By Activity.csv').filter(pl.col('Id').is_in(selected_subject))
    else:
        full_week_heart_rate_of_sleep_summary_means_df = pl.DataFrame({
            'Id': pl.Series(''),
        })

    if (aggregated_outputs / 'Summary Of HRV Temperature Respiratory At Sleep.csv').exists():
        HRV_tempe_resp_summary_df = pl.read_csv(aggregated_outputs / 'Summary Of HRV Temperature Respiratory At Sleep.csv').filter(pl.col('Id').is_in(selected_subject))
    else:
        HRV_tempe_resp_summary_df = pl.DataFrame({
            'Id': pl.Series(''),
        })

    if (aggregated_outputs / 'EDA Summary.csv').exists():
        mindfulness_summary_df = pl.read_csv(aggregated_outputs / 'EDA Summary.csv').filter(pl.col('Id').is_in(selected_subject))
    else:
        mindfulness_summary_df = pl.DataFrame({
            'Id': pl.Series(''),
        })

    defColumnsSleepDailySummaryFullWeekdf = [
        {'field': col, 'headerName': col, 'sortable': True, 'filter': True} for col in sleep_daily_summary_full_week_df.columns
    ]

    defColumnsFullWeekHeartRateOfSleepSummaryMeansdf = [
        {'field': col, 'headerName': col, 'sortable': True, 'filter': True} for col in full_week_heart_rate_of_sleep_summary_means_df.columns
    ]

    defColumnsHRVTempeRespSummarydf = [
        {'field': col, 'headerName': col, 'sortable': True, 'filter': True} for col in HRV_tempe_resp_summary_df.columns
    ]

    defColumnsMindfulnessSummarydf = [
        {'field': col, 'headerName': col, 'sortable': True, 'filter': True} for col in mindfulness_summary_df.columns
    ]

    rows_sleep_daily_summary_full_week = sleep_daily_summary_full_week_df.to_pandas().to_dict('records')

    grid_sleep_daily_summary_full_week = dag.AgGrid(
        id={
            'type': 'sleep-daily-summary-full-week-files-table',
            'index': 1
        },
        columnDefs=defColumnsSleepDailySummaryFullWeekdf,
        rowData=rows_sleep_daily_summary_full_week,
        defaultColDef={
            'resizable': True,
            'editable': True,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 5,
            'rowSelection': 'single',
        }
    )

    rows_full_week_heart_rate_of_sleep_summary_means = full_week_heart_rate_of_sleep_summary_means_df.to_pandas().to_dict('records')

    grid_full_week_heart_rate_of_sleep_summary_means = dag.AgGrid(
        id={
            'type': 'full-week-heart-rate-of-sleep-summary-means-files-table',
            'index': 1
        },
        columnDefs=defColumnsFullWeekHeartRateOfSleepSummaryMeansdf,
        rowData=rows_full_week_heart_rate_of_sleep_summary_means,
        defaultColDef={
            'resizable': True,
            'editable': True,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 5,
            'rowSelection': 'single',
        }
    )

    rows_HRV_tempe_resp_summary = HRV_tempe_resp_summary_df.to_pandas().to_dict('records')

    grid_HRV_tempe_resp_summary = dag.AgGrid(
        id={
            'type': 'HRV-tempe-resp-summary-files-table',
            'index': 1
        },
        columnDefs=defColumnsHRVTempeRespSummarydf,
        rowData=rows_HRV_tempe_resp_summary,
        defaultColDef={
            'resizable': True,
            'editable': True,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 5,
            'rowSelection': 'single',
        }
    )

    rows_mindfulness_summary = mindfulness_summary_df.to_pandas().to_dict('records')

    grid_mindfulness_summary = dag.AgGrid(
        id={
            'type': 'mindfulness-summary-files-table',
            'index': 1
        },
        columnDefs=defColumnsMindfulnessSummarydf,
        rowData=rows_mindfulness_summary,
        defaultColDef={
            'resizable': True,
            'editable': True,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions={
            'pagination': True,
            'paginationPageSize': 5,
            'rowSelection': 'single',
        }
    )

    selected_subject_str = ', '.join(selected_subject)

    return [
        html.H4(f'Available data for {selected_subject_str}:'),
        html.H5('Sleep Daily Summary Full Week files:'),
        grid_sleep_daily_summary_full_week,
        html.H5('Full Week Summary of Heart Rate Metrics By Activity files:'),
        grid_full_week_heart_rate_of_sleep_summary_means,
        html.H5('HRV Temperature Respiratory Summary files:'),
        grid_HRV_tempe_resp_summary,
        html.H5('Mindfulness Summary files:'),
        grid_mindfulness_summary
    ]



        
@callback(
    Output('confirm-dialog-Final', 'displayed'),
    Output('error-gen-dialog-Final', 'displayed'),
    Output('error-gen-dialog-Final', 'message'),
    Input({'type': 'run-Final-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-Final-data-table', 'index': ALL}, 'rowData'),
    State('usenname-Final', 'value'),
    State('project-selection-dropdown-FitBit-Final', 'value')
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
        return False, True, 'No subjects selected to run Final'
    
    df = (
        df
        .filter(
            pl.col('sleep_daily_summary') | pl.col('full_week_heart_rate_of_sleep_summary_means') | pl.col('HRV_tempe_resp_summary') | pl.col('mindfulness_summary')
        )
    )
    df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_folders_Final.parquet')

    if username == '':
        return False, True, 'Please enter your name before running the Final generation script'
        

    try:

        param = project
        param2 = now
        param3 = username
        script_path = r'.\pages\scripts\getFinal.py'

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
    Output('outputs-Final', 'children'),
    Input('load-outputs-button-Final', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Final', 'value')
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
                'type': 'Final-outputs-dropdown',
                'index': 1
            },
            options=dropdown_options,
            value=outputs_folders[0]
        ),
        dbc.Button('Load', id={
            'type': 'load-Final-outputs-button',
            'index': 1
        }, n_clicks=0, color='primary'),

        html.Div(id={
            'type': 'Final-outputs-container',
            'index': 1
        })
    ])

@callback(
    Output({'type': 'Final-outputs-container', 'index': ALL}, 'children'),
    Input({'type': 'load-Final-outputs-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'Final-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-Final', 'value')
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
            'type': 'Final-files-table',
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
        'type': 'show-preview-button-Final',
        'index': 1
    }, n_clicks=0, color='primary')

    file_cotent = html.Div(id={
        'type': 'file-content-Final',
        'index': 1
    })

    return [html.Div([
        files_table,
        show_button,
        file_cotent
    ])]


@callback(
    Output({'type': 'file-content-Final', 'index': ALL}, 'children'),
    Input({'type': 'show-preview-button-Final', 'index': ALL}, 'n_clicks'),
    State({'type': 'Final-files-table', 'index': ALL}, 'selectedRows'),
    State({'type': 'Final-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-Final', 'value')
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
                'type': 'Final-file-table',
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
                'type': 'Final-file-table',
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
    
