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

# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings



FIRST, LAST = 0, -1
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # for the output files





dash.register_page(__name__, name='Combined File Generation', order=5)

pages = {}
try:
    Pconfigs = json.load(open(r"C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\Pconfigs\paths.json", "r"))
except:
    Pconfigs = json.load(open(r"C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\Pconfigs\paths.json", "r"))

for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Generate HR & Steps & Sleep File"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-Combined',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-Combined', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-Combined',
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
                    html.Div(id='raw-data-subjects-container-Combined')
                ])
            ])
        ]),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Generate File',
                        id='generate-file-button-Combined',
                        n_clicks=0,
                        color='success'
                    )
                ]),
                # dbc.Col([
                #     dbc.Accordion([
                #         dbc.AccordionItem([
                #             html.P('Include weekends in the aggregated file?'),
                #             dcc.Checklist(
                #                 id='weekends-radio-Combined',
                #                 options=[
                #                     {'label': 'With Weekends', 'value': 'with'},
                #                     {'label': 'Without Weekends', 'value': 'without'}
                #                 ],
                #                 value=[]
                #             )
                #             ], title='By Activity Aggregation File'),
                #     ]),
                # ]),
            ])
        ]),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-Combined',
                                message=''
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-Combined',
                        message=''
                    ),
                
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button('Load outputs', id='load-outputs-button-Combined', n_clicks=0, color='primary')
                ])
            ]),
            dbc.Row([
                    dbc.Col([
                        html.Div(id='outputs-Combined')
                    ])
                ]),
        

])

@callback(
    Output('raw-data-subjects-container-Combined', 'children'),
    Input('load-fitbit-button-Combined', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Combined', 'value')
)
def load_raw_data_subjects(n_clicks, project):
    
    if n_clicks == 0:
        raise PreventUpdate

    if n_clicks == []:
        raise PreventUpdate
    
    print(f'n_clicks: {n_clicks}')

    path = Path(Pconfigs[project])

    outputs_path = Path(path.joinpath('Outputs'))

    if not outputs_path.exists():
        return html.Div('No outputs found')

    subjects_pattern = r'\d{3}$'

    if os.path.exists(r'C:\Users\PsyLab-6028'):
        updated_basic_files_stats = rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_Combined.parquet'
    else:
        updated_basic_files_stats = rf'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\sub_selection\{project}_sub_selection_Combined.parquet'   

    if not Path(updated_basic_files_stats).exists():
        initiate_basic_table_button = dbc.Button(
            'Initiate Basic Table',
            id={
                'type': 'Initiate-Basic-Table',
                'index': 1
            },
            n_clicks=0,
            color='primary'
        )

        return [
            html.Div('No basic files stats found please initiate the basic table'),
            initiate_basic_table_button
        ]
    

    else:
        updated_basic_files_stats = pl.read_parquet(updated_basic_files_stats)

        if updated_basic_files_stats.shape[0] == 0:
            initiate_basic_table_button = dbc.Button(
                'Initiate Basic Table',
                id={
                    'type': 'Initiate-Basic-Table',
                    'index': 1
                },
                n_clicks=0,
                color='primary'
            )

            return [
                html.Div('No basic files stats found please initiate the basic table'),
                initiate_basic_table_button
            ]

        updated_basic_files_stats = (
            updated_basic_files_stats
            .with_columns(
                refresh = False,
                run = True
            )
            .with_columns(
                pl.col('Mean Missing HR %').round(2),
                pl.col('Mean Missing Sleep %').round(2),
            )
        )

        def_columns_basic = [
            {'field': 'Subject', 'headerName': 'Subject', 'sortable': True, 'filter': True},
            {'field': 'Mean Missing HR %', 'headerName': 'Mean Missing HR %', 'sortable': True, 'filter': True},
            {'field': 'Mean Missing Sleep %', 'headerName': 'Mean Missing Sleep %', 'sortable': True, 'filter': True},
            {'field': 'refresh', 'headerName': 'refresh', 'sortable': True, 'filter': True, 'editable': True},
            {'field': 'run', 'headerName': 'run', 'sortable': True, 'filter': True, 'editable': True},
            {'field': 'Last Updated', 'headerName': 'Last Updated', 'sortable': True, 'filter': True}
        ]

        rows_basic = updated_basic_files_stats.to_pandas().to_dict('records')

        grid_basic = dag.AgGrid(
            id={
                'type': 'basic-table',
                'index': 1
            },
            columnDefs=def_columns_basic,
            rowData=rows_basic,
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
            }
        )

        refresh_button = dbc.Button(
            'Refresh Selected',
            id={
                'type': 'Refresh-Selected',
                'index': 1
            },
            n_clicks=0,
            color='primary'
        )
            

        return [
            html.H4('Basic Files Stats:'),
            grid_basic,
            refresh_button
        ]


@callback(
    Output('confirm-dialog-Combined', 'displayed'),
    Output('confirm-dialog-Combined', 'message'),
    Output('error-gen-dialog-Combined', 'displayed'),
    Output('error-gen-dialog-Combined', 'message'),
    Input({'type': 'Initiate-Basic-Table','index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-Combined', 'value'),
    State('usenname-Combined', 'value'),
    prevent_initial_call=True
)
def initiate_basic_table(n_clicks, project, user_name):
    if not n_clicks:
        raise PreventUpdate
    if n_clicks[0] == 0:
        raise PreventUpdate
    
    if n_clicks == []:
        raise PreventUpdate
    
    print(f'n_clicks initiate: {n_clicks}')

    path = Path(Pconfigs[project])

    if user_name == '':
        return False, '', True, 'Please enter your name before generating the file'

    include_weekends = True
    exclude_weekends = False

    try:
        param = project
        param2 = now
        param3 = user_name
        param4 = include_weekends
        param5 = exclude_weekends

        if os.path.exists(r'C:\Users\PsyLab-6028'):
            script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\combinedFileScript.py'
        else:
            script_path = r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\scripts\combinedFileScript.py'

        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3} {param4} {param5}'
            print(command)
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3} {param4} {param5}'
            print(command)

        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return True, 'Initiating Basic Table...', False, ''
    except Exception as e:
        return False, '', True, str(e)
    

@callback(
    Output('confirm-dialog-Combined', 'displayed', allow_duplicate=True),
    Output('confirm-dialog-Combined', 'message', allow_duplicate=True),
    Output('error-gen-dialog-Combined', 'displayed', allow_duplicate=True),
    Output('error-gen-dialog-Combined', 'message', allow_duplicate=True),
    Input({'type': 'Refresh-Selected', 'index': ALL}, 'n_clicks'),
    State({'type': 'basic-table', 'index': ALL}, 'rowData'),
    State('project-selection-dropdown-FitBit-Combined', 'value'),
    State('usenname-Combined', 'value'),
    prevent_initial_call=True
)
def refresh_selected(n_clicks, rows, project, user_name):
    if n_clicks == 0:
        raise PreventUpdate
    
    if n_clicks == []:
        raise PreventUpdate

    if n_clicks[0] == 0:
        raise PreventUpdate

    path = Path(Pconfigs[project])

    selected_rows = [row for row in rows[0]]

    selected_rows = (
        pl.DataFrame(selected_rows)
        .filter(
            pl.col('refresh') == True
        )
        .drop('refresh')
        .drop('Last Updated')
        .drop('run')
    )

    if os.path.exists(r'C:\Users\PsyLab-6028'):
        selected_rows.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_refresh_Combined.parquet')
    else:
        selected_rows.write_parquet(rf'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\sub_selection\{project}_sub_selection_refresh_Combined.parquet')


    if user_name == '':
        return False, '', True, 'Please enter your name before generating the file'

    include_weekends = True
    exclude_weekends = False


    try:
        param = project
        param2 = now
        param3 = user_name
        param4 = include_weekends
        param5 = exclude_weekends
        if os.path.exists(r'C:\Users\PsyLab-6028'):
            script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\combinedFileScript.py'
        else:
            script_path = r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\scripts\combinedFileScript.py'
        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3} {param4} {param5}'
            print(command)
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3} {param4} {param5}'
            print(command)

        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return True, 'Refreshing Selected...', False, ''
    except Exception as e:
        return False, '', True, str(e)
    

@callback(
    Output('confirm-dialog-Combined', 'displayed', allow_duplicate=True),
    Output('confirm-dialog-Combined', 'message', allow_duplicate=True),
    Output('error-gen-dialog-Combined', 'displayed', allow_duplicate=True),
    Output('error-gen-dialog-Combined', 'message', allow_duplicate=True),
    Input('generate-file-button-Combined', 'n_clicks'),
    State({'type': 'basic-table', 'index': ALL}, 'rowData'),
    State('project-selection-dropdown-FitBit-Combined', 'value'),
    State('usenname-Combined', 'value'),
    prevent_initial_call=True
)
def generate_file(n_clicks, rows, project, user_name):
    if n_clicks == 0:
        raise PreventUpdate
    print(f'n_clicks generate: {n_clicks}')
    if n_clicks == []:
        raise PreventUpdate
    
    # if n_clicks[0] == 0:
    #     raise PreventUpdate

    print(f'n_clicks generate: {n_clicks}')

    path = Pconfigs[project]

    updated_basic_df = (
        pl.DataFrame(rows[0])
        .filter(
            pl.col('run') == True
        )
        .drop('run')
        .drop('refresh')
        .drop('Last Updated')
    )

    include_weekends = True
    exclude_weekends = False


    if os.path.exists(r'C:\Users\PsyLab-6028'):
        updated_basic_df.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_gen_Combined.parquet')
    else:
        updated_basic_df.write_parquet(rf'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\sub_selection\{project}_sub_selection_gen_Combined.parquet')

    if user_name == '':
        return False, '', True, 'Please enter your name before generating the file'
    


    try:
        print('Generating File')
        param = project
        param2 = now
        param3 = user_name
        param4 = include_weekends
        param5 = exclude_weekends

        if os.path.exists(r'C:\Users\PsyLab-6028'):
            script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\getCombinedFileScript.py'
        else:
            script_path = r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\scripts\getCombinedFileScript.py'
            

        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {param} {param2} {param3} {param4} {param5}'
            print(command)
        else:
            command = f'python3 "{script_path}" {param} {param2} {param3} {param4} {param5}'
            print(command)

        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        
        return True, 'Generating File...', False, ''
    except Exception as e:
        return False, '', True, str(e)
    

    

                            

                

@callback(
    Output('outputs-Combined', 'children'),
    Input('load-outputs-button-Combined', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Combined', 'value')
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
                'type': 'Combined-outputs-dropdown',
                'index': 1
            },
            options=dropdown_options,
            value=outputs_folders[0]
        ),
        dbc.Button('Load', id={
            'type': 'load-Combined-outputs-button',
            'index': 1
        }, n_clicks=0, color='primary'),

        html.Div(id={
            'type': 'Combined-outputs-container',
            'index': 1
        })
    ])

@callback(
    Output({'type': 'Combined-outputs-container', 'index': ALL}, 'children'),
    Input({'type': 'load-Combined-outputs-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'Combined-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-Combined', 'value')
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
            'type': 'Combined-files-table',
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
        'type': 'show-preview-button-Combined',
        'index': 1
    }, n_clicks=0, color='primary')

    file_cotent = html.Div(id={
        'type': 'file-content-Combined',
        'index': 1
    })

    return [html.Div([
        files_table,
        show_button,
        file_cotent
    ])]


@callback(
    Output({'type': 'file-content-Combined', 'index': ALL}, 'children'),
    Input({'type': 'show-preview-button-Combined', 'index': ALL}, 'n_clicks'),
    State({'type': 'Combined-files-table', 'index': ALL}, 'selectedRows'),
    State({'type': 'Combined-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-Combined', 'value')
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
                'type': 'Combined-file-table',
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
                'type': 'Combined-file-table',
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
    
