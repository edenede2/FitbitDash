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





dash.register_page(__name__, name='Combined File Generation', order=5)

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
                dbc.Col([
                    dbc.Accordion([
                        dbc.AccordionItem([
                            html.P('Include weekends in the aggregated file?'),
                            dcc.Checklist(
                                id='weekends-radio-Combined',
                                options=[
                                    {'label': 'With Weekends', 'value': 'with'},
                                    {'label': 'Without Weekends', 'value': 'without'}
                                ],
                                value=[]
                            )
                            ], title='By Activity Aggregation File'),
                    ]),
                ]),
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
        return html.Div('No outputs found found')

    subjects_pattern = r'\d{3}$'

    updated_basic_files_stats = rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_Combined.parquet'

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
    State('weekends-radio-Combined', 'value'),
    prevent_initial_call=True
)
def initiate_basic_table(n_clicks, project, weekends, user_name):
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

    include_weekends = True if 'with' in weekends else False
    exclude_weekends = True if 'without' in weekends else False

    print(f'include_weekends: {include_weekends}')

    try:
        param = project
        param2 = now
        param3 = user_name
        param4 = include_weekends
        param5 = exclude_weekends


        script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\combinedFileScript.py'

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
    State('weekends-radio-Combined', 'value'),
    prevent_initial_call=True
)
def refresh_selected(n_clicks, rows, project, user_name, weekends):
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

    selected_rows.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_refresh_Combined.parquet')

    include_weekends = True if 'with' in weekends else False
    exclude_weekends = True if 'without' in weekends else False

    if user_name == '':
        return False, '', True, 'Please enter your name before generating the file'

    try:
        param = project
        param2 = now
        param3 = user_name
        param4 = include_weekends
        param5 = exclude_weekends

        script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\combinedFileScript.py'

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
    State('weekends-radio-Combined', 'value'),
    prevent_initial_call=True
)
def generate_file(n_clicks, rows, project, user_name, weekends):
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

    include_weekends = True if 'with' in weekends else False
    exclude_weekends = True if 'without' in weekends else False

    updated_basic_df.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_gen_Combined.parquet')

    if user_name == '':
        return False, '', True, 'Please enter your name before generating the file'
    


    try:
        print('Generating File')
        param = project
        param2 = now
        param3 = user_name
        param4 = include_weekends
        param5 = exclude_weekends

        script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\getCombinedFileScript.py'

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
    

    

                            

