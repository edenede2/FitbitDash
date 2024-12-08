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



dash.register_page(__name__, name='download api', order=1)

pages = {}

try:
    Pconfigs = json.load(open(r"C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\Pconfigs\paths.json", "r"))
except:
    Pconfigs = json.load(open(r"C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\Pconfigs\paths.json", "r"))
print(Pconfigs)


for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Download fitbit data from api"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-Download',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load', id='load-fitbit-button-Download', n_clicks=0, color='primary')
                ]),
                dbc.Col([
                    html.Div('Please enter your name:'),
                    dbc.Textarea(
                        id='usenname-Download',
                        placeholder='user_name',
                        value=''
                    )
                ]),

                html.Hr()
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='raw-Download-subjects-container')
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Generate .zip file for Selected Subjects',
                        id='Generate-file-button-Download',
                        n_clicks=0,
                        color='primary'
                    ),
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-Download',
                                message='File Generation Started'
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-Download',
                        message='Error generating file'
                    )
                ]),
            ])
        ])
])


@callback(
    Output('raw-Download-subjects-container', 'children'),
    Input('load-fitbit-button-Download', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Download', 'value')
)
def load_fitbit_subjects(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate

    print(f'Project{project}')
    print(Pconfigs)
    path = Pconfigs[project]
    print(path)
    subjects_dates = pl.read_csv(os.path.join(path,'Metadata', 'Subjects Dates.csv'))
    print(subjects_dates)
    subjects_dates = subjects_dates.with_row_index().to_pandas()
    print(subjects_dates)
    subjects = subjects_dates['Id'].unique()
    print(subjects)

    if 'token' not in subjects_dates.columns:
        subjects_dates['token'] = None


    rows = subjects_dates.to_dict('records')

    columns_def = [
        {'headerName': 'Index', 'field': 'index', 'sortable': True, 'filter': True, 'checkboxSelection': True},
        {'headerName': 'Id', 'field': 'Id', 'sortable': True, 'filter': True},
        {'headerName': 'token', 'field': 'token', 'sortable': True, 'filter': True},
        {'headerName': 'ExperimentStartDate', 'field': 'ExperimentStartDate', 'sortable': True, 'filter': True},
        {'headerName': 'ExperimentStartTime', 'field': 'ExperimentStartTime', 'sortable': True, 'filter': True},
        {'headerName': 'ExperimentEndDate', 'field': 'ExperimentEndDate', 'sortable': True, 'filter': True},
        {'headerName': 'ExperimentEndTime', 'field': 'ExperimentEndTime', 'sortable': True, 'filter': True},
    ]

    subsTable = dag.AgGrid(
        id={
            'type': 'subjects-table-Download',
            'index': n_clicks
        },
        columnDefs=columns_def,
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

    return [
        html.H4('Subjects:'),
        subsTable
    ]


@callback(
    Output('confirm-dialog-Download', 'displayed'),
    Output('error-gen-dialog-Download', 'displayed'),
    Output('error-gen-dialog-Download', 'message'),
    Input('Generate-file-button-Download', 'n_clicks'),
    State({'type': 'subjects-table-Download', 'index': ALL}, 'selectedRows'),
    State('usenname-Download', 'value'),
    State('project-selection-dropdown-FitBit-Download', 'value')
)
def generate_file(n_clicks, selected_rows, username, project):
    if n_clicks == 0:
        raise PreventUpdate

    if not selected_rows:
        return True, False, 'No subjects selected'

    path = Pconfigs[project]
    print(path)
    subjects_dates = pl.read_csv(os.path.join(path, 'Metadata', 'Subjects Dates.csv'))
    subjects_dates = subjects_dates.with_row_index().to_pandas()
    subjects = subjects_dates['Id'].unique()

    selected_subjects = pl.DataFrame({
        'Id': [row['Id'] for row in selected_rows[0]],
        'ExperimentStartDate': [row['ExperimentStartDate'] for row in selected_rows[0]],
        'ExperimentStartTime': [row['ExperimentStartTime'] for row in selected_rows[0]],
        'ExperimentEndDate': [row['ExperimentEndDate'] for row in selected_rows[0]],
        'ExperimentEndTime': [row['ExperimentEndTime'] for row in selected_rows[0]],
        'token': [row['token'] for row in selected_rows[0]]
    })
    
    if os.path.exists(rf'C:\Users\PsyLab-6028'):
        selected_subjects.write_parquet(rf'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\sub_selection\{project}_sub_selection_download_api.parquet')
    else:
        selected_subjects.write_parquet(rf'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\sub_selection\{project}_sub_selection_download_api.parquet')

    print(selected_subjects)

    if username == '':
        return False, True, 'Please enter your name'

    try:

        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        print(now)
        if os.path.exists(rf'C:\Users\PsyLab-6028'):
            script_path = r'C:\Users\PsyLab-6028\Desktop\FitbitDash\pages\scripts\download_fitbit_data.py'
        else:
            script_path = r'C:\Users\PsyLab-7084\Documents\GitHub\FitbitDash\pages\scripts\download_fitbit_data.py'
            
        print(script_path)
        if platform.system() == 'Windows':
            command = f'start cmd /c python "{script_path}" {project} {username} {now}'
        else:
            command = f'python3 "{script_path}" {project} {username} {now}'
        print(command)
        process = subprocess.Popen(command, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        print(process)
        return False, False, ''
    
    except Exception as e:
        return False, True, str(e)
