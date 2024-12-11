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





dash.register_page(__name__, name='extract sleep data', order=3)

pages = {}

Pconfigs = json.load(open(ut.convert_path_to_current_os(r".\pages\Pconfigs\paths.json"), "r"))


for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dcc.Store(id='file-data-store-Sleep-All-Subjects', storage_type='memory'),
        dcc.Store(id='output-file-path-Sleep-All-Subjects', storage_type='memory'),

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
            ]),
           dbc.Row([
                    dbc.Col([
                        dbc.Button('Load outputs', id='load-outputs-button-Sleep-All-Subjects', n_clicks=0, color='primary')
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id='outputs-Sleep-All-Subjects'),
                    ]),
                    dbc.Col([
                        html.Div(id='outputs-dropdowns-Sleep-All-Subjects'),
                    ])
                ])
            ]),
            html.Div([
                html.Hr(),
                html.Div(id='outputs-plots-Sleep-All-Subjects'),
            ], style={'margin-top': '50px'})
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
    

    
    paths_json = json.load(open(ut.convert_path_to_current_os(r".\pages\Pconfigs\paths.json"), "r"))
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
        'Subject': pl.Series([], dtype=pl.Utf8),
        'Sleep jsons': pl.Series([], dtype=pl.Int64),
        'Subject dates': pl.Series([], dtype=pl.Boolean)
    })


    folders_tqdm = tqdm(os.listdir(DATA_PATH), desc='Checking folders')
    for folder in folders_tqdm:
        folders_tqdm.set_description(f'Checking folder {folder}')
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

    paths_json = json.load(open(r".\pages\Pconfigs\paths.json", "r"))
    project_path = Path(paths_json[project])

    DATA_PATH, OUTPUT_PATH, ARCHIVE_PATH, AGGREGATED_OUTPUT_PATH, METADATA_PATH, SUBJECT_FOLDER_FORMAT = ut.declare_project_global_variables(project_path)

    PROJECT_CONFIG = json.load(open(r'.\pages\Pconfigs\paths data.json', 'r'))

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
    
    selected_rows_df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_sleep_all_subjects.parquet') 

    
    if username == '':
        return False, True, 'Please enter your name'


    try:

        param = project
        param2 = now
        param3 = username
        # Define the command to run the script
        script_path = r'.\pages\scripts\sleepAllSubjectsScript.py'   
            
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
    



     
@callback(
    Output('outputs-Sleep-All-Subjects', 'children'),
    Input('load-outputs-button-Sleep-All-Subjects', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Sleep-All-Subjects', 'value')
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
                'type': 'Sleep-All-Subjects-outputs-dropdown',
                'index': 1
            },
            options=dropdown_options,
            value=outputs_folders[0]
        ),
        dbc.Button('Load', id={
            'type': 'load-Sleep-All-Subjects-outputs-button',
            'index': 1
        }, n_clicks=0, color='primary'),

        dbc.Container(id={
            'type': 'Sleep-All-Subjects-outputs-container',
            'index': 1
        })
    ])

@callback(
    Output({'type': 'Sleep-All-Subjects-outputs-container', 'index': ALL}, 'children'),
    Input({'type': 'load-Sleep-All-Subjects-outputs-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'Sleep-All-Subjects-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-Sleep-All-Subjects', 'value')
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
            'type': 'Sleep-All-Subjects-files-table',
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
        'type': 'show-preview-button-Sleep-All-Subjects',
        'index': 1
    }, n_clicks=0, color='primary')

    

    file_cotent = html.Div(id={
        'type': 'file-content-Sleep-All-Subjects',
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
    Output('output-file-path-Sleep-All-Subjects', 'data'),
    Output('outputs-dropdowns-Sleep-All-Subjects', 'children'),
    Input({'type': 'show-preview-button-Sleep-All-Subjects', 'index': ALL}, 'n_clicks'),
    State({'type': 'Sleep-All-Subjects-files-table', 'index': ALL}, 'selectedRows'),
    State({'type': 'Sleep-All-Subjects-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-Sleep-All-Subjects', 'value')
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
            'type': 'select-id-Dropdown-Sleep-All-Subjects',
            'index': 1
        },
        options=id_options,
        placeholder='Select an ID',
        disabled=id_dropdown_disabled
    )

    column_selection_dropdown = dcc.Dropdown(
        id={
            'type': 'column-selection-dropdown-Sleep-All-Subjects',
            'index': 1
        },
        options=column_options,
        placeholder='Select a column'
    )

    load_content_button = dbc.Button('Load content', id={
        'type': 'load-content-button-Sleep-All-Subjects',
        'index': 1
    }, n_clicks=0, color='primary')

    return str(selected_file_path), [column_selection_dropdown, select_id_dropdown, load_content_button]

@callback(
    Output('outputs-plots-Sleep-All-Subjects', 'children'),
    Output({'type': 'file-content-Sleep-All-Subjects', 'index': ALL}, 'children'),
    Input({'type': 'load-content-button-Sleep-All-Subjects', 'index': ALL}, 'n_clicks'),
    State({'type': 'select-id-Dropdown-Sleep-All-Subjects', 'index': ALL}, 'value'),
    State({'type': 'column-selection-dropdown-Sleep-All-Subjects', 'index': ALL}, 'value'),
    State('output-file-path-Sleep-All-Subjects', 'data'),
    State({'type': 'select-id-Dropdown-Sleep-All-Subjects', 'index': ALL}, 'disabled')
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
            'type': 'Sleep-All-Subjects-file-table',
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