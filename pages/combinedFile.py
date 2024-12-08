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





dash.register_page(__name__, name='Combined File Generation', order=5)

pages = {}
Pconfigs = json.load(open(r".\pages\Pconfigs\paths.json", "r"))


for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value


layout = html.Div([
        dcc.Store(id='file-data-store-Combined', storage_type='memory'),
        dcc.Store(id='output-file-path-Combined', storage_type='memory'),
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
                        html.Div(id='outputs-Combined'),
                    ]),
                    dbc.Col([
                        html.Div(id='outputs-dropdowns-Combined'),
                    ])
                ]),
            html.Div([
                html.Hr(),
                html.Div(id='outputs-plots-Combined'),
            ], style={'margin-top': '50px'})
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

    updated_basic_files_stats = rf'.\pages\sub_selection\{project}_sub_selection_Combined.parquet'   


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

        script_path = r'.\pages\scripts\combinedFileScript.py'

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

    selected_rows.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_refresh_Combined.parquet')


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
        script_path = r'.\pages\scripts\combinedFileScript.py'
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


    updated_basic_df.write_parquet(rf'.\pages\sub_selection\{project}_sub_selection_gen_Combined.parquet')

    if user_name == '':
        return False, '', True, 'Please enter your name before generating the file'
    


    try:
        print('Generating File')
        param = project
        param2 = now
        param3 = user_name
        param4 = include_weekends
        param5 = exclude_weekends

        script_path = r'.\pages\scripts\getCombinedFileScript.py'

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

        dbc.Container(id={
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

    return [dbc.Row(
        dbc.Col([
        html.Hr(),
        files_table,
        show_button,
        file_cotent
    ]))]



@callback(
    Output('output-file-path-Combined', 'data'),
    Output('outputs-dropdowns-Combined', 'children'),
    Input({'type': 'show-preview-button-Combined', 'index': ALL}, 'n_clicks'),
    State({'type': 'Combined-files-table', 'index': ALL}, 'selectedRows'),
    State({'type': 'Combined-outputs-dropdown', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-Combined', 'value')
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
            'type': 'select-id-Dropdown-Combined',
            'index': 1
        },
        options=id_options,
        placeholder='Select an ID',
        disabled=id_dropdown_disabled
    )

    column_selection_dropdown = dcc.Dropdown(
        id={
            'type': 'column-selection-dropdown-Combined',
            'index': 1
        },
        options=column_options,
        placeholder='Select a column'
    )

    load_content_button = dbc.Button('Load content', id={
        'type': 'load-content-button-Combined',
        'index': 1
    }, n_clicks=0, color='primary')

    return str(selected_file_path), [column_selection_dropdown, select_id_dropdown, load_content_button]

@callback(
    Output('outputs-plots-Combined', 'children'),
    Output({'type': 'file-content-Combined', 'index': ALL}, 'children'),
    Input({'type': 'load-content-button-Combined', 'index': ALL}, 'n_clicks'),
    State({'type': 'select-id-Dropdown-Combined', 'index': ALL}, 'value'),
    State({'type': 'column-selection-dropdown-Combined', 'index': ALL}, 'value'),
    State('output-file-path-Combined', 'data'),
    State({'type': 'select-id-Dropdown-Combined', 'index': ALL}, 'disabled')
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