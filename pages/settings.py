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





dash.register_page(__name__, name='Settings', order=0)

pages = {}

Pconfigs = json.load(open(r".\pages\Pconfigs\paths.json", "r"))


for key in Pconfigs.keys():
    page_name = key
    page_value = key

    pages[page_name] = page_value

pages['NEW'] = 'NEW'


layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Project Settings page"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project or open a new one:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-settings',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='NEW'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load', id='load-fitbit-button-settings', n_clicks=0, color='primary')
                ]),
            ]),
            html.Hr(),
        ]),
        dbc.Container(
            children=[],
            id='settings-container'),
        dcc.ConfirmDialog(
            id='confirm-dialog-settings',
            message='Settings saved successfully',
        ),
        dcc.ConfirmDialog(
            id='error-dialog-settings',
            message='Error saving settings',
        )
    ])

@callback(
    Output('settings-container', 'children'),
    Input('load-fitbit-button-settings', 'n_clicks'),
    State('project-selection-dropdown-FitBit-settings', 'value'),
    State('settings-container', 'children')
)
def update_paths(n_clicks, value, children):

    if not n_clicks:
        raise PreventUpdate

    if n_clicks == 0:
        raise PreventUpdate

    if value == 'NEW':
        new_children = [
            dbc.Row([
                dbc.Col([
                    html.Div('Please enter the name of the new project:'),
                    dbc.Input(
                        id={ 'type': 'new-project-name-settings', 'index': 1},
                        placeholder='project_name',
                        value=''
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please enter the path to the raw data folder:'),
                    dbc.Input(
                        id={
                            'type': 'raw-data-path-settings',
                            'index': 1
                        },
                        placeholder='raw_data_path',
                        value=''
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please enter the path to the processed data folder:'),
                    dbc.Input(
                        id={
                            'type': 'processed-data-path-settings',
                            'index': 1
                        },
                        placeholder='processed_data_path',
                        value=''
                    )
                ]),

                html.Hr()
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Save',
                        id={
                            'type': 'save-settings-button',
                            'index': 1
                        }, 
                        n_clicks=0, 
                        color='primary')
                ]),
            ]),
        ]
        children = new_children
        return children
    
    else:
        raw_data_json_path = r'.\pages\Pconfigs\paths data.json' 

        raw_data_path = json.load(open(raw_data_json_path, "r"))[value]
        new_children = [
            dbc.Row([
                dbc.Col([
                    html.Div('Please enter the name of the new project:'),
                    dbc.Input(
                        id={ 'type': 'new-project-name-settings', 'index': 1},
                        placeholder='project_name',
                        value=value
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please enter the path to the raw data folder:'),
                    dbc.Input(
                        id={
                            'type': 'raw-data-path-settings',
                            'index': 1
                        },
                        placeholder='raw_data_path',
                        value=raw_data_path
                    )
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please enter the path to the processed data folder:'),
                    dbc.Input(
                        id={
                            'type': 'processed-data-path-settings',
                            'index': 1
                        },
                        placeholder='processed_data_path',
                        value=Pconfigs[value]
                    )
                ]),

                html.Hr()
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Save', 
                        id={
                            'type': 'save-settings-button',
                            'index': 1
                        },
                        n_clicks=0, 
                        color='primary')
                ]),
                dbc.Col([
                    dbc.Button(
                        'Delete', 
                        id={
                            'type': 'delete-settings-button',
                            'index': 1
                        },
                        n_clicks=0, 
                        color='danger')
                ]),

            ]),
        ]
        children = new_children 
        return  children


@callback(
    Output('confirm-dialog-settings', 'displayed'),
    Input({'type': 'save-settings-button', 'index': ALL}, 'n_clicks'),
    State({'type': 'raw-data-path-settings', 'index': ALL}, 'value'),
    State({'type': 'processed-data-path-settings', 'index': ALL}, 'value'),
    State({'type': 'new-project-name-settings', 'index': ALL}, 'value'),
    State('project-selection-dropdown-FitBit-settings', 'value')
)
def save_settings(n_clicks, raw_data_path, processed_data_path,new_project_name, project_name):
    if not n_clicks:
        raise PreventUpdate

    if n_clicks == 0:
        raise PreventUpdate
    
    if n_clicks[0] == 0:
        raise PreventUpdate

    if project_name == 'NEW':
        raw_data_json_path = r'.\pages\Pconfigs\paths data.json'


        raw_data = json.load(open(raw_data_json_path, "r"))
        raw_data[new_project_name[0]] = raw_data_path[0]
        json.dump(raw_data, open(raw_data_json_path, "w"), indent=4)
        
        Pconfigs[new_project_name[0]] = processed_data_path[0]
        json.dump(Pconfigs, open(r".\pages\Pconfigs\paths.json", "w"), indent=4) 

        return True

    else:
        
        raw_data_json_path = r'.\pages\Pconfigs\paths data.json'
        
        
        raw_data = json.load(open(raw_data_json_path, "r"))
        raw_data[new_project_name[0]] = raw_data_path[0]
        json.dump(raw_data, open(raw_data_json_path, "w"), indent=4)

        Pconfigs[new_project_name[0]] = processed_data_path[0]
        json.dump(Pconfigs, open(r".\pages\Pconfigs\paths.json", "w"), indent=4)

        return True


@callback(
    Output('confirm-dialog-settings', 'displayed', allow_duplicate=True),
    Output('confirm-dialog-settings', 'message'),
    Input({'type': 'delete-settings-button', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-settings', 'value'),
    prevent_initial_call=True
)
def delete_settings(n_clicks, project_name):
    if not n_clicks:
        raise PreventUpdate

    if n_clicks == 0:
        raise PreventUpdate

    if n_clicks[0] == 0:
        raise PreventUpdate

    raw_data_json_path = r'.\pages\Pconfigs\paths data.json'

    raw_data = json.load(open(raw_data_json_path, "r"))
    raw_data.pop(project_name)
    json.dump(raw_data, open(raw_data_json_path, "w"), indent=4)

    Pconfigs.pop(project_name)
    json.dump(Pconfigs, open(r".\pages\Pconfigs\paths.json", "w"), indent=4)

    message = f'{project_name} deleted successfully! Please close the app and open it again to see the changes.'

    return True, message