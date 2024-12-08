from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import bioread as br
import h5py
import plotly.express as px
import webview
import plotly.figure_factory as ff
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





dash.register_page(__name__, name='Time Series Visualization', order=7)

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
                    html.H1("TS Visualization Page"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-TS',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-TS', n_clicks=0, color='primary')
                ]),

                html.Hr()
            ]),
        ]),
        dbc.Container(
            id='select-subjects-container-TS',
            className='ny-5',
            fluid=True
        ),
        dbc.Container(
            id='raw-data-subjects-container-TS',
            className='ny-5',
            fluid=True
            
        ),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-TS',
                                message=''
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-TS',
                        message=''
                    ),
                
                ]),
            ]),
            dcc.Store(id='subjects-TS', data=[]),
        

])



@callback(
    Output('select-subjects-container-TS', 'children'),
    Input('load-fitbit-button-TS', 'n_clicks'),
    State('project-selection-dropdown-FitBit-TS', 'value')
)
def select_subjects(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    # Get the subjects
    minute_reso_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', 'All_Subjects_1_Minute_Resolution.parquet')

    if not minute_reso_path.exists():
        return dbc.Alert('The file does not exist. Please generate it first in the combining files page.', color='danger')
    
    df = pl.read_parquet(minute_reso_path)

    if df.is_empty():
        return dbc.Alert('The file is empty.', color='danger')
    
    subjects = df['Id'].unique().to_list()

    select_subjects_df = (
        pl.DataFrame(
            {'Id': subjects}
        )
        .sort('Id')
        .with_columns(
            show=pl.lit(False),
            done=pl.lit(False)
        )
        .with_row_index()
    )

    select_subjects_df = select_subjects_df.to_pandas()

    columns = [
        {'headerName': 'index', 'field': 'index', 'sortable': True, 'filter': True},
        {'headerName': 'Id', 'field': 'Id', 'sortable': True, 'filter': True},
        {'headerName': 'Show', 'field': 'show', 'sortable': True, 'filter': True, 'editable': True},
        {'headerName': 'Done', 'field': 'done', 'sortable': True, 'filter': True}
    ]

    rows = select_subjects_df.to_dict('records')

    grid_options = {
        'pagination': True,
        'paginationPageSize': 10,
        'rowSelection': 'multiple',
    }

    grid = dag.AgGrid(
        id={
            'type': 'select-subjects-grid-TS',
            'index': 1
        },
        columnDefs=columns,
        rowData=rows,
        defaultColDef={
            'resizable': True,
            'editable': False,
            'sortable': True,
            'filter': True
        },
        columnSize= 'autoSize',
        dashGridOptions=grid_options,
    )

    show_selected_button = dbc.Button(
        'Show Selected',
        id={
            'type': 'show-selected-button-TS',
            'index': 1
        },
        color='primary',
        n_clicks=0
    )

    hide_selected_button = dbc.Button(
        'Hide Plots',
        id={
            'type': 'hide-selected-button-TS',
            'index': 1
        },
        color='primary',
        n_clicks=0
    )

    show_all_button = dbc.Button(
        'Show All',
        id={
            'type': 'show-all-button-TS',
            'index': 1
        },
        color='success',
        n_clicks=0
    )

    return [
        dbc.Row([
            dbc.Col([
                grid
            ]),
            dbc.Col([
                show_selected_button,
                hide_selected_button,
                show_all_button
            ])
        ])
    ]




def make_cards(df, subjects=None):
    cards = []

    # Convert DateAndMinute to datetime format and filter by Mode
    df = (
        df.sort('Id')
    )

    for i, subject in zip(range(0,len(subjects)) ,subjects):


        if subjects and subject not in subjects:
            continue

        subject_data = (
            df
            .filter(
                pl.col('Id') == subject
                )
        )

        # Create line plot for BpmMean
        fig_bpm = go.Figure()
        

        custom_data = subject_data.select(['Mode', 'Mode2', 'Feature', 'ValidSleep', 'Weekend', 'outliers', 'not_in_israel'])


        # Create line plot for StepsInMinute
        fig_steps = go.Figure()
        fig_bpm.add_trace(go.Scatter(
            x=subject_data['DateAndMinute'], 
            y=subject_data['StepsInMinute'], 
            mode='lines', 
            name='StepsInMinute'
        ))

        max_bpm = subject_data.select(pl.col('BpmMean').max()).item()
        min_bpm = subject_data.select(pl.col('BpmMean').min()).item()
        max_steps = subject_data.select(pl.col('StepsInMinute').max()).item()
        min_steps = subject_data.select(pl.col('StepsInMinute').min()).item()

        subject_data = (
            subject_data
            .with_columns(
                sleep_intervals_bpm=pl.when(pl.col('Mode') == 'sleeping').then(pl.lit(min_bpm)).otherwise(pl.lit(max_bpm)),
                sleep_intervals_steps=pl.when(pl.col('Mode') == 'sleeping').then(pl.lit((min_steps))).otherwise(pl.lit(max_steps))
                )
        )
  

        fig_bpm.add_trace(go.Scatter(
            x=subject_data['DateAndMinute'],
            y=subject_data['sleep_intervals_bpm'],
            mode='lines',
            name='Sleeping',
            line=dict(color='red'),
            customdata=subject_data.select('Mode').to_pandas(),
            hovertemplate='Time: %{x} <br> Mode: %{customdata[0]}'
        ))
        
   
        q_25_bpm = (
            subject_data
            .select(pl.col('BpmMean'))
            .quantile(0.25)
            .item()
        )

        q_50_bpm = (
            subject_data
            .select(pl.col('BpmMean'))
            .quantile(0.50)
            .item()
        )

        q_75_bpm = (
            subject_data
            .select(pl.col('BpmMean'))
            .quantile(0.75)
            .item()
        )

        # Dendrogram for sleep stages from Mode2
        sleep_data = (
            subject_data
            .select('DateAndMinute', 'Mode2')
            .with_columns(
                Mode2_num=pl.when(pl.col('Mode2') == 'awake')
                .then(pl.lit(max_bpm))
                .when(pl.col('Mode2') == 'sleep_awake')
                .then(pl.lit(max_bpm))
                .when(pl.col('Mode2') == 'light')
                .then(pl.lit(q_75_bpm))
                .when(pl.col('Mode2') == 'rem')
                .then(pl.lit(q_50_bpm))
                .when(pl.col('Mode2') == 'deep')
                .then(pl.lit(q_25_bpm))
                .when(pl.col('Mode2') == 'restless')
                .then(pl.lit(q_75_bpm))
                .when(pl.col('Mode2') == 'asleep')
                .then(pl.lit(q_25_bpm))
                .otherwise(pl.lit(np.nan))
            )
        )

        fig_bpm.add_trace(go.Scatter(
            x=sleep_data['DateAndMinute'],
            y=sleep_data['Mode2_num'],
            mode='lines',
            name='Sleep Stages',
            visible='legendonly',
            customdata=sleep_data.select('Mode2').to_pandas(),
            hovertemplate='Time: %{x} <br> Sleep Stage: %{customdata}'
        ))

        feature_data = (
            subject_data
            .select('DateAndMinute', 'Feature')
            .with_columns(
                features_num=pl.when(pl.col('Feature') == 'high_activity')
                .then(pl.lit(max_bpm))
                .when(pl.col('Feature') == 'med_activity')
                .then(pl.lit(q_75_bpm))
                .when(pl.col('Feature') == 'low_activity')
                .then(pl.lit(q_50_bpm))
                .when(pl.col('Feature') == 'rest')
                .then(pl.lit(q_25_bpm))
                .when(pl.col('Feature') == 'sleep')
                .then(pl.lit(min_bpm))
                .otherwise(pl.lit(np.nan))
            )
        )

        fig_bpm.add_trace(go.Scatter(
            x=feature_data['DateAndMinute'],
            y=feature_data['features_num'],
            mode='lines',
            name='Feature',
            visible='legendonly',
            customdata=feature_data.select('Feature').to_pandas(),
            hovertemplate='Time: %{x} <br> BpmMean: %{y} <br> Feature: %{customdata}'
        ))

        valid_sleep = (
            subject_data
            .select('DateAndMinute', 'ValidSleep')
            .with_columns(
                valid_sleep_num=pl.when(pl.col('ValidSleep') == True)
                .then(pl.lit(min_bpm))
                .when(pl.col('ValidSleep') == False)
                .then(pl.lit(max_bpm))
                .otherwise(pl.lit(np.nan))
            ) 
        )

        fig_bpm.add_trace(go.Scatter(
            x=valid_sleep['DateAndMinute'],
            y=valid_sleep['valid_sleep_num'],
            mode='lines',
            name='ValidSleep',
            visible='legendonly',
            customdata=valid_sleep.select('ValidSleep').to_pandas(),
            hovertemplate='Time: %{x} <br> BpmMean: %{y} <br> ValidSleep: %{customdata}'
        )
        )

        weekend = (
            subject_data
            .select('DateAndMinute', 'Weekend')
            .with_columns(
                weekend_num=pl.when(pl.col('Weekend') == True)
                .then(pl.lit(max_bpm))
                .when(pl.col('Weekend') == False)
                .then(pl.lit(min_bpm))
                .otherwise(pl.lit(np.nan))
            )
        )

        fig_bpm.add_trace(go.Scatter(
            x=weekend['DateAndMinute'],
            y=weekend['weekend_num'],
            mode='lines',
            name='Weekend',
            visible='legendonly',
            customdata=weekend.select('Weekend').to_pandas(),
            hovertemplate='Time: %{x} <br> BpmMean: %{y} <br> Weekend: %{customdata}'
        )
        )


        outliers = (
            subject_data
            .select('DateAndMinute', 'outliers')
            .with_columns(
                outliers_num=pl.when(pl.col('outliers') == True)
                .then(pl.lit(max_bpm))
                .when(pl.col('outliers') == False)
                .then(pl.lit(min_bpm))
                .otherwise(pl.lit(np.nan))
            )
        )

        fig_bpm.add_trace(go.Scatter(
            x=outliers['DateAndMinute'],
            y=outliers['outliers_num'],
            mode='lines',
            name='Outliers',
            visible='legendonly',
            customdata=outliers.select('outliers').to_pandas(),
            hovertemplate='Time: %{x} <br> BpmMean: %{y} <br> Outliers: %{customdata}'
        )
        )

        not_in_IL = (
            subject_data
            .select('DateAndMinute', 'not_in_israel')
            .with_columns(
                not_in_IL_num=pl.when(pl.col('not_in_israel') == True)
                .then(pl.lit(max_bpm))
                .when(pl.col('not_in_israel') == False)
                .then(pl.lit(min_bpm))
                .otherwise(pl.lit(np.nan))
            )
        )

        fig_bpm.add_trace(go.Scatter(
            x=not_in_IL['DateAndMinute'],
            y=not_in_IL['not_in_IL_num'],
            mode='lines',
            name='Not in IL',
            visible='legendonly',
            customdata=not_in_IL.select('not_in_israel').to_pandas(),
            hovertemplate='Time: %{x} <br> BpmMean: %{y} <br> Not in IL: %{customdata}'
        )
        )

        fig_bpm.add_trace(go.Scatter(
            x=subject_data['DateAndMinute'], 
            y=subject_data['BpmMean'], 
            mode='lines', 
            name='BpmMean',
            customdata=custom_data.to_pandas(),
            hovertemplate='Time: %{x} <br> BpmMean: %{y} <br> Sleep Stage: %{customdata[1]} <br> Feature: %{customdata[2]} <br> Weekend: %{customdata[4]} <br> Outliers: %{customdata[5]} <br> Not in IL: %{customdata[6]}'
        ))

        select_segments_to_exclude_slider = dbc.Button(
            'Select Segments to Exclude',
            id={
                'type': 'select-segments-to-exclude-button-TS',
                'index': i
            },
            color='Warning',
            n_clicks=0
        )

        # Add the figures to the card
        sub_card = dbc.Card(
            id={'type': 'card-TS', 
                'index': i
            },
            children=[
                dbc.CardHeader(f'Subject: {subject}'),
                dbc.CardBody(
                    [
                        dcc.Graph(figure=fig_bpm),
                    ],
                    id={
                        'type': 'card-body-TS1',
                        'index': i
                    }
                ),
                dbc.CardBody(
                    [
                        select_segments_to_exclude_slider
                    ],
                    id={
                        'type': 'card-body-TS',
                        'index': i
                    }
                )
            ]
        )
        

        cards.append(sub_card)

    return cards 




        



@callback(
    Output('raw-data-subjects-container-TS', 'children'),
    Output('subjects-TS', 'data'),
    Input({'type': 'show-selected-button-TS', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-TS', 'value'),
    State({'type': 'select-subjects-grid-TS', 'index': ALL}, 'rowData')
)
def load_subjects(n_clicks, project, rows):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    # Get the subjects
    minute_reso_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', 'All_Subjects_1_Minute_Resolution.parquet')

    if not minute_reso_path.exists():
        return dbc.Alert('The file does not exist. Please generate it first in the combining files page.', color='danger'), []
    
    df = pl.read_parquet(minute_reso_path)

    if df.is_empty():
        return dbc.Alert('The file is empty.', color='danger'), []
    
    selected_subjects = [row['Id'] for row in rows[0] if row['show']]

    print(selected_subjects)

    if not selected_subjects:
        return dbc.Alert('Please select at least one subject.', color='danger'), []
    
    df = df.filter(pl.col('Id').is_in(selected_subjects))




    cards = make_cards(df, selected_subjects)



    return cards , selected_subjects


@callback(
    Output('raw-data-subjects-container-TS', 'children', allow_duplicate=True),
    Output('subjects-TS', 'data', allow_duplicate=True),
    Input({'type': 'hide-selected-button-TS', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-TS', 'value'),
    prevent_initial_call=True
)
def hide_plots(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    cards = []

    return cards, []


@callback(
    Output('raw-data-subjects-container-TS', 'children', allow_duplicate=True),
    Output('subjects-TS', 'data', allow_duplicate=True),
    Input({'type': 'show-all-button-TS', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-TS', 'value'),
    prevent_initial_call=True
)
def show_all(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    if n_clicks[0] == 0:
        raise PreventUpdate
    path = Pconfigs[project]

    # Get the subjects
    minute_reso_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', 'All_Subjects_1_Minute_Resolution.parquet')

    if not minute_reso_path.exists():
        return dbc.Alert('The file does not exist. Please generate it first in the combining files page.', color='danger'), []
    
    df = pl.read_parquet(minute_reso_path)

    subjects = df['Id'].unique().to_list()

    if df.is_empty():
        return dbc.Alert('The file is empty.', color='danger'), []
    
    cards = make_cards(df, subjects)

    return cards, subjects




@callback(
    Output({'type': 'card-body-TS', 'index': MATCH}, 'children'),
    Input({'type': 'select-segments-to-exclude-button-TS', 'index': MATCH}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-TS', 'value'),
    State({'type': 'card-body-TS', 'index': MATCH}, 'children'),
    State('subjects-TS', 'data'),
    prevent_initial_call=True
)
def select_segments_to_exclude(n_clicks, project, children, subjects):
    if n_clicks == 0:
        raise PreventUpdate
    
    print(f'children: {children}')
    print(f'n_clicks: {n_clicks}')
    
    slider = dcc.RangeSlider(
        id={
            'type': 'range-slider-TS',
            'index': n_clicks
        },
        min=0,
        max=100,
        step=1,
        marks={i: str(i) for i in range(0, 101, 10)},
        value=[0, 100]
    )
    
    accept_button = dbc.Button(
        'Save',
        id={
            'type': 'accept-button-TS',
            'index': n_clicks
        },
        color='success',
        n_clicks=0,
        className='d-block'
    )

    new_children = children + [slider, accept_button]

    return new_children
