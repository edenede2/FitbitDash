from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import bioread as br
import h5py
import plotly.express as px
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





dash.register_page(__name__, name='Stats Visualization', order=6)

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
                    html.H1("Stats Visualization Page"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-Visualization',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-Visualization', n_clicks=0, color='primary')
                ]),

                html.Hr()
            ]),
        ]),
        dbc.Container(
            id='select-subjects-container-Visualization',
            className='ny-5',
            fluid=True
        ),            
        dbc.Container(
            id='raw-data-subjects-container-Visualization',
            className='ny-5',
            fluid=True
        ),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-Visualization',
                                message=''
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-Visualization',
                        message=''
                    ),
                
                ]),
            ]),
        

])



@callback(
    Output('select-subjects-container-Visualization', 'children'),
    Input('load-fitbit-button-Visualization', 'n_clicks'),
    State('project-selection-dropdown-FitBit-Visualization', 'value')
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
            'type': 'select-subjects-grid-Visualization',
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
            'type': 'show-selected-button-Visualization',
            'index': 1
        },
        color='primary',
        n_clicks=0
    )

    hide_selected_button = dbc.Button(
        'Hide Plots',
        id={
            'type': 'hide-selected-button-Visualization',
            'index': 1
        },
        color='primary',
        n_clicks=0
    )

    show_all_button = dbc.Button(
        'Show All',
        id={
            'type': 'show-all-button-Visualization',
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




def create_cards(df, subjects=None):

    cards = []

    df = df.sort('Id')

    print(subjects)

    for subject in df['Id'].unique().to_list():
        
        if subjects and subject not in subjects:
            continue
        print(subject)
        hr_data = (
            df
            .filter(
                pl.col('Id') == subject
                )
            .select(
                ['Id', 'DateAndMinute', 'BpmMean']
                )
            .sort('DateAndMinute')
        )

        hr_data_total_minutes = (
            hr_data
            .with_columns(
                Date = pl.col('DateAndMinute').dt.truncate('1d')
            )
            .group_by('Date')
            .agg(
                total_minutes = pl.count()
            )
        )

        hr_data = (
            hr_data
            .with_columns(
                (pl.col('BpmMean').is_null()).alias('gap')
            )
        )

        hr_data = (
            hr_data
            .with_columns(
                (pl.col('gap') != pl.col('gap').shift(1)).cum_sum().alias('gap_group')
            )
        )

        gap_sizes_hr = (
            hr_data
            .filter(
                pl.col('gap')
            )
            .group_by('gap_group')
            .agg(
                pl.count().name.suffix('_count')
            )
        )

        gap_sizes_hr_counts = (
            gap_sizes_hr
            .group_by('len_count')
            .agg(
                pl.len()
            )
            .sort('len')
            .with_columns(
                pl.col('len_count').cast(pl.String)
            )
        )

        hr_data = (
            hr_data
            .with_columns(
                pl.col('DateAndMinute').dt.truncate('1d').alias('Date')
            )
        )

        missing_per_day_hr = (
            hr_data
            .group_by('Date')
            .agg(
                missing_percentage = (pl.col('BpmMean').is_null().mean() * 100)
            )
            .join(
                hr_data_total_minutes,
                on='Date',
                how='inner'
            )
            .with_columns(
                missing_minutes = pl.col('total_minutes') * pl.col('missing_percentage') / 100
            )
        )

        mean_missing_percentage_hr = (
            missing_per_day_hr
            .select(
                pl.col('missing_percentage').mean()
            ) 
        ).item()

        gap_sizes_counts_hr_pd = gap_sizes_hr_counts.to_pandas()
        missing_per_day_hr_pd = missing_per_day_hr.to_pandas()

        fig1 = px.bar(
            gap_sizes_counts_hr_pd,
            x='len',
            y='len_count',
            orientation='h',
            labels={'len': 'Occurences', 'len_count': 'Gap Size'},
            title='Gap Sizes of HR Samples',
        )

        fig2 = px.bar(
            missing_per_day_hr_pd,
            x='Date',
            y='missing_percentage',
            labels={'Date': 'Date', 'missing_percentage': 'Missing Percentage'},
            title='Missing HR Samples per Day',
            hover_data={
                'missing_minutes': True,  # Ensure missing_minutes is displayed
                'total_minutes': True     # Ensure total_minutes is displayed
            }
        )

        fig2.update_layout(
            yaxis=dict(
                range=[0, 100]
            )
        )

        fig2.add_shape(
            type="line",
            x0=missing_per_day_hr_pd['Date'].min(),
            x1=missing_per_day_hr_pd['Date'].max(),
            y0=mean_missing_percentage_hr,
            y1=mean_missing_percentage_hr,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            ),
            xref="x",
            yref="y"
        )

        # Add annotation for the mean line
        fig2.add_annotation(
            x=missing_per_day_hr_pd['Date'].max(),
            y=mean_missing_percentage_hr,
            text=f'Mean: {mean_missing_percentage_hr:.2f}%',
            showarrow=False,
            yshift=10,
            font=dict(color="Red", size=12)
        )

        sleep_data = (
            df
            .filter(
                pl.col('Id') == subject
            )
            .select(
                ['Id', 'DateAndMinute', 'Mode']
            )
        )

        sleep_data = (
            sleep_data
            .with_columns(
                Mode_bin = pl.when(pl.col('Mode') == 'awake')
                .then(0)
                .otherwise(1),
                Date = pl.col('DateAndMinute').dt.truncate('1d')
            )
        )

        number_of_sleeps = (
            sleep_data
            .group_by('Date')
            .agg(
                number_of_sleeps = pl.sum('Mode_bin')
            )
        )

        number_of_sleeps = (
            number_of_sleeps
            .with_columns(
                number_of_sleeps_bin = pl.when(pl.col('number_of_sleeps') > 0)
                .then(1)
                .otherwise(0)
            )
            .with_columns(
                number_of_sleeps_str = pl.when(pl.col('number_of_sleeps_bin') == 0)
                .then(pl.lit('Missing'))
                .when(pl.col('number_of_sleeps_bin') == 1)
                .then(pl.lit('Exists'))
            )
        )

        number_of_sleeps_pd = number_of_sleeps.to_pandas()

        fig3 = px.pie(
            number_of_sleeps_pd,
            names='number_of_sleeps_str',
            title='Missing Sleeps',
            color='number_of_sleeps_str',  # Align the color parameter with names
            color_discrete_map={'Missing': 'red', 'Exists': 'blue'},
            
        )

        fig3.update_traces(textinfo='percent+label')

        sub_card = dbc.Card([
            html.H5(f'Subject: {subject}'),
            dbc.Row([
                dbc.Col([
                    html.Div(['Amount of gaps per size and number of occurences']),
                    dcc.Graph(figure=fig1)]),
   
                dbc.Col([
                    html.Div(['Percentage of missing minutes of HR samples per day']),
                    dcc.Graph(figure=fig2)]),

                dbc.Col([
                    html.Div(['Percentage of days with sleep from the total days of experiment']),
                    dcc.Graph(figure=fig3)])
            ]),

        ])

        cards.append(sub_card)

    return cards




        




@callback(
    Output('raw-data-subjects-container-Visualization', 'children'),
    Input({'type': 'show-selected-button-Visualization', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-Visualization', 'value'),
    State({'type': 'select-subjects-grid-Visualization', 'index': ALL}, 'rowData'),
)
def load_subjects(n_clicks, project, rows):
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
    
    select_subjects = [row['Id'] for row in rows[0] if row['show']]

    if not select_subjects:
        return dbc.Alert('No subjects selected.', color='danger')
    
    df = df.filter(pl.col('Id').is_in(select_subjects))
    print(f'Selected subjects: {select_subjects}')
    cards = create_cards(df, select_subjects)




    return cards



@callback(
    Output('raw-data-subjects-container-Visualization', 'children', allow_duplicate=True),
    Input({'type': 'hide-selected-button-Visualization', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-Visualization', 'value'),
    prevent_initial_call=True
)
def hide_plots(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    cards = []

    return cards



@callback(
    Output('raw-data-subjects-container-Visualization', 'children', allow_duplicate=True),
    Input({'type': 'show-all-button-Visualization', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-Visualization', 'value'),
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
        return dbc.Alert('The file does not exist. Please generate it first in the combining files page.', color='danger')
    
    df = pl.read_parquet(minute_reso_path)

    if df.is_empty():
        return dbc.Alert('The file is empty.', color='danger')
    
    cards = create_cards(df)

    return cards
