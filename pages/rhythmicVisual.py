from flask import session
from dash import dcc, html, Dash, dependencies, dash_table, Input, Output, State, Patch, MATCH, ALL, callback
import pickle as pkl
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
import scripts.UTILS.utils as ut

# from Test.Remove_sub.utils import get_latest_file_by_term as new_get_latest_file_by_term
import warnings



FIRST, LAST = 0, -1
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') # for the output files





dash.register_page(__name__, name='Rhythmic Visualization', order=9)

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
                    html.H1("Rhythmic Visualization Page"),
                    html.Hr()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div('Please select the project:'),
                    dcc.Dropdown(
                        id='project-selection-dropdown-FitBit-RV',
                        options=[
                            {'label': key, 'value': value} for key, value in pages.items()
                        ],
                        style={'color': 'black'},
                        value='FIBRO_TESTS'
                    )
                ]),
                dbc.Col([
                    dbc.Button('Load ', id='load-fitbit-button-RV', n_clicks=0, color='primary')
                ]),

                html.Hr()
            ]),
        ]),
        dbc.Container(
            id='select-cosinor-file',
            className='ny-5',
            fluid=True
        ),
        dbc.Container(
            id='select-subjects-container-RV',
            className='ny-5',
            fluid=True
        ),
        dbc.Container(
            id='raw-data-subjects-container-RV',
            className='ny-5',
            fluid=True
            
        ),
        dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        [
                            dcc.ConfirmDialog(
                                id='confirm-dialog-RV',
                                message=''
                            )
                        ],
                        overlay_style = {'visibility': 'visible', 'opacity': 0.5, 'background': 'white', 'filter': 'blur(2px)'},
                        custom_spinner = html.H2(['Processing File...', dbc.Spinner(color='primary')])
                    ),
                    dcc.ConfirmDialog(
                        id='error-gen-dialog-RV',
                        message=''
                    ),
                
                ]),
            ]),
            dcc.Store(id='subjects-RV', data=[]),
            dcc.Store(id='selected-cosinor-file-name', data=''),
            dcc.Store(id='polar-fig-wind', data={}),
            dcc.Store(id='cartes-fig-wind', data={}),
        

])



@callback(
    Output('select-cosinor-file', 'children'),
    Input('load-fitbit-button-RV', 'n_clicks'),
    State('project-selection-dropdown-FitBit-RV', 'value')
)
def select_cosinor_file(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    cosinor_folder_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output')

    if not cosinor_folder_path.exists():
        return dbc.Alert('The folder does not exist. Please generate it first in the combining files page.', color='danger')
    else:

        file_pattern = r'^all_subjects_cosinor_w'
        cosinor_files = [file.split('/')[-1] for file in os.listdir(cosinor_folder_path) if re.search(file_pattern, file.split('/')[-1])]

        # cosinor_files = list(cosinor_folder_path.glob('^all_subjects_cosinor_w'))

        if not cosinor_files:
            return dbc.Alert('There is no cosinor concatenate files.', color='danger')
        
        cosinor_files = [file for file in cosinor_files if file.endswith('.csv')]


        cosinor_files_dropdown = dcc.Dropdown(
            id={'type':'cosinor-files-dropdown-RV', 'index': 1},
            options=[
                {'label': file, 'value': file} for file in cosinor_files
            ],
            style={'color': 'black'},
            value=cosinor_files[0]
        )

        loading_button = dbc.Button('Load', id={'type': 'load-cosinor-button-RV', 'index': 1}, n_clicks=0, color='primary')

        return [
            html.Div('Please select the Cosinor file:'),
            cosinor_files_dropdown,
            loading_button
        ]
        



@callback(
    Output('select-subjects-container-RV', 'children'),
    Output('selected-cosinor-file-name', 'data'),
    # Input('load-fitbit-button-RV', 'n_clicks'),
    Input({'type': 'load-cosinor-button-RV', 'index': 1}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-RV', 'value'),
    State({'type':'cosinor-files-dropdown-RV', 'index': ALL}, 'value')
)
def select_subjects(n_clicks, project, file):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    file = file[0]

    params = {}

    interval_size = 24
    downsample_rate = 5

    file_partitioned = file.split('_')
    for i in file_partitioned:
        if i.startswith('w'):
            params['interval size'] = i[1:]
            interval_size = i[1:]
        if i.startswith('incr'):
            params['increment'] = i[4:]
        if i.startswith('ds'):
            params['downsample rate'] = i[2:]
            downsample_rate = i[2:]
        if i.startswith('mThr'):
            params['missing threshold'] = i[10:]
        if i.startswith('True'):
            params['interpolate'] = True
        if i.startswith('False'):
            params['interpolate'] = False

    params_to_store = str(interval_size) + '_' + str(downsample_rate)

    # Get the subjects
    cosinor_file_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', file)

    if not cosinor_file_path.exists():
        return dbc.Alert('The file does not exist. Please generate it first in the rhythmic page.', color='danger'), ''
    
    df = pl.read_csv(cosinor_file_path, try_parse_dates=True)

    if df.is_empty():
        return dbc.Alert('The file is empty.', color='danger'), ''
    
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
            'type': 'select-subjects-grid-RV',
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
            'type': 'show-selected-button-RV',
            'index': 1
        },
        color='primary',
        n_clicks=0
    )

    hide_selected_button = dbc.Button(
        'Hide Plots',
        id={
            'type': 'hide-selected-button-RV',
            'index': 1
        },
        color='primary',
        n_clicks=0
    )

    show_all_button = dbc.Button(
        'Show All',
        id={
            'type': 'show-all-button-RV',
            'index': 1
        },
        color='success',
        n_clicks=0
    )

    params_description = dbc.Card(
        dbc.CardBody(
            [
                html.H4('Parameters:'),
                html.P(f'Interval Size: {params["interval size"]}'),
                html.P(f'Increment: {params["increment"]}'),
                html.P(f'Downsample Rate: {params["downsample rate"]}'),
                html.P(f'Missing Threshold: {params["missing threshold"]}'),
                html.P(f'Interpolate: {params["interpolate"]}')
            ]
        )
    )

    return [
        dbc.Row([
            dbc.Col([
                grid
            ]),
            dbc.Col([
                params_description,
                show_selected_button,
                hide_selected_button,
                show_all_button
            ])
        ])
    ], params_to_store




def make_cards(df_est:pl.DataFrame, df_visu:pl.DataFrame, desc_dict:dict ,subjects=None, params_to_store=None):
    cards = []

    df_est = df_est.sort('Id')
    df_visu = df_visu.sort('Id')
    visu_dict = {}

    downsample_rate = int(params_to_store.split('_')[1])
    period_size = int(params_to_store.split('_')[0]) *60 / downsample_rate
    

    print(r'start iterating through subjects')
    for i, subject in zip(range(0,len(subjects)) ,subjects):
        sub_cards = []


        if subjects and subject not in subjects:
            continue

        subject_est_data = (
            df_est
            .filter(
                pl.col('Id') == subject
                )
        )

        subject_visu_data = (
            df_visu
            .filter(
                pl.col('Id') == subject
                )
        )

        subject_desc = desc_dict[subject]

        # Create line plot for BpmMean
        fig_cartes = go.Figure()
        visu_dict[subject] = {}
        print(f'visu_dict: {visu_dict}')
        tqdm_dates = tqdm(subject_est_data['test'].unique().to_list(), desc='Iterating through dates', position=0, leave=True)
        for date in tqdm_dates:
            tqdm_dates.set_description(f'Interval: {date}')
            est_date_data = subject_est_data.filter(pl.col('test') == date)
            visu_date_data = subject_visu_data.filter(pl.col('test') == date)
            date_desc = subject_desc[date]

            x_data = visu_date_data['x']
            y_data = visu_date_data['y']
            y_interpolated = visu_date_data['interpolated_y']

            amplitude = date_desc['amplitude'] + date_desc['mesor']
            mesor = date_desc['mesor']
            acrophase = date_desc['acrophase']

            adj_r2 = date_desc['r_squared_adj']
            r_squared = date_desc['r_squared']

            x_est = est_date_data['estimated_x']
            y_est = est_date_data['estimated_y']

            y_est_max_loc = date_desc['y_estimated_max_loc']
            y_est_min_loc = date_desc['y_estimated_min_loc']
            y_estimated_min = date_desc['y_estimated_min']
            theta = date_desc['theta']


            visu_dict[subject][date] = {
                'x': x_data,
                'y': y_data,
                'y_interpolated': y_interpolated,
                'x_est': x_est,
                'y_est': y_est,
                'y_est_max_loc': y_est_max_loc,
                'y_est_min_loc': y_est_min_loc,
                'y_estimated_min': y_estimated_min,
                'r_squared': r_squared,
                'adj_r2': adj_r2,
                'amplitude': amplitude,
                'mesor': mesor,
                'acrophase': acrophase,
                'theta': theta,
                'period_size': period_size,
                'downsample_rate': downsample_rate
            }
            # if y_interpolated is not None:
            #     fig_cartes.add_trace(go.Scatter(x=x_data, y=y_interpolated, mode='markers', name='Interpolated Data'))

            # fig_cartes.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Original Data'))
            # fig_cartes.add_trace(go.Scatter(x=x_est, y=y_est, mode='lines', name='Estimated Curve'))
            # fig_cartes.add_trace(go.Scatter(x=x_est, y=[mesor]*len(x_est), mode='lines', name='MESOR'))
            # fig_cartes.add_shape(
            #     type='line',
            #     x0=y_est_max_loc,
            #     y0=mesor,
            #     x1=y_est_max_loc,
            #     y1=amplitude,
            #     line=dict(
            #         color='red',
            #         width=2,
            #         dash='dashdot'
            #     )
            # )

            # fig_cartes.add_shape(
            #     type='line',
            #     x0=y_est_min_loc,
            #     y0=mesor,
            #     x1=y_est_min_loc,
            #     y1=y_estimated_min,
            #     line=dict(
            #         color='red',
            #         width=2,
            #         dash='dashdot'
            #     )
            # )

            # fig_cartes.add_annotation(
            #     x=y_est_max_loc,
            #     y=amplitude,
            #     text='Max',
            #     showarrow=True,
            #     arrowhead=1
            # )

            # fig_cartes.add_annotation(
            #     x=y_est_min_loc,
            #     y=y_estimated_min,
            #     text='Min',
            #     showarrow=True,
            #     arrowhead=1
            # )

            # fig_cartes.update_layout(
            #     title=f'Subject: {subject} Interval: {date}',
            #     xaxis_title='Time (absolute)',
            #     yaxis_title='Signal',
            #     showlegend=True
            # )

            # fig_cartes.update_layout(
            #     annotations=[
            #         dict(
            #             x=0.5,
            #             y=1.1,
            #             xref='paper',
            #             yref='paper',
            #             text=f'R^2: {r_squared} Adj R^2: {adj_r2}',
            #             showarrow=False
            #         )
            #     ]
            # )

            # theta = date_desc['theta']
            # acrophase = quadrant_adjustment(theta, date_desc['acrophase'], radian=True)

            # amplitude = date_desc['amplitude']
            # mesor = date_desc['mesor']


            # center_r = [0, amplitude]
            # center_theta = [0, acrophase]

            # fig_polar = go.Figure()
            # fig_polar.add_trace(go.Scatterpolar(
            #     r=[amplitude],
            #     theta=[acrophase],
            #     mode='markers',
            #     name='Acrophase',
            #     marker=dict(
            #         size=10,
            #         color='red'
            #     )
            # ))

            # fig_polar.add_trace(go.Scatterpolar(
            #     r=center_r,
            #     theta=center_theta,
            #     mode='lines',
            #     line=dict(
            #         color='green',
            #         width=2
            #     ),
            #     name='Radius Line'
            # ))

            # hours, hours_deg = generate_polarticks(period_size, period_size)

            # fig_polar.update_layout(
            #     title=f'Subject: {subject} Interval: {date}',
            #     polar=dict(
            #         angularaxis=dict(
            #             tickmode='array',
            #             tickvals=hours_deg,
            #             ticktext=hours,
            #             direction='clockwise',
            #             rotation=0,
            #             thetaunit='degrees'
            #         ),
            #     )
            # )

            # sub_cards.append(
            #     dbc.Card(
            #         children=[
            #             dbc.CardHeader(f'Subject: {subject} Interval: {date}'),
            #             dbc.CardBody(
            #                 [
            #                     dcc.Graph(figure=fig_cartes),
            #                 ],
            #                 id={ 'type': 'card-body-RV1', 'index': i}
            #             ),
            #             dbc.CardBody(
            #                 [
            #                     dcc.Graph(figure=fig_polar),
            #                 ],
            #                 id={ 'type': 'card-body-RV2', 'index': i}
            #             )
            #         ]
            #     )
            # )

            



            # cards.append(sub_cards)
        print(f'done with subject {subject}')
    return visu_dict




def quadrant_adjustment(thta, acrphs, radian=True):
    # Check which quadrant the acrophase falls into
    if 0 <= thta < (np.pi / 2):
        if radian:
            corrected_acrophase = acrphs
        else:
            # First quadrant: no correction needed
            corrected_acrophase = np.rad2deg(acrphs)
    elif (np.pi / 2) <= thta < np.pi:
        # Second quadrant: subtract a constant to realign
        if radian:
            corrected_acrophase = acrphs 
        else:
            corrected_acrophase =  np.rad2deg(acrphs)
    elif np.pi <= thta < (3 * np.pi / 2):
        # Third quadrant: make it negative
        if radian:
            corrected_acrophase = 2 * np.pi - acrphs
        else:
            corrected_acrophase = 360 - np.rad2deg(acrphs)
    elif (3 * np.pi / 2) <= thta < (2 * np.pi):
        if radian:
            corrected_acrophase = 2 * np.pi - acrphs
        else:
            # Fourth quadrant: shift to bring into biological range
            corrected_acrophase = 360 - np.rad2deg(acrphs)
    else:
        # If outside normal bounds, wrap it
        corrected_acrophase = acrphs % (2 * np.pi)

    return corrected_acrophase
        


def generate_polarticks(period, select_period_size, half_day = False):
    total_hours = period

    num_ticks = 12

    tick_interval = select_period_size / num_ticks

    hours = []
    hours_deg = []

    if (select_period_size /24) < 1 :
        for i in range(num_ticks + 1):
            hour = i * tick_interval
            deg = (i * 360) / num_ticks
            hour_int = int(hour % 24)
            label = f"{hour_int:02d}:00"
            hours.append(label)
            hours_deg.append(deg)
    else:
        for i in range(num_ticks +1):
            hour = i * tick_interval
            deg = (i * 360) / num_ticks
            if half_day:
                hour_int = int(hour % 24) + 12
            else:
                hour_int = int(hour % 24)
            day = int(hour // 24) + 1
            label = f"Day {day} - {hour_int:02d}:00"
            hours.append(label)
            hours_deg.append(deg)

    return hours, hours_deg

@callback(
    Output('raw-data-subjects-container-RV', 'children'),
    Output('subjects-RV', 'data'),
    Output('polar-fig-wind', 'data'),
    Output('cartes-fig-wind', 'data'),
    Input({'type': 'show-selected-button-RV', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-RV', 'value'),
    State({'type': 'select-subjects-grid-RV', 'index': ALL}, 'rowData'),
    State({'type': 'cosinor-files-dropdown-RV', 'index': ALL}, 'value'),
    State('selected-cosinor-file-name', 'data'),
)
def load_subjects(n_clicks, project, rows, file, params_to_store):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    file = file[0]

    # Get the subjects
    estimated_curve_file_name = (str(file).replace('cosinor_', 'estimates_')).replace('.csv', '.parquet')
    estimated_curve_file_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', estimated_curve_file_name)

    visu_file_name = (str(file).replace('cosinor_', 'visu_')).replace('.csv', '.parquet')
    visu_file_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', visu_file_name)

    desc_json_file_name = (str(file).replace('cosinor_', 'json_')).replace('.csv', '.json')
    desc_json_file_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', desc_json_file_name)

    if not estimated_curve_file_path.exists():
        return dbc.Alert('The file estimated curve file does not exist. Please generate it first in the rhythmic page.', color='danger'), []
    
    if not visu_file_path.exists():
        return dbc.Alert('The file visu file does not exist. Please generate it first in the rhythmic page.', color='danger'), []
    
    if not desc_json_file_path.exists():
        return dbc.Alert('The file description json file does not exist. Please generate it first in the rhythmic page.', color='danger'), []
    
    estimated_df = pl.read_parquet(estimated_curve_file_path)
    visu_df = pl.read_parquet(visu_file_path)
    desc_json = json.load(open(desc_json_file_path, 'r'))
    

    if estimated_df.is_empty() or visu_df.is_empty():
        return dbc.Alert('One of the files is empty.', color='danger'), []
    
    selected_subjects = [row['Id'] for row in rows[0] if row['show']]


    print(selected_subjects)

    
    estimated_sub_df = estimated_df.filter(pl.col('Id').is_in(selected_subjects))
    visu_sub_df = visu_df.filter(pl.col('Id').is_in(selected_subjects))

    subs_desc_dict = {subject: desc_json[subject] for subject in selected_subjects}



    print('start making cards')
    cards_dict = make_cards(estimated_sub_df, visu_sub_df, subs_desc_dict, selected_subjects, params_to_store)
    with open('cards_dict.pkl', 'wb') as f:
        pickle.dump(cards_dict, f)
    cards = []
    annoncement = f'The subjects are {selected_subjects} are ready to be visualized.'

    for sub in cards_dict.keys():
        for i,date in enumerate(cards_dict[sub].keys()):
            visu_data = cards_dict[sub][date]
            fig_cartes = go.Figure()
            x_data = visu_data['x']
            y_data = visu_data['y']
            y_interpolated = visu_data['y_interpolated']

            amplitude = visu_data['amplitude']
            mesor = visu_data['mesor']
            acrophase = visu_data['acrophase']
            theta = visu_data['theta']

            period_size = visu_data['period_size']
            downsample_rate = visu_data['downsample_rate']

            adj_r2 = visu_data['r_squared']
            r_squared = visu_data['adj_r2']

            x_est = visu_data['x_est']
            y_est = visu_data['y_est']

            y_est_max_loc = x_est.to_numpy()[np.argmax(y_est.to_numpy())]
            y_est_min_loc = x_est.to_numpy()[np.argmin(y_est.to_numpy())]
            y_estimated_min = np.min(y_est.to_numpy())



            if y_interpolated is not None:
                fig_cartes.add_trace(go.Scatter(x=x_data, y=y_interpolated, mode='markers', name='Interpolated Data'))

            fig_cartes.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Original Data'))
            fig_cartes.add_trace(go.Scatter(x=x_est, y=y_est, mode='lines', name='Estimated Curve'))
            fig_cartes.add_trace(go.Scatter(x=x_est, y=[mesor]*len(x_est), mode='lines', name='MESOR'))
            fig_cartes.add_shape(
                type='line',
                x0=y_est_max_loc,
                y0=mesor,
                x1=y_est_max_loc,
                y1=amplitude,
                line=dict(
                    color='red',
                    width=2,
                    dash='dashdot'
                )
            )

            fig_cartes.add_shape(
                type='line',
                x0=y_est_min_loc,
                y0=mesor,
                x1=y_est_min_loc,
                y1=y_estimated_min,
                line=dict(
                    color='red',
                    width=2,
                    dash='dashdot'
                )
            )

            fig_cartes.add_annotation(
                x=y_est_max_loc,
                y=amplitude,
                text='Max',
                showarrow=True,
                arrowhead=1
            )

            fig_cartes.add_annotation(
                x=y_est_min_loc,
                y=y_estimated_min,
                text='Min',
                showarrow=True,
                arrowhead=1
            )

            fig_cartes.update_layout(
                title=f'Subject: {sub} Interval: {date}',
                xaxis_title='Time (absolute)',
                yaxis_title='Signal',
                showlegend=True
            )

            fig_cartes.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=1.1,
                        xref='paper',
                        yref='paper',
                        text=f'R^2: {r_squared} Adj R^2: {adj_r2}',
                        showarrow=False
                    )
                ]
            )

            theta = visu_data['theta']
            acrophase = quadrant_adjustment(theta, visu_data['acrophase'], radian=True)

            amplitude = visu_data['amplitude']
            mesor = visu_data['mesor']


            center_r = [0, amplitude]
            center_theta = [0, acrophase]

            fig_polar = go.Figure()
            fig_polar.add_trace(go.Scatterpolar(
                r=[amplitude],
                theta=[acrophase],
                mode='markers',
                name='Acrophase',
                marker=dict(
                    size=10,
                    color='red'
                )
            ))

            fig_polar.add_trace(go.Scatterpolar(
                r=center_r,
                theta=center_theta,
                mode='lines',
                line=dict(
                    color='green',
                    width=2
                ),
                name='Radius Line'
            ))

            hours, hours_deg = generate_polarticks(period_size, period_size)

            fig_polar.update_layout(
                title=f'Subject: {sub} Interval: {date}',
                polar=dict(
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=hours_deg,
                        ticktext=hours,
                        direction='clockwise',
                        rotation=0,
                        thetaunit='degrees'
                    ),
                )
            )

            cards.append(
                dbc.Card( id={'type': 'card-RV', 'index': i},
                    children=[
                        dbc.CardHeader(f'Subject: {sub} Interval: {date}'),
                        dbc.CardBody(
                            [
                                dcc.Graph(id={'type': 'fig-cartes-RV', 'index':i}, figure=fig_cartes),
                                dbc.Button('Next Interval', id={'type': 'next-cartes-button-RV', 'index': i}, n_clicks=0),
                                dbc.Button('Previous Interval', id={'type': 'previous-cartes-button-RV', 'index': i}, n_clicks=0)
                            ],
                            id={ 'type': 'card-body-RV1', 'index': i}
                        ),
                        dbc.CardBody(
                            [
                                dcc.Graph(id={'type': 'fig-polar-RV', 'index':i},figure=fig_polar),
                                dbc.Button('Next Interval', id={'type': 'next-polar-button-RV', 'index': i}, n_clicks=0),
                                dbc.Button('Previous Interval', id={'type': 'previous-polar-button-RV', 'index': i}, n_clicks=0)
                            ],
                            id={ 'type': 'card-body-RV2', 'index': i}
                        ),
                        dcc.Store(id={'type': 'polar-fig-wind', 'index': i}, data={}),
                        dcc.Store(id={'type': 'cartes-fig-wind', 'index': i}, data={})
                    ]
                )
            )
            break
    polar_fig_wind = {}
    cartes_fig_wind = {}
    for sub in selected_subjects:
        polar_fig_wind[sub] = 0
        cartes_fig_wind[sub] = 0
    print(f'returning cards and selected subjects {selected_subjects}')
    return cards , selected_subjects, polar_fig_wind, cartes_fig_wind

@callback(
    Output({'type': 'card-RV', 'index': MATCH}, 'children'),
    Output({'type': 'polar-fig-wind', 'index': MATCH}, 'data'),
    Output({'type': 'cartes-fig-wind', 'index': MATCH}, 'data'),
    Input({'type': 'next-cartes-button-RV', 'index': MATCH}, 'n_clicks'),
    Input({'type': 'previous-cartes-button-RV', 'index': MATCH}, 'n_clicks'),
    Input({'type': 'next-polar-button-RV', 'index': MATCH}, 'n_clicks'),
    Input({'type': 'previous-polar-button-RV', 'index': MATCH}, 'n_clicks'),
    State({'type': 'card-RV', 'index': MATCH}, 'children'),
    State('subjects-RV', 'data'),
    State({'type': 'polar-fig-wind', 'index': MATCH}, 'data'),
    State({'type': 'cartes-fig-wind', 'index': MATCH}, 'data'),
    State({'type': 'cosinor-files-dropdown-RV', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def update_card_plots(nc_next, nc_prev, np_next, np_prev,
                      current_card_body, subjects,
                      polar_wind_store, cartes_wind_store, file):
    # ...existing code...
    ctx = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    card_index = dash.callback_context.triggered_id['index']
    subject = subjects[0]  # adjust as needed

    if subject not in polar_wind_store:
        polar_wind_store[subject] = 0
    if subject not in cartes_wind_store:
        cartes_wind_store[subject] = 0
    
    with open('cards_dict.pkl', 'rb') as f:
        cards_dict = pickle.load(f)

    if 'next-polar-button-RV' in ctx or 'previous-polar-button-RV' in ctx:
        # For polar vs cartes
        current_idx = polar_wind_store[subject]
        if 'next-polar-button-RV' in ctx:
            current_idx += 1
        else:
            current_idx -= 1
        possible_keys = list(cards_dict[subject].keys())
        current_idx %= len(possible_keys)
        polar_wind_store[subject] = current_idx
        date_key = possible_keys[current_idx]
        visu_data = cards_dict[subject][date_key]
        new_polar_fig = go.Figure()
        # ...build new polar figure from visu_data...
        new_card_body = current_card_body[:]
        new_card_body[2]['props']['children'][0]['props']['figure'] = new_polar_fig
        return new_card_body, polar_wind_store, cartes_wind_store

    # Otherwise, cartes
    current_idx = cartes_wind_store[subject]
    if 'next-cartes-button-RV' in ctx:
        current_idx += 1
    elif 'previous-cartes-button-RV' in ctx:
        current_idx -= 1
    possible_keys = list(cards_dict[subject].keys())
    current_idx %= len(possible_keys)
    cartes_wind_store[subject] = current_idx
    date_key = possible_keys[current_idx]
    visu_data = cards_dict[subject][date_key]
    new_cartes_fig = go.Figure()
    # ...build new cartesian figure from visu_data...
    new_card_body = current_card_body[:]
    new_card_body[1]['props']['children'][0]['props']['figure'] = new_cartes_fig
    return new_card_body, polar_wind_store, cartes_wind_store

@callback(
    Output('raw-data-subjects-container-RV', 'children', allow_duplicate=True),
    Output('subjects-RV', 'data', allow_duplicate=True),
    Input({'type': 'hide-selected-button-RV', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-RV', 'value'),
    prevent_initial_call=True
)
def hide_plots(n_clicks, project):
    if n_clicks == 0:
        raise PreventUpdate
    
    path = Pconfigs[project]

    cards = []

    return cards, []


@callback(
    Output('raw-data-subjects-container-RV', 'children', allow_duplicate=True),
    Output('subjects-RV', 'data', allow_duplicate=True),
    Input({'type': 'show-all-button-RV', 'index': ALL}, 'n_clicks'),
    State('project-selection-dropdown-FitBit-RV', 'value'),
    State('subjects-RV', 'data'),
    State({'type': 'cosinor-files-dropdown-RV', 'index': ALL}, 'value'),
    State('selected-cosinor-file-name', 'data'),
    prevent_initial_call=True
)
def show_all(n_clicks, project, subjects, file, params_to_store):
    if n_clicks == 0:
        raise PreventUpdate
    if n_clicks[0] == 0:
        raise PreventUpdate
    path = Pconfigs[project]

    file = file[0]
    # Get the subjects
    estimated_curve_file_name = (str(file).replace('cosinor_', 'estimates_')).replace('.csv', '.parquet')
    estimated_curve_file_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', estimated_curve_file_name)

    visu_file_name = (str(file).replace('cosinor_', 'visu_')).replace('.csv', '.parquet')
    visu_file_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', visu_file_name)

    desc_json_file_name = (str(file).replace('cosinor_', 'json_')).replace('.csv', '.json')
    desc_json_file_path = Path(path).joinpath('Outputs').joinpath('Aggregated Output', desc_json_file_name)

    if not estimated_curve_file_path.exists():
        return dbc.Alert('The file estimated curve file does not exist. Please generate it first in the rhythmic page.', color='danger'), []
    
    if not visu_file_path.exists():
        return dbc.Alert('The file visu file does not exist. Please generate it first in the rhythmic page.', color='danger'), []
    
    if not desc_json_file_path.exists():
        return dbc.Alert('The file description json file does not exist. Please generate it first in the rhythmic page.', color='danger'), []
    
    estimated_df = pl.read_parquet(estimated_curve_file_path)
    visu_df = pl.read_parquet(visu_file_path)
    desc_json = json.load(open(desc_json_file_path, 'r'))

    if estimated_df.is_empty() or visu_df.is_empty():
        return dbc.Alert('One of the files is empty.', color='danger'), []
    
    subs_desc_dict = {subject: desc_json[subject] for subject in subjects}

    cards = make_cards(estimated_df, visu_df, subs_desc_dict, subjects, params_to_store)

    return cards, subjects
# ...existing code...





