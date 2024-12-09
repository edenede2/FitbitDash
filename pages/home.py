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





dash.register_page(__name__,  path='/')



layout = html.Div([
    html.H1('Home Page'),
    html.P('Welcome to the FitBit App!'),
    html.P('You using the FitBit App 1.0 version'),
    html.Hr(),
    html.P('This app guided by the FitBit tutorial website'),
    html.P('You can find the tutorial here:'),
    html.A('FitBit Tutorial', href='https://fitbitutorial.streamlit.app/'),
    html.Hr(),
    html.P('If you need the dashboard, you can find it here:'),
    html.A('FitBit Dashboard', href='https://fitbitestapipy.streamlit.app/'),
    html.Hr(),
    html.P('If you need the FitBit Alerts System, you can find it here:'),
    html.A('FitBit Alerts System', href='https://fitbitalartsservice.streamlit.app/'),
])