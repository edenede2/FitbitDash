import dash
from dash import html, dcc
from flask import Flask
import dash_bootstrap_components as dbc
import webview
import pages.scripts.UTILS.utils as ut


flask_app = Flask(__name__)

app = dash.Dash(__name__, server=flask_app, external_stylesheets=[dbc.themes.CERULEAN], use_pages=True)

# window = webview.create_window('FitBit App', flask_app, width=800, height=600, resizable=True, fullscreen=False)

sidebar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page['name'], className='ms-2'),
            ],
            href=page['path'],
            active="exact",
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className="bg-light",
)

app.layout = dbc.Container([
    dbc.Row(
        [
        dbc.Col(
            [
            html.Img(src=ut.convert_path_to_current_os(r'\assets\logo transperant (1).png'), style={'width': '100%'})
            ], width=2),
        dbc.Col(
            [
            html.H1("FitBit App"),
            html.Hr()
            ], width=8, style={'textAlign': 'left'})
        
    ]),
    dbc.Row(
        [
            dbc.Col([
                sidebar
            ], width=2),
            dbc.Col([
                dash.page_container
            ], width=10)
        ]
    )
], fluid=True)



if __name__ == '__main__':
    # webview.start()
    app.run(host='127.0.0.1', port=8080, debug=True)
