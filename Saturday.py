from dash import dcc, html
from dash.dependencies import Input, Output
import dash
import os
from flask import Flask, send_from_directory
import pandas as pd
from joblib import load
import plotly.express as px
from prophet import Prophet
from datetime import datetime, timedelta

# Load the saved Prophet model
# Create a Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load the saved Prophet model using joblib
model_filename = 'E:\Excelr-P269-Internship-Project-Group2-OIl-Price-Prediction-And-Forecasting-Using-Python\models\prophet_model.joblib'
model = load(model_filename)

external_css = ['/assets/background1.css', '/assets/styles.css']  # Add path to styles.css

# Wrap the navigation bar inside a container
# Create the navigation bar layout
nav_bar = html.Div([
    dcc.Link('Oil_Price_Prediction', id='page-1-link', href='/page-1', className='nav-link'),
    dcc.Link('Oil_Price_Prediction_Yearly', id='page-2-link', href='/page-2', className='nav-link'),
    dcc.Link('Graphs and Charts', id='page-3-link', href='/page-3', className='nav-link'),
    dcc.Link('Presentations', id='page-4-link', href='/page-4', className='nav-link'),  # Add this line
], className='nav-bar-container')

# Add callback to dynamically set background color based on active page
@app.callback(
    Output('page-1-link', 'style'),
    Output('page-2-link', 'style'),
    Output('page-3-link', 'style'),
    Output('page-4-link', 'style'),  # Add this line
    Input('url', 'pathname')
)
def set_active_page_style(pathname):
    page_styles = [{'margin-right': '10px', 'background-color': '#a6e7ff'}] * 4  # Change to 4
    
    if pathname == '/page-1':
        page_styles[0]['background-color'] = '#000080'  # Set color for active page
    elif pathname == '/page-2':
        page_styles[1]['background-color'] = '#9932cc'  # Set color for active page
    elif pathname == '/page-3':
        page_styles[2]['background-color'] = '#7b68ee'  # Set color for active page
    elif pathname == '/page-4':  # Add this condition
        page_styles[3]['background-color'] = '#4169e1'  # Set color for active page
        
    return page_styles


# Add the navigation bar container to the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    nav_bar,  # Include the navigation bar container
    html.Div(id='page-content')
] + [html.Link(rel='stylesheet', href=css) for css in external_css])


# Page 1 Layout
page_1_layout = html.Div(
    style={
        'background-image': 'url("/assets/Page_1bg.jpg")',  # Replace with your image path
        'background-size': 'cover',  # Scale the image to cover the entire container
        'background-repeat': 'no-repeat',  # Prevent the image from repeating
        'background-position': 'center',  # Center the background image
        'display': 'flex',
        'flex-direction': 'column',  # Set flex-direction to column
        'justify-content': 'center',  # Center vertically
        'align-items': 'center',  # Center horizontally
        'height': '100vh',
    },
    children=[
        html.Div(
            style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'padding': '20px',  # Add dynamic padding here
                'background-color': 'rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
                'border-radius': '8px',
                'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'  # Shadow
            },
            children=[
                dcc.DatePickerSingle(
                    id='date-picker',
                    date=pd.to_datetime('2023-08-21').date(),
                    style={
                        'font-size': '16px',
                        'padding': '8px',
                        'margin-bottom': '10px',
                        'background-color': '#ffd700',  # Background color
                        'border': 'none',
                        'border-radius': '8px',  # Rounded corners
                        'color': '#333',  # Text color
                        'cursor': 'pointer'
                    },
                ),
                html.Button(
                    'Predict Price',
                    id='predict-button',
                    n_clicks=0,
                    style={'background-color': '#007BFF', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'}
                ),
                html.Div(id='output-div', style={'margin-top': '20px', 'font-size': '18px', 'color': 'black'})  # Set the color here
            ]
        ),
        dcc.Link(
            'Go to Oil_Price_Prediction_Yearly',
            href='/page-2',
            style={
                'color': 'blue',
                'text-decoration': 'none',
                'align-self': 'center',  # Align link to the center
                'border': '2px solid #007BFF',
                'border-radius': '8px',
                'padding': '10px',
                'background-color': '#007BFF',
                'color': 'white',
                'cursor': 'pointer',
                'transition': 'background-color 0.3s'
            }
        ),
        dcc.Link(
            'Go to Graphs and Visualizations',  # Link text
            href='/page-3',  # Link to Page 3
            style={
                'color': 'blue',
                'text-decoration': 'none',
                'align-self': 'center',  # Align link to the center
                'border': '2px solid #007BFF',
                'border-radius': '8px',
                'padding': '10px',
                'background-color': '#007BFF',
                'color': 'white',
                'cursor': 'pointer',
                'transition': 'background-color 0.3s',
                'margin-top': '20px'  # Add margin to create space between the link and other elements
            }
        ),
    ]
)


# Predict price callback function
@app.callback(
    Output('output-div', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('date-picker', 'date')]
)
def predict_price(n_clicks, selected_date):
    if n_clicks > 0:
        selected_date = pd.to_datetime(selected_date)
        future = pd.DataFrame({'ds': [selected_date]})
        forecast = model.predict(future)
        predicted_price = forecast.loc[0, 'yhat']
        
        return [
            html.Span("The Predicted Price On The ",style={'color' : 'black'}),
            html.Span(selected_date, style={'color': 'red'}),  # Add color to the date
            html.Span(f" is {predicted_price:.2f}$", style={'color': 'blue'})  # Add color to the price
        ]
    return ""
# Layout for Page 2
page_2_layout = html.Div([
    html.H2('Select Forecast Duration', style={'text-align': 'center', 'margin-bottom': '20px'}),
    dcc.Dropdown(
        id='forecast-duration-dropdown',
        options=[
            {'label': '1 Year', 'value': 1},
            {'label': '2 Years', 'value': 2},
            {'label': '3 Years', 'value': 3},
            {'label': '5 Years', 'value': 5},
            {'label': '10 Years', 'value': 10}
        ],
        value=1,
        style={'width': '100%', 'margin-bottom': '20px'}
    ),
    dcc.Graph(id='forecast-graph', style={'height': '400px'}),  # Placeholder for the forecasting graph
    dcc.Link(
        'Go to Oil_Price_Prediction',  # Link text
        href='/',  # Link to Page 1
        style={
            'color': 'blue',
            'text-decoration': 'none',
            'align-self': 'center',  # Align link to the center
            'border': '2px solid #007BFF',
            'border-radius': '8px',
            'padding': '10px',
            'background-color': '#007BFF',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 1.0s',
            'margin-top': '20px'  # Add margin to create space between the graph and the link
        }
    ),
    dcc.Link(
        'Go to Graphs and Visualizations',  # Link text
        href='/page-3',  # Link to Page 3
        style={
            'color': 'blue',
            'text-decoration': 'none',
            'align-self': 'center',  # Align link to the center
            'border': '2px solid #007BFF',
            'border-radius': '8px',
            'padding': '10px',
            'background-color': '#007BFF',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 1.0s',
            'margin-top': '20px'  # Add margin to create space between the link and other elements
        }
    ),
], style={
    'max-width': '800px',
    'margin': '0 auto',
    'padding': '20px',
    'border': '1px solid #ddd',
    'border-radius': '10px',
    'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
    'animation': 'changeColor 10s infinite alternate'  # CSS animation
}, className='dynamic-background')

# Callback to update the forecasting graph based on user selection
@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-duration-dropdown', 'value')]
)
def update_forecast_graph(selected_duration):
    # Calculate the start date (2023) for the forecast
    start_date = pd.Timestamp('2023-01-01')
    
    # Calculate the end date based on the selected_duration
    end_date = start_date + pd.DateOffset(days=365 * selected_duration)
    
    # Create a future DataFrame with the desired date range
    future = pd.DataFrame({'ds': pd.date_range(start_date, end_date)})
    
    # Perform forecasting using the saved Prophet model
    forecast = model.predict(future)
    
    # Rename columns for user-friendly names
    forecast.rename(columns={'ds': 'Date', 'yhat': 'Price'}, inplace=True)
    
    # Create the forecasting graph using Plotly Express
    fig = px.line(forecast, x='Date', y='Price', title=f'Forecasted Prices for {selected_duration} Years')
    return fig

# Layout for Page 3
page_3_layout = html.Div([
    html.H2('Graphs and Visualizations -Project Timeline', style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Img(src='/assets/image1.jpg', style={'width': '100%'}),
            html.P(" This graph compares two data sets, Data 1 (raw data) and Data 2 (cleaned data), over time from 2005 to 2025. The x-axis represents the date and the y-axis represents the price. The highest point on the graph is 145.2899932861328 for Data 1 and the lowest point is 37.63000016815234 for Data 2")
        ], className='image-container'),
        html.Div([
            html.Img(src='/assets/image2.jpg', style={'width': '100%'}),
            html.P("We can see that there are outliers in the raw data and no outliers in the cleaned data , this cleaning process was done by recoding the outliers above the upper fence with the upper fence value and below lower fence with lower fence values using iqr capping techinique")
        ], className='image-container'),
        html.Div([
        html.Img(src='/assets/image4.png', style={'width': '100%'}),
        html.P("This is the python code for removing outliers using IQR capping method")
        ], className='image-container'),
        html.Div([
        html.Img(src='/assets/image5.png', style={'width': '100%'}),
        html.P("We can see that there are no outliers in our data for now as shown in the above box plot")
        ], className='image-container'),
        html.Div([
        html.Img(src='/assets/image6.png', style={'width': '100%'}),
        html.P("We had our data non-stationary.so, we made it stationary in order to apply the forecasting models")
        ], className='image-container'),
        html.Div([
        html.Img(src='/assets/image3.jpg', style={'width': '100%'}),
        html.P("Cause of lower rmse score we went with prophet model for our prediction")
        ], className='image-container'),
    ], className='image-container-wrapper'),
    dcc.Link(
        'Go to Oil_Price_Prediction',  # Link text
        href='/',  # Link to Page 1
        style={
            'color': 'blue',
            'text-decoration': 'none',
            'align-self': 'center',  # Align link to the center
            'border': '2px solid #007BFF',
            'border-radius': '8px',
            'padding': '10px',
            'background-color': '#007BFF',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 1.0s',
            'margin-top': '20px'  # Add margin to create space between the containers and the link
        }
    ),
], style={
    'max-width': '800px',
    'margin': '0 auto',
    'padding': '20px',
    'border': '1px solid #ddd',
    'border-radius': '10px',
    'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
    'animation': 'changeColor 10s infinite alternate'  # CSS animation
}, className='dynamic-background')


# Define a route to serve PowerPoint presentations
@app.server.route('/presentation/<path:filename>')
def serve_presentation(filename):
    presentation_path = os.path.join('presentation', filename)
    return send_from_directory('presentation', filename, mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation')

# Layout for Page 4
page_4_layout = html.Div([
    html.H2('Presentations', style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.H3('Video Presentation'),
        html.Video(src='/presentation/P269-video.mp4', controls=True, style={'width': '100%'}),
    ], className='presentation-container'),

    html.Div([
        html.H3('PowerPoint Presentation'),
        html.Iframe(src='/presentation/P269-group2-Presentation.pptx', style={'width': '0%', 'height': '0px'}),
    ], className='presentation-container'),

    dcc.Link(
        'Go to Oil_Price_Prediction',  # Link text
        href='/',  # Link to Page 1
        style={
            'color': 'blue',
            'text-decoration': 'none',
            'align-self': 'center',  # Align link to the center
            'border': '2px solid #007BFF',
            'border-radius': '8px',
            'padding': '10px',
            'background-color': '#007BFF',
            'color': 'white',
            'cursor': 'pointer',
            'transition': 'background-color 1.0s',
            'margin-top': '20px'  # Add margin to create space between the containers and the link
        }
    ),
], style={
    'max-width': '800px',
    'margin': '0 auto',
    'padding': '20px',
    'border': '1px solid #ddd',
    'border-radius': '10px',
    'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
    'animation': 'changeColor 10s infinite alternate'
}, className='dynamic-background')


# Updating the page content based on URL
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    elif pathname == '/page-4':
        return page_4_layout 
    else:
        return page_1_layout


if __name__ == '__main__':
    app.run_server(debug=True)
