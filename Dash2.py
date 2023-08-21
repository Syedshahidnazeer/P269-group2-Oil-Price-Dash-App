from dash import dcc, html
from dash.dependencies import Input, Output
import dash
import pandas as pd
import pickle
import plotly.express as px
from prophet import Prophet
from datetime import datetime, timedelta

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load the saved Prophet model
with open('prophet_model_11k.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

external_css = ['/assets/background1.css']  # Adjust the path to your CSS file

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
] + [html.Link(rel='stylesheet', href=css) for css in external_css])

# Define the CSS classes for styling
css_classes = {
    'page_1_container': 'page-1-container',
    'predict_button_container': 'predict-button-container',
    'go_to_page_2': 'go-to-page-2'
}

# Page 1 Layout
page_1_layout = html.Div(
    className=css_classes['page_1_container'],
    children=[
        html.Div(
            className=css_classes['predict_button_container'],
            children=[
                dcc.DatePickerSingle(
                    id='date-picker',
                    date=pd.to_datetime('2023-08-10').date(),
                    style={
                        'font-size': '16px',
                        'padding': '8px',
                        'margin-bottom': '10px',
                        'background-color': '#ffd700',  # Background color
                        'border': 'none',
                        'border-radius': '8px',  # Rounded corners
                        'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',  # Shadow
                        'color': '#333',  # Text color
                        'cursor': 'pointer'
                    },
                ),
                html.Button(
                    'Predict Price',
                    id='predict-button',
                    n_clicks=0,
                    style={'background-color': '#007BFF', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'cursor': 'pointer'}
                )
            ]
        ),
        html.Div(id='output-div', style={'margin-top': '20px', 'font-size': '18px', 'color': 'black'}),
        dcc.Link('Go to Page 2', href='/page-2', className=css_classes['go_to_page_2'])
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
page_2_layout = html.Div(
    className='page-2-container',  # Apply the CSS class to the entire container
    children=[
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
        dcc.Graph(id='forecast-graph', style={'height': '400px'})  # Placeholder for the forecasting graph
    ],
    style={
        'max-width': '800px',
        'margin': '0 auto',
        'padding': '20px',
        'border': '1px solid #ddd',
        'border-radius': '10px',
        'box-shadow': '0px 0px 10px rgba(0, 0, 0, 0.1)',
        'animation': 'changeColor 10s infinite alternate'  # CSS animation
    }
)

@app.callback(
    Output('forecast-duration-dropdown', 'style'),
    [Input('forecast-duration-dropdown', 'value')]
)
def update_dropdown_color(selected_value):
    colors = ['red', 'green', 'blue', 'orange', 'purple']  # Add more colors if needed
    selected_color = colors[selected_value - 1]
    dropdown_style = {
        'width': '100%',
        'margin-bottom': '20px',
        'background-color': selected_color,
        'color': 'white'
    }
    return dropdown_style


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

# Update page content based on URL
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/page-2':
        return page_2_layout
    else:
        return page_1_layout

if __name__ == '__main__':
    app.run_server(debug=True)
