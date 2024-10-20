import joblib
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yaml
from flask import Flask

# Load configuration from YAML file
with open('data/config.YAML', 'r') as file:
    config = yaml.safe_load(file)

feature_names = config['features']
class_names = config['classes']

# Initialize Flask app
server = Flask(__name__)

# Load model and t-SNE object
model = joblib.load('models/model.joblib')
tsne = joblib.load('models/tsne_model.joblib')

# Load original data and t-SNE data
X = np.load('data/X.npy')
y = np.load('data/y.npy')
X_tsne = np.load('data/X_tsne.npy')

# Create a DataFrame for t-SNE data
df_tsne = pd.DataFrame(X_tsne, columns=['x', 'y'])
df_tsne['label'] = y

# Map numerical labels to class names
df_tsne['label'] = df_tsne['label'].map(lambda x: class_names[int(x)])

# Initialize Dash app
dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Function to create decision boundary plot
def create_decision_boundary_plot():
    # Create a mesh grid in the original feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict class for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Ensure grid_points have the same number of features as the original data
    if grid_points.shape[1] < X.shape[1]:
        grid_points = np.hstack([grid_points, np.zeros((grid_points.shape[0], X.shape[1] - grid_points.shape[1]))])
    
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Transform the grid points to t-SNE space
    grid_points_tsne = tsne.fit_transform(grid_points)

    # Create a DataFrame for the t-SNE transformed grid points
    df_grid_tsne = pd.DataFrame(grid_points_tsne, columns=['x', 'y'])
    df_grid_tsne['label'] = Z.ravel()

    # Map numerical labels to class names
    df_grid_tsne['label'] = df_grid_tsne['label'].map(lambda x: class_names[int(x)])

    # Create a plotly figure
    fig = px.scatter(df_tsne, x='x', y='y', color='label', title='t-SNE Projection', labels={'color': 'Class'})
    fig.update_traces(marker=dict(size=5))
    
    # Add decision boundary as contour lines
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=Z,
        showscale=False,
        contours=dict(
            start=0,
            end=len(class_names) - 1,
            size=1,
            coloring='lines'
        ),
        line=dict(width=2)
    ))

    return fig

# Initial t-SNE plot with decision boundaries
initial_fig = create_decision_boundary_plot()
initial_fig.update_layout(legend_title_text='Classes', legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

# Dash layout
dash_app.layout = html.Div([
    html.H1("Model Prediction Dashboard"),
    html.Div([
        html.Div([
            html.Label(f"{name}:"),
            dcc.Input(id=f'feature-{i}', type='number', required=True)
        ]) for i, name in enumerate(feature_names)
    ]),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-result'),
    dcc.Graph(id='tsne-plot', figure=initial_fig)
])

# Dash callback for prediction and t-SNE plot update
@dash_app.callback(
    [Output('prediction-result', 'children'),
     Output('tsne-plot', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [State(f'feature-{i}', 'value') for i in range(len(feature_names))]
)
def update_output(n_clicks, *features):
    if None in features:
        return "Please enter all feature values.", dash.no_update

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]

    # Combine the original data with the new data point
    X_combined = np.vstack([X, features])

    # Transform the combined data to t-SNE space
    X_tsne_combined = tsne.fit_transform(X_combined)

    # Separate the transformed data
    new_point_tsne = X_tsne_combined[-1].reshape(1, -1)
    X_tsne_original = X_tsne_combined[:-1]

    # Update the DataFrame for t-SNE data
    df_tsne_combined = pd.DataFrame(X_tsne_original, columns=['x', 'y'])
    df_tsne_combined['label'] = y

    # Map numerical labels to class names
    df_tsne_combined['label'] = df_tsne_combined['label'].map(lambda x: class_names[int(x)])

    # Update t-SNE plot
    fig = create_decision_boundary_plot()
    fig.add_scatter(x=[new_point_tsne[0][0]], y=[new_point_tsne[0][1]], mode='markers', 
                    marker=dict(color=px.colors.qualitative.Plotly[prediction], size=10), 
                    name=f'New Data (Class: {class_names[prediction]})')
    fig.update_layout(legend_title_text='Classes', legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return f"Prediction: {class_names[prediction]}", fig

# Serve the Dash app through a route in the Flask app
@server.route('/')
def render_dashboard():
    return dash_app.index()

if __name__ == '__main__':
    server.run(debug=True, host='0.0.0.0', port=8050)
