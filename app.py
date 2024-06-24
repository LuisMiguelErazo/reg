import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px

x = df_dummies.drop(columns=['medium_salary','job_id', 'company_id',  
                'state', 'city', 'industry_id', 'job_title', 'description',
                'location', ])#'employee_count','minimum_salary','maximum_salary',  # Features (dummy variables) #IMPORTANT: INCLUDING 
# "EMPLOYEE_COUNT","MINIMUM_SALARY" AND "MAXIMUM_SALARY"
y = df_dummies['medium_salary']                 # Target variable

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format (optimized for XGBoost)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  # Regression task with squared error loss
    'eval_metric': 'rmse',            # Evaluation metric: Root Mean Squared Error
    'eta': 0.09,                      # Learning rate
    'max_depth': 4,                   # Maximum depth of each tree
    'subsample': 0.9,                 # Subsample ratio of the training instance
    'colsample_bytree': 0.9,          # Subsample ratio of columns when constructing each tree
    'seed': 42                        # Random seed for reproducibility
}

# Training
num_rounds = 770  # Number of boosting rounds
model = xgb.train(params, dtrain, num_rounds)

# Predict on test data
y_pred = model.predict(dtest)

# Calculate RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2:.4f}")

# Function to predict salary for each state
def predict_state_salaries(experience_level, industry, category, min_salary, max_salary):
    predictions = []
    for state in df['state_formatted'].unique():
        employee_count = df.loc[df['state_formatted'] == state, 'employee_count'].values[0]
        input_data = pd.DataFrame({
            'formatted_experience_level': [experience_level],
            'group_industry': [industry],
            'category': [category],
            'state_formatted': [state],
            'minimum_salary': [min_salary],
            'maximum_salary': [max_salary],
            'employee_count': [employee_count]
        })
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=x.columns, fill_value=0)
        dinput = xgb.DMatrix(input_data)
        prediction = model.predict(dinput)
        predictions.append((state, prediction[0]))
    return predictions

# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Salary Prediction Dashboard"),
    
    html.Label("Experience Level:"),
    dcc.Dropdown(
        id='experience-dropdown',
        options=[{'label': level, 'value': level} for level in df['formatted_experience_level'].unique()],
        value=df['formatted_experience_level'].iloc[0]
    ),
    
    html.Label("Industry Group:"),
    dcc.Dropdown(
        id='industry-dropdown',
        options=[{'label': industry, 'value': industry} for industry in df['group_industry'].unique()],
        value=df['group_industry'].iloc[0]
    ),
    
    html.Label("Category:"),
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': category, 'value': category} for category in df['category'].unique()],
        value=df['category'].iloc[0]
    ),
    
    html.Label("State:"),
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': state, 'value': state} for state in df['state_formatted'].unique()],
        value=df['state_formatted'].iloc[0]
    ),
    
    html.Label("Minimum Salary:"),
    dcc.Input(id='min-salary-input', type='number', value=0),
    
    html.Label("Maximum Salary:"),
    dcc.Input(id='max-salary-input', type='number', value=0),
    
    html.Button('Predict', id='predict-button', n_clicks=0),
    
    html.Div(id='prediction-output'),
    
    dcc.Graph(id='map-graph')
])

# Define callback to update prediction output
@app.callback(
    Output('prediction-output', 'children'),
    Output('map-graph', 'figure'),
    Input('predict-button', 'n_clicks'),
    State('experience-dropdown', 'value'),
    State('industry-dropdown', 'value'),
    State('category-dropdown', 'value'),
    State('state-dropdown', 'value'),
    State('min-salary-input', 'value'),
    State('max-salary-input', 'value')
)
def update_output(n_clicks, experience_level, industry, category, state, min_salary, max_salary):
    if n_clicks > 0:
        employee_count = df.loc[df['state_formatted'] == state, 'employee_count'].values[0]
        
        input_data = pd.DataFrame({
            'formatted_experience_level': [experience_level],
            'group_industry': [industry],
            'category': [category],
            'state_formatted': [state],
            'minimum_salary': [min_salary],
            'maximum_salary': [max_salary],
            'employee_count': [employee_count]
        })
        
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=x.columns, fill_value=0)
        dinput = xgb.DMatrix(input_data)
        prediction = model.predict(dinput)[0]
        
        state_predictions = predict_state_salaries(experience_level, industry, category, min_salary, max_salary)
        prediction_df = pd.DataFrame(state_predictions, columns=['state', 'predicted_salary'])
        
        fig = px.choropleth(
            prediction_df,
            locations='state',
            locationmode="USA-states",
            color='predicted_salary',
            scope="usa",
            color_continuous_scale="Viridis",
            labels={'predicted_salary': 'Predicted Salary'}
        )
        
        return f"Predicted Salary for {state}: ${prediction:,.2f}", fig
    else:
        return '', {}

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
