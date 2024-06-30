import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import zipfile
import streamlit as st
import plotly.express as px

# Cargar datos
@st.cache_data
def load_data():
    with zipfile.ZipFile('df_clean_final.zip', 'r') as zipf:
        with zipf.open('df_clean_final - copia.csv') as f:
            df = pd.read_csv(f)
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df[["formatted_experience_level", "group_industry", "category", "state_formatted"]] = df[["formatted_experience_level", "group_industry", "category", "state_formatted"]].astype("string")
    return df

df = load_data()

# Preprocesar datos
@st.cache_data
def preprocess_data(df):
    df_dummies = pd.get_dummies(df, columns=["formatted_experience_level", "group_industry", "category", "state_formatted"])
    x = df_dummies.drop(columns=['medium_salary', 'job_id', 'company_id', 'state', 'city', 'industry_id', 'job_title', 'description', 'location'])
    y = df_dummies['medium_salary']
    return x, y

x, y = preprocess_data(df)

# Entrenar modelo
@st.cache_resource
def train_model(x_train, y_train):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.09,
        'max_depth': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'seed': 42
    }
    num_rounds = 770
    model = xgb.train(params, dtrain, num_rounds)
    return model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = train_model(x_train, y_train)

dtest = xgb.DMatrix(x_test)
y_pred = model.predict(dtest)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.write(f"RMSE: {rmse}")
st.write(f"R-squared score: {r2:.4f}")

# Función para predecir salarios por estado
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

st.title("Salary Prediction Dashboard")

experience_level = st.selectbox("Experience Level:", df['formatted_experience_level'].unique())
industry = st.selectbox("Industry Group:", df['group_industry'].unique())
category = st.selectbox("Category:", df['category'].unique())
state = st.selectbox("State:", df['state_formatted'].unique())
min_salary = st.number_input("Minimum Salary:", value=0)
max_salary = st.number_input("Maximum Salary:", value=0)

if st.button('Predict'):
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
    
    st.write(f"Predicted Salary for {state}: ${prediction:,.2f}")
    st.plotly_chart(fig)
