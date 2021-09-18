import streamlit as st
import pickle
import pandas as pd
import numpy as np
from FeatureEngineeringAndDataModeling import model_pipeline

st.write("""
# Insurance prediction """)
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/samie-hash/insurance-prediction/blob/main/Data/train.csv)
""")
model = model_pipeline.load_model('models/gradient_boost.pkl')
pipeline = model_pipeline.load_pipeline('models/pipeline.pkl')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

def display_prediction(pred, proba):
    st.subheader('Prediction')
    prediction_data = pd.DataFrame(columns=['Response', 'Probability'])
    response = np.array(['Not Interested','Interested'])
    prediction_data['Response'] = response[pred]
    prediction_data['Probability'] = proba
    st.write(prediction_data)

if uploaded_file is None:
    def user_input_features():
        gender = st.sidebar.selectbox('Gender',('Male', 'Female'))
        age = st.sidebar.slider('Age',18,100,18)
        driving_license = st.sidebar.selectbox('Driving_License', (0, 1))
        region_code = st.sidebar.slider('Region_Code',1,60,1)
        previously_insured = st.sidebar.selectbox('Previously_Insured',(1, 0))
        vehicle_age = st.sidebar.selectbox('Vehicle_Age',('1-2 Year', '< 1 Year', '> 2 Year'))
        vehicle_damage = st.sidebar.selectbox('Vehicle_Damage',('Yes', 'No'))
        annual_premium = st.sidebar.slider('Annual_Premium',10000, 600000,10000)
        policy_sales_channel = st.sidebar.slider('Policy_Sales_Channel',1,200,1)
        vintage= st.sidebar.slider('Vintage',1,500,1)

        data = {
            'Gender': [gender],
            'Age': [age],
            'Driving_License': [driving_license],
            'Region_Code': [region_code],
            'Previously_Insured' : [previously_insured],
            'Vehicle_Age' : [vehicle_age],
            'Vehicle_Damage' : [vehicle_damage],
            'Annual_Premium': [annual_premium],
            'Policy_Sales_Channel': [policy_sales_channel],
            'Vintage': [vintage]
        }
        features = pd.DataFrame(data)
        return features

    data = user_input_features()
    st.write(data)
    pred, proba, _ = model_pipeline.run_pipeline(data, model, pipeline)
    display_prediction(pred, proba)
else:
    data = model_pipeline.load_data(uploaded_file)
    st.subheader('User Input features')
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    pred, proba, _ = model_pipeline.run_pipeline(data, model, pipeline)
    response = np.array(['Not Interested','Interested'])
    data['Response'] = response[pred]
    data['Probability'] = proba
    st.write(data)