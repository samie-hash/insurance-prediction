import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

st.write("""
# Insurance prediction """)
st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/samie-hash/insurance-prediction/blob/main/Data/train.csv)
""")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Gender = st.sidebar.selectbox('Gender',('Male', 'Female'))
        Age = st.sidebar.slider('Age',18,100,1)
        Driving_License = st.sidebar.selectbox('Driving_License',(' 0 : Customer does not have DL',' 1 : Customer already has DL'))
        Region_Code = st.sidebar.slider('Region_Code',1,100,1)
        Previously_Insured = st.sidebar.selectbox('Previously_Insured',(' 1 : Customer already has Vehicle Insurance'," 0 : Customer doesn't have Vehicle Insurance"))
        Vehicle_Age = st.sidebar.slider('Vehicle_Age',1,10,1)
        Vehicle_Damage = st.sidebar.selectbox('Vehicle_Damage',('Yes', 'No'))
        Annual_Premium = st.sidebar.slider('Annual_Premium',10000,1000000,1000)
        Policy_Sales_Channel = st.sidebar.slider('Policy_Sales_Channel',1,1000,1)
        Vintage= st.sidebar.slider('Vintage',1,1000,1)

        data = {
            'Gender': Gender,
            'Age': Age,
            'Driving_License': Driving_License,
            'Region_Code': Region_Code,
            'Previously_Insured' : Previously_Insured,
            'Vehicle_Age' : Vehicle_Age,
            'Vehicle_Damage' : Vehicle_Damage,
            'Annual_Premium': Annual_Premium,
            'Policy_Sales_Channel': Policy_Sales_Channel,
            'Vintage': Vintage
        }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

train_raw = pd.read_csv('Data/train.csv')
insurance = train_raw.drop(columns=['Response','id'], axis=1)

df = pd.concat([input_df,insurance],axis=0)

encode = ['Gender', 'Vehicle_Damage','Vehicle_Age']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]

st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

load_clf = pickle.load(open('models/gradient_boost.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
Response = np.array(['Yes','No'])
st.write(Response[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

        