# model_pipeline.py

"""
    This module contains the preprocessing and model pipeline
"""
import sys, os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
from utils.utility import DataFrameImputer, MyLabelEncoder, DataFrameSelector

def load_data(path):
    """
    Load the data

    Parameter
    _________
    path:   str
            Location of the file

    Returns
    _______

    data: Pandas DataFrame
    """

    data = pd.read_csv(path)
    return data 


def load_model(path):
    pass


def run_pipeline(path):
    data = load_data(path)
    cat_attribs = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']
    num_attribs = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel']
    
    cat_pipeline = Pipeline(
        [   
            ('selector', DataFrameSelector(cat_attribs)),
            ('imputer', DataFrameImputer()),
            ('encoder', MyLabelEncoder(cat_attribs)),
        ]
    )

    num_pipeline = Pipeline(
        [
            ('imputer', DataFrameImputer()),
            ('selector', DataFrameSelector(num_attribs)),
            ('scaler', StandardScaler())
        ]
    )

    full_pipeline = FeatureUnion(
        transformer_list = [
            ('cat_pipeline', cat_pipeline),
            ('num_pipeline', num_pipeline),
        ]
    )

    data_prepared = full_pipeline.fit_transform(data)
    print(data_prepared)
    model = load_model('models/log_reg.pkl')
    prediction = model.predict(data_prepared)
    return 0
    

if __name__ == '__main__':
    pred = run_pipeline('Data/train.csv')
    print(pred)