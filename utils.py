import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer


def feature_gen(df):
    df['Call Failure Rate'] = df['Call Failure'] / df['Frequency of use']
    df['Usage Intensity'] = df['Seconds of Use'] / df['Frequency of use']
    df['SMS Rate'] = df['Frequency of SMS'] / df['Frequency of use']
    df['Communication Density'] = df['Frequency of use'] / df['Seconds of Use']
    df['Usage Pattern Stability'] = df['Frequency of use'].std() / df['Subscription Length']
    df['Subscription Length Stability'] = df['Subscription Length'].std() / df['Frequency of use']
    df['Communication Intensity'] = (df['Frequency of use'] + df['Frequency of SMS']) / df['Seconds of Use']
    df['Average Subscription Length for Callers'] = df.groupby('Distinct Called Numbers')['Subscription Length']\
        .transform('mean')
        
    df.fillna(0, inplace=True)
    
    return df


def preprocess(dataframe):
    df = dataframe.copy()
    
    df.loc[df['Call Failure'].isnull(),'Call Failure'] = feature_predict(df, 'Call Failure', 'Charge Amount', degree=2)
    df.loc[df['Seconds of Use'].isnull(), 'Seconds of Use'] = seconds_of_use_predict(df)
    df.loc[df['Frequency of use'].isnull(),'Frequency of use'] = feature_predict(df, 'Frequency of use', 'Seconds of Use')
    df.loc[df['Distinct Called Numbers'].isnull(),'Distinct Called Numbers'] = feature_predict(df, 'Distinct Called Numbers', 'Frequency of use')

    
    df['Complains'].fillna(0, inplace=True)
    df['Subscription Length'].fillna(df['Subscription Length'].mean(), inplace=True)
    df['Frequency of SMS'].fillna(df['Frequency of SMS'].mean(), inplace=True)
    
    
    df['Call Failure'] = df['Call Failure'].astype(int)
    df['Complains'] = df['Complains'].astype(int)
    df['Subscription Length'] = df['Subscription Length'].astype(int)
    df['Charge Amount'] = df['Charge Amount'].astype(int)
    df['Seconds of Use'] = df['Seconds of Use'].astype(int)
    df['Frequency of use'] = df['Frequency of use'].astype(int)
    df['Frequency of SMS'] = df['Frequency of SMS'].astype(int)
    df['Distinct Called Numbers'] = df['Distinct Called Numbers'].astype(int)
    df['Churn'] = df['Churn'].astype(int)
    
    return df

    
def seconds_of_use_predict(df):
    data_for_imputation = df[['Seconds of Use']]

    knn_imputer = KNNImputer(n_neighbors=5)  

    imputed_data = knn_imputer.fit_transform(data_for_imputation)

    imputed_df = pd.DataFrame(imputed_data, columns=['Seconds of Use'])

    return imputed_df

def feature_predict(data, target_feature, input_feature, degree=1):
    df = data.copy()

    df_missing = df[df[target_feature].isnull()]
    df_no_missing = df.dropna(subset=[target_feature])

    X = df_no_missing[[input_feature]]
    y = df_no_missing[target_feature]

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_missing = df_missing[[input_feature]]
    X_missing_poly = poly.transform(X_missing)
    predicted_values = model.predict(X_missing_poly)

    return predicted_values