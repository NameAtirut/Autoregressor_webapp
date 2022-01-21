from calendar import c
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

st.set_page_config(page_title="ML for Regression GUI",layout='wide')

st.sidebar.title("Upload your file")
uploaded= st.sidebar.file_uploader("Upload your file here", type=['csv','tsv','txt'],help='Support csv, tsv, and txt files')
if uploaded:
    st.sidebar.write('Using the uploaded file as input')



def train_models(x,y, task='Regression'):
    split_size = 0.2
    seed_number = 42
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = split_size,random_state = seed_number)
    if task == 'Regression':
        model = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    elif task == 'Classification':
        model = LazyClassifier(verbose=0,ignore_warnings=False, custom_metric=None)
    else:
        raise ValueError('Invalid task')
    models, predictions= model.fit(x_train,x_test,y_train,y_test)
    return models, predictions

target_col=None
result_models=None

with st.spinner(' << Upload your data in the sidebar'):
    example = st.button('DEMO dataset & training')

if example:
    st.header("Exemplifying diabetes dataset")
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    Y = pd.Series(diabetes.target, name='response')
    
    col1,col2=st.columns(2)
    with col1: st.write(X.head(5))
    with col2: st.write(Y.head(5))

    with st.spinner("Models training..."):
        with st.empty():
            st.write('Just a second!!')
            result_models,result_predictions = train_models(X, Y)
            message = st.info('Training finished')
    st.balloons()
    st.write("Training Performance",result_models)
    st.write("Test Performance",result_predictions)
    
    five_best = result_models.head(5)
    st.write(five_best)

elif uploaded:
    
    uploaded_file = pd.read_csv(uploaded,header=0)
    df = pd.DataFrame(uploaded_file)
    st.header("Your Dataset")
    st.write(df)
    
    task = st.radio("What do you want to do?",("Regression","Classification"))

    target_col = st.radio("Which one is the target label?", options=[name for name in df.columns])
    drop_cols = st.multiselect("Any columns you want to drop?", options=[name for name in df.columns])
    if target_col:
        Y = df[target_col]
        X = df.drop([target_col],axis=1)
        
        if drop_cols:
            X = X.drop(drop_cols,axis=1)

        for col in list(X.columns):
            if X[col].dtype == str:
                X[col] = LabelEncoder().fit_transform(X[col])

        if type(Y.values[0]) == str:
            Y = pd.DataFrame(LabelEncoder().fit_transform(Y))
        
        
        col1,col2=st.columns(2)
        with col1: st.write(X.head(5))
        with col2: st.write(Y.head(5))
        start_training=st.button("Start training")


            
        if start_training:
            with st.spinner("Models training..."):
                with st.empty():
                    st.write('Just a second!!')
                    result_models,result_predictions = train_models(X, Y, task = task)
                    message = st.info('Training finished')
            st.balloons()
            st.write("Model Performance",result_models)
            result_models.to_csv('models.csv')
        
# if result_models is not None:
#     models = pd.read_csv('models.csv',header=0)
#     models_task = [m for m in models['Model'].iloc[0:5]]
#     selected_model = st.radio("Select a model", options=models_task)
#     selected_model=str(selected_model)

#     from sklearn.ensemble import selected_model
    