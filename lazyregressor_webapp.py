import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

st.set_page_config(page_title="ML for Regression GUI",layout='wide')

st.sidebar.title("Upload your file or specify your inputs")
uploaded= st.sidebar.file_uploader("Upload your file", type=['csv','tsv','txt'],help='Support csv, tsv, and txt files')
st.sidebar.write("==============OR================",fontsize=40)
if uploaded:
    st.sidebar.write('Using the uploaded file as input')
else:
    st.sidebar.slider("Feature_1",min_value=0,max_value=10)
    st.sidebar.slider("Feature_2",min_value=10,max_value=100,step=10)
    st.sidebar.slider("Feature_3",min_value=0.00,max_value=1.00,step=0.05)
    
    side_col1,side_col2 = st.sidebar.columns(2)

    with side_col1: st.multiselect("Select places you want to go", options = ['Osaka','Okinawa','Tokyo','Sendai','Gifu'])
    with side_col2: free_response = st.text_area("Can you tell us why?",height=50,
                                                    value="Show us how, Look at where you came from, look at you now")


def train_models(x,y):
    split_size = 0.2
    seed_number = 42
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = split_size,random_state = seed_number)
    reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
    models, predictions= reg.fit(x_train,x_test,y_train,y_test)
    # models_test,predictions_test = reg.fit(x_test,x_test,y_test,y_test)
    
    # st.write(predictions_train)
    return models, predictions

st.info(' << Upload your data in the sidebar')
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

    target_col = st.radio("Which one is the target label?", options=[name for name in df.columns])

    if target_col:
        Y = df[target_col]
        X = df.drop([target_col],axis=1)
        

        col1,col2=st.columns(2)
        with col1: st.write(X.head(5))
        with col2: st.write(Y.head(5))
        start_training=st.button("Start training")

        if start_training:
            with st.spinner("Models training..."):
                with st.empty():
                    st.write('Just a second!!')
                    result_models,result_predictions = train_models(X, Y)
                    message = st.info('Training finished')
            st.balloons()
            st.write("Training Performance",result_models)
            st.cache()