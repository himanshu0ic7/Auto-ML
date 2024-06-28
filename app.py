import os
import pandas as pd
import ydata_profiling
import streamlit as st
from streamlit_pandas_profiling import st_profile_report

# Initialize session state for the dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar for navigation
with st.sidebar:
    st.image("AutoML.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ['Upload', 'Profiling', 'Modelling', 'Inference', 'Download'])
    st.info("Wanna do ML but don't know much about it? No worries, just upload your dataset.")

# Handling dataset upload
if choice == 'Upload':
    st.title("Upload your Dataset")
    file = st.file_uploader("Upload CSV data")
    if file:
        st.session_state.df = pd.read_csv(file, index_col=None)
        st.dataframe(st.session_state.df)
        st.success("Dataset uploaded successfully!")

# Exploratory Data Analysis
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if st.session_state.df is not None:
        profile_df = st.session_state.df.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("Please upload a dataset first.")

# Model building with PyCaret
if choice == "Modelling":
    if st.session_state.df is not None:
        st.title("Model Building")
        target = st.selectbox('Choose the Target Column', st.session_state.df.columns)
        task = st.selectbox('Regression or Classification:', ['Regression', 'Classification'])
        
        if task == 'Regression':
            from pycaret.regression import setup, compare_models, pull, save_model
        else:
            from pycaret.classification import setup, compare_models, pull, save_model

        if st.button('Run Modelling'):
            setup(st.session_state.df, target=target, session_id=123)
            setup_df = pull()
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')
            st.success("Modeling completed successfully! You can now download your model.")
    else:
        st.warning("Please upload a dataset first.")

# Model inference
if choice == "Inference":
    st.title("Make Predictions")
    try:
        from pycaret.classification import load_model, predict_model
        model = load_model('best_model')
        
        file = st.file_uploader("Upload data for predictions")
        if file:
            predict_df = pd.read_csv(file, index_col=None)
            predictions = predict_model(model, data=predict_df)
            st.dataframe(predictions)
            st.success("Predictions made successfully!")
    except FileNotFoundError:
        st.warning("No model found. Please run the modeling step first.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Model download
if choice == "Download":
    st.title("Download Model")
    try:
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")
    except FileNotFoundError:
        st.warning("No model found. Please run the modeling step first.")

# Footer with credits
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Himanshu Kumar Saw (https://github.com/himanshu0ic7)")
