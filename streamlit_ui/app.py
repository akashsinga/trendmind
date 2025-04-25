import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from core.trainer.trainer import run_daily_training
from core.trainer.weekly_trainer import run_weekly_training
from core.predictor.predictor import run_daily_prediction
from core.predictor.weekly_predictor import run_weekly_prediction
from core.backtest.backtest import run_daily_backtest
from core.backtest.weekly_backtest import run_weekly_backtest

st.set_page_config(page_title="Stock Prediction Control Panel", layout="wide", initial_sidebar_state="expanded")

# Header
st.markdown("""
    <div style='background-color:#0066cc;padding:20px;border-radius:10px'>
    <h1 style='color:white;text-align:center;'>📈 Stock Prediction Control Panel</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/stocks-growth.png", width=80)
st.sidebar.title("Pipeline Actions")
pipeline_option = st.sidebar.radio(
    "Select Action",
    ["🏠 Home", "📤 Upload Bhavcopy", "⚙️ Train Model", 
     "🔮 Run Predictions", "🧪 Run Backtest", "📅 View Predictions", "📊 Backtest Analysis"]
)

# Home Screen
if pipeline_option == "🏠 Home":
    st.subheader("🏠 Welcome to the Stock Prediction Control Panel!")
    st.markdown("""
    Manage your stock prediction workflows easily:
    - **Upload Data:** Manage bhavcopy uploads.
    - **Train Models:** Daily or Weekly model training.
    - **Generate Predictions:** Quickly run predictions.
    - **Backtest and Analysis:** Evaluate and visualize model accuracy.
    """)

# Upload Bhavcopy Section
elif pipeline_option == "📤 Upload Bhavcopy":
    st.subheader("📥 Upload Bhavcopy Data")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Choose CSV Bhavcopy file", type="csv")
    with col2:
        upload_type = st.selectbox("Upload Type", ["Daily", "Weekly"])
    
    if uploaded_file:
        file_path = os.path.join("data", "bhavcopies", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {upload_type} bhavcopy '{uploaded_file.name}' uploaded successfully.")

# Model Training Section
elif pipeline_option == "⚙️ Train Model":
    st.subheader("🛠️ Model Training")
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model Type", ["Daily", "Weekly"])
    with col2:
        train_button = st.button("🚀 Start Training", use_container_width=True)
    
    if train_button:
        with st.spinner(f"Training {model_type.lower()} model..."):
            if model_type == "Daily":
                run_daily_training()
            else:
                run_weekly_training()
        st.success(f"🎉 {model_type} model trained successfully!")

# Predictions Section
elif pipeline_option == "🔮 Run Predictions":
    st.subheader("🔭 Generate Predictions")
    col1, col2 = st.columns(2)
    with col1:
        prediction_type = st.selectbox("Prediction Type", ["Daily", "Weekly"])
    with col2:
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7, step=0.05)
    
    predict_button = st.button("🔍 Run Predictions", use_container_width=True)
    
    if predict_button:
        with st.spinner(f"Generating {prediction_type.lower()} predictions..."):
            if prediction_type == "Daily":
                run_daily_prediction(confidence_threshold)
            else:
                run_weekly_prediction(confidence_threshold)
        st.success(f"✅ {prediction_type} predictions generated successfully!")

# Backtesting Section
elif pipeline_option == "🧪 Run Backtest":
    st.subheader("📊 Model Backtesting")
    col1, col2 = st.columns(2)
    with col1:
        backtest_type = st.selectbox("Backtest Type", ["Daily", "Weekly"])
    with col2:
        backtest_button = st.button("📐 Start Backtest", use_container_width=True)
    
    if backtest_button:
        with st.spinner(f"Running {backtest_type.lower()} backtest..."):
            if backtest_type == "Daily":
                run_daily_backtest()
            else:
                run_weekly_backtest()
        st.success(f"🏅 {backtest_type} backtest completed successfully!")

# View Predictions Section
elif pipeline_option == "📅 View Predictions":
    st.subheader("📅 View Historical Predictions")

    prediction_type = st.selectbox("Select Prediction Type", ["Daily", "Weekly"])
    prediction_dir = os.path.join("outputs", prediction_type.lower())

    if not os.path.exists(prediction_dir):
        st.warning("No predictions available. Please generate predictions first.")
    else:
        available_files = sorted([
            f for f in os.listdir(prediction_dir) if f.endswith(".csv")
        ], reverse=True)

        if available_files:
            selected_file = st.selectbox("Select Prediction File", available_files)

            file_path = os.path.join(prediction_dir, selected_file)
            df_pred = pd.read_csv(file_path)
            st.dataframe(df_pred, use_container_width=True)

            st.markdown("### 📈 Confidence Distribution")
            fig_conf = px.histogram(df_pred, x='confidence', nbins=20, title='Confidence Level Distribution')
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.warning("No prediction files found.")

# Backtest Analysis Section
elif pipeline_option == "📊 Backtest Analysis":
    st.subheader("📉 Backtest Analysis")

    backtest_type = st.selectbox("Select Backtest Type", ["Daily", "Weekly"])
    backtest_dir = os.path.join("outputs", backtest_type.lower())

    if not os.path.exists(backtest_dir):
        st.warning("No backtest results found. Please run backtests first.")
    else:
        available_files = sorted([
            f for f in os.listdir(backtest_dir) if "backtest_results" in f and f.endswith(".csv")
        ], reverse=True)

        if available_files:
            selected_file = st.selectbox("Select Backtest Result", available_files)
            file_path = os.path.join(backtest_dir, selected_file)
            df_backtest = pd.read_csv(file_path)
            st.dataframe(df_backtest, use_container_width=True)

            accuracy = df_backtest['correct'].mean() * 100
            st.metric("🏆 Overall Accuracy", f"{accuracy:.2f}%")

            st.markdown("### 📌 Confidence vs. Accuracy")
            df_backtest['confidence_bucket'] = pd.cut(df_backtest['confidence'], bins=[0.5, 0.7, 0.9, 1.0], labels=["0.5-0.7", "0.7-0.9", "0.9-1.0"])
            conf_accuracy = df_backtest.groupby('confidence_bucket')['correct'].mean().reset_index()
            conf_accuracy['correct'] *= 100

            fig_conf_acc = px.bar(conf_accuracy, x='confidence_bucket', y='correct',
                labels={"confidence_bucket": "Confidence Level", "correct": "Accuracy (%)"},
                title="Accuracy by Confidence Level")
            st.plotly_chart(fig_conf_acc, use_container_width=True)

            st.markdown("### 🚀 Percent Move Distribution")
            fig_move = px.histogram(df_backtest, x='percent_move', nbins=30, title='Distribution of Percent Moves')
            st.plotly_chart(fig_move, use_container_width=True)
        else:
            st.warning("No backtest files found.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: grey;'>
    Stock Prediction Control Panel © 2025
</div>
""", unsafe_allow_html=True)
