import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("📊 Backtest Analysis Dashboard")

backtest_type = st.selectbox("Select Backtest Results", ["Daily", "Weekly"])

file_path = f"outputs/{backtest_type.lower()}/{backtest_type.lower()}_backtest_results.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.dataframe(df)

    st.subheader("Accuracy Overview")
    accuracy = df['correct'].mean() * 100
    st.metric("Overall Accuracy", f"{accuracy:.2f}%")

    st.subheader("Confidence Analysis")
    fig_confidence = px.histogram(df, x='confidence', title="Confidence Distribution")
    st.plotly_chart(fig_confidence, use_container_width=True)

    st.subheader("Percent Move Analysis")
    fig_move = px.histogram(df, x='percent_move', title="Percent Move Distribution")
    st.plotly_chart(fig_move, use_container_width=True)
else:
    st.error(f"No backtest results found for {backtest_type}. Please run a backtest first.")
