import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def open_price(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date)
    ma100_open = df.Open.rolling(100).mean()
    ma200_open = df.Open.rolling(200).mean()
    st.subheader('Opening price vs time')
    fig = plt.figure(figsize= (12,6))
    plt.plot(df.Open)
    st.pyplot(fig)
    st.subheader('Opening price vs time with MA100 and MA200')
    fig = plt.figure(figsize= (12,6))
    plt.plot(df.Open)
    plt.plot(ma100_open, label = '100 days moving average')
    plt.plot(ma200_open, label = '200 days moving average')
    plt.legend()
    st.pyplot(fig)