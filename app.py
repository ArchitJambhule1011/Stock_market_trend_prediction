import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from src.closing_price_plot import close_price
from src.opening_price_plot import open_price

st.title('Stock market trend prediction')
st.markdown('''
Predicting stock market trends using LSTM neural networks.
''')

choice = st.radio('Please select an option you want to predict for', ('Open Price', 'Close Price'))

start_date = '2010-01-01'
end_date = '2019-12-31'

user_input = st.text_input('Enter stock ticker', 'TSLA')
df = yf.download(user_input, start=start_date, end=end_date)

st.subheader('Data Description')
st.write(df.describe())

#Moving averages for close
if choice == 'Close Price':
    close_price(user_input = user_input, start_date = start_date, end_date = end_date)

if choice == 'Open Price':
    open_price(user_input = user_input, start_date = start_date, end_date = end_date)


