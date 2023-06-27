import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow

def close_price(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date)
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    st.subheader('Closing price vs time')
    fig = plt.figure(figsize= (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)
    st.subheader('Closing price vs time with MA100 and MA200')
    fig = plt.figure(figsize= (12,6))
    plt.plot(df.Close)
    plt.plot(ma100, label = '100 days moving average')
    plt.plot(ma200, label = '200 days moving average')
    plt.legend()
    st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0 : int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70) : int(len(df))])

    scaler = MinMaxScaler(feature_range= (0,1))
    data_training_array = scaler.fit_transform(data_training)
    data_training_array = np.array(data_training_array)


    #ML model
    model = tensorflow.keras.models.load_model('E:\Github Projects\Stock market prediction\models\close_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')

    past_100 = data_training.tail(100)
    final_test_df = past_100.append(data_testing, ignore_index= True)

    final = scaler.fit_transform(final_test_df)

    x_test = []
    y_test = []

    for i in range(100, final.shape[0]):
        x_test.append(final[i-100 : i])
        y_test.append(final[i, 0])

    x_test = np.array(x_test) 
    y_test = np.array(y_test)  

    y_pred = model.predict(x_test)

    #Scaling up
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))


    st.subheader('Predictions')
    fig_2 = plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Closing price')
    plt.plot(y_pred, 'r', label = 'Predicted Closing Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig_2)