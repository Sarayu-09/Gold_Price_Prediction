import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM

def load_data():
    df = pd.read_csv('Gold Price.csv')
    df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    NumCols = df.columns.drop(['Date'])
    df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
    df[NumCols] = df[NumCols].astype('float64')
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()
    scaler.fit(df.Price.values.reshape(-1,1))
    window_size = 60
    test_size = df[df.Date.dt.year == 2022].shape[0]
    
    train_data = df.Price[:-test_size]
    train_data = scaler.transform(train_data.values.reshape(-1,1))
    test_data = df.Price[-test_size-60:]
    test_data = scaler.transform(test_data.values.reshape(-1,1))
    
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
        y_test.append(test_data[i, 0])
        
    X_train, X_test = np.array(X_train), np.array(X_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1,1))
    y_test = np.reshape(y_test, (-1,1))
    
    return X_train, y_train, X_test, y_test, scaler, test_size

def define_model(window_size):
    input1 = Input(shape=(window_size,1))
    x = LSTM(units = 64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    dnn_output = Dense(1)(x)
    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    return model

def plot_data(df, test_size, y_test_true=None, y_test_pred=None):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df['Date'].iloc[:-test_size], df['Price'].iloc[:-test_size], color='black', lw=2)
    if y_test_true is not None and y_test_pred is not None:
        ax.plot(df['Date'].iloc[-test_size:], y_test_true, color='blue', lw=2)
        ax.plot(df['Date'].iloc[-test_size:], y_test_pred, color='red', lw=2)
        ax.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})
    else:
        ax.plot(df['Date'].iloc[-test_size:], df['Price'].iloc[-test_size:], color='blue', lw=2)
        ax.legend(['Training Data', 'Test Data'], loc='upper left', prop={'size': 15})
    ax.set_title('Model Performance on Gold Price Prediction', fontsize=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    plt.grid(color='white')
    st.pyplot(fig)

def main():
    st.title("Gold Price Prediction")
    st.write("This app predicts the price of gold using historical data.")
    
    df = load_data()
    st.write("## Data Overview")
    st.write(df)
    
    X_train, y_train, X_test, y_test, scaler, test_size = preprocess_data(df)
    
    st.write("## Training the Model")
    model = define_model(60)
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
    
    st.write("## Model Evaluation")
    result = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    Accuracy = 1 - MAPE
    st.write("Test Loss:", result)
    st.write("Test MAPE:", MAPE)
    st.write("Test Accuracy:", Accuracy)
    
    y_test_true = scaler.inverse_transform(y_test)
    y_test_pred = scaler.inverse_transform(y_pred)
    
    st.write("## Model Performance on Gold Price Prediction")
    plot_data(df, test_size, y_test_true, y_test_pred)
    
if __name__ == "__main__":
    main()
