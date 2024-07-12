import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load and preprocess data
df = pd.read_csv('Gold Price.csv')
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')

# Sidebar for user inputs
st.sidebar.header('User Input Parameters')
test_year = st.sidebar.slider('Test Year', 2000, 2023, 2022)
window_size = st.sidebar.slider('Window Size', 10, 120, 60)
epochs = st.sidebar.slider('Epochs', 10, 200, 150)
batch_size = st.sidebar.slider('Batch Size', 16, 128, 32)


test_size = df[df.Date.dt.year == test_year].shape[0]

# Data scaling and windowing
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))
train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1, 1))
X_train, y_train = [], []

for i in range(window_size, len(train_data)):
    X_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Model definition
def define_model():
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Model training
model = define_model()
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# Predict function
def predict_gold_price(input_data):
    input_data = scaler.transform(np.array(input_data).reshape(-1, 1))
    X_input = []
    for i in range(window_size, len(input_data)):
        X_input.append(input_data[i - window_size:i, 0])
    X_input = np.array(X_input)
    X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
    prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction)
    return prediction[-1][0]

# User input for prediction
st.header('Predict Gold Price')
user_input = st.text_area('Enter the past 60 days prices separated by comma')
if st.button('Predict'):
    try:
        input_data = list(map(float, user_input.split(',')))
        if len(input_data) == window_size:
            prediction = predict_gold_price(input_data)
            st.write(f'The predicted gold price is: {prediction}')
        else:
            st.write('Please enter exactly 60 values.')
    except ValueError:
        st.write('Please enter valid numeric values separated by commas.')

# Model evaluation and performance plot
if st.checkbox('Show Model Performance'):
    # Prepare test data for evaluation
    test_data = df.Price[-test_size - window_size:]
    test_data = scaler.transform(test_data.values.reshape(-1, 1))
    X_test, y_test = [], []

    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i - window_size:i, 0])
        y_test.append(test_data[i, 0])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = np.reshape(y_test, (-1, 1))

    # Model evaluation
    result = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    Accuracy = 1 - MAPE

    st.write("Test Loss:", result)
    st.write("Test MAPE:", MAPE)
    st.write("Test Accuracy:", Accuracy)

    # Returning actual and predicted price values to primary scale
    y_test_true = scaler.inverse_transform(y_test)
    y_test_pred = scaler.inverse_transform(y_pred)

    # Investigating the closeness of the prices predicted by the model to the actual prices
