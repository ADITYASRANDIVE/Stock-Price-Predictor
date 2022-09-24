import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import pandas_datareader.data as pdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.layers import Dense, LSTM
yf.pdr_override()
plt.style.use('fivethirtyeight')
import streamlit as st
from yahoo_fin import stock_info


start = '2010-01-01'
end = datetime.date.today()
#st.write(newend)
st.title('Stock Trend Prediction')
user_input_stock = st.text_input('Enter stock Ticker')
df = pdr.get_data_yahoo(user_input_stock, start, end)
#describing data
st.subheader('Data from 2010-2022')
st.write(df.describe())
'''
#Particular Date
st.subheader('Want data for a specific Date?')
year = st.text_input('Enter year')
month = st.text_input('Enter month')
day = st.text_input('Enter day')
a = pdr.get_data_yahoo(user_input_stock, start=datetime(int(year), int(month), int(day)+1), end=datetime(int(year), int(month), int(day)+1))
st.write(a)'''

#visualisations
st.subheader('Closing price VS Time Series')
fig = plt.figure(figsize=(16,8))
plt.plot(df.Close)
st.pyplot(fig)

#Create a new dataframe with only close cloumn
data = df.filter(['Close'])
dataset = data.values
train_data_len = math.ceil(len(dataset)*0.8)
#train_data_len

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#scaled_data

#create training dataset
train_data = scaled_data[0:train_data_len,:]
x_train=[]
y_train=[]
for i in range(100,len(train_data)):
  x_train.append(train_data[i-100:i,0])
  y_train.append(train_data[i,0])
  #if i <=101:
    #print(x_train)
    #print(y_train)
    #print()

#convert x and y train into numpy arrays
x_train,y_train = np.array(x_train), np.array(y_train)
#reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
#x_train.shape
#loading the model
model = load_model('keras_prediction_model.h5')

#creating test datset
#create a new array containing scaled values
test_data = scaled_data[train_data_len-100: ,:]
x_test=[]
y_test=dataset[train_data_len: ,:]
for i in range(100,len(test_data)):
  x_test.append(test_data[i-100:i,0])
#convert test data to numpy array
x_test = np.array(x_test)
#x_test.shape
#reshape
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
#get the models predicted values
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
#get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean((pred-y_test)**2))
#rmse

#plot the data
st.subheader('Predictions VS Actual Price')
train = data[:train_data_len]
valid = data[train_data_len:]
valid['Predictions'] = pred
#visiualise the data
fig1 = plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close price USD($)')
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc = 'lower right')
st.pyplot(fig1)

#make the Predictions
apple = pdr.get_data_yahoo(user_input_stock, start, end)

new_df = apple.filter(['Close'])

past_100_days = new_df[-100:].values
past_100_days_scaled = scaler.transform(past_100_days)
X_test = []
X_test.append(past_100_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
live_price = stock_info.get_live_price(user_input_stock)
st.subheader('Live Price: ')
st.write(f"Live price is: ",live_price)
st.subheader('Prediction for'+' '+ user_input_stock +' '+ 'Tomorrow is:')
st.write(f"PREDICTION:",pred_price)


st.subheader("Stock-O-Bot Recommendation: ")
if live_price>pred_price:
  st.write("You should sell the stocks")
elif pred_price>live_price:
  st.write("You should buy more of this stocks")
else:
  st.write("You should hold the stocks for now")

