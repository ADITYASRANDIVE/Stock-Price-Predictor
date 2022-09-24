import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
from io import BytesIO
import base64

def get_details(symbol):
    ticker = yf.Ticker(symbol)
    data = ticker.info['longBusinessSummary']
    return data
def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    price = float("{:.2f}".format(todays_data['Close'][0]))
    return price

def get_chart(symbol):
   
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='3y')
    df = pd.DataFrame(todays_data)
    df.reset_index(inplace=True)
    plt.style.use("fivethirtyeight")
    df.plot(x ='Date', y='Close')
    # function to show the plot
    buffer = BytesIO() 
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    return graphic

def get_recommend(pred_price,live_price):
    if live_price>pred_price:
        rec = "You should sell the stocks"
    elif pred_price>live_price:
        rec = "You should buy more of this stocks"
    else:
        rec = "You should hold the stocks for now"
    return rec
def get_compare(symbol1,symbol2):
    ticker1 = yf.Ticker(symbol1)
    ticker2 = yf.Ticker(symbol2)
    todays_data1 = ticker1.history(period='1y')
    todays_data2 = ticker2.history(period='1y')
    df1 = pd.DataFrame(todays_data1)
    df1.reset_index(inplace=True)
    ax = df1.plot(x ='Date', y='Close')
    df2 = pd.DataFrame(todays_data2)
    df2.reset_index(inplace=True)
    df2.plot(x ='Date', y='Close',ax=ax)
    # function to show the plot
    buffer = BytesIO() 
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    return graphic
def get_prediction(symbol):
    from keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5y")
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
    model = load_model('mystockapp/keras_prediction_model.h5')

    #create test datset
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
    train = data[:train_data_len]
    valid = data[train_data_len:]
    valid['Predictions'] = pred
    #visiualise the data
    fig1 = plt.figure(figsize=(10,5))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close price USD($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train','Val','Predictions'],loc = 'lower right')
    # function to show the plot
    buffer = BytesIO() 
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    return graphic
    
def get_pred_price(symbol,date):
    from keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5y")
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
    model = load_model('mystockapp/keras_prediction_model.h5')

    #create test datset
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
    start = '2010-01-01'
    end = datetime.datetime.today() - datetime.timedelta(days=1)
    end = end.strftime("%Y-%m-%d") 
    newdata = yf.download(symbol,start=start,end=end)
    new_df = newdata.filter(['Close'])

    past_100_days = new_df[-100:].values

    past_100_days_scaled = scaler.transform(past_100_days)

    X_test = []

    X_test.append(past_100_days_scaled)

    X_test = np.array(X_test)

    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    text = pred_price
    strippedText = str(text).replace('[','').replace(']','').replace('\'','').replace('\"','')
    strippedText = float("{:.2f}".format(float(strippedText)))
    return strippedText