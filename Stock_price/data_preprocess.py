import pandas as pd
import numpy as np
from sklearn import preprocessing

file="DJIA_table.csv"

def cal_MACD(close_prices):
    EMA12=close_prices.ewm(span=12).mean() 
    EMA26=close_prices.ewm(span=26).mean() 
    MACD=EMA12-EMA26
    return MACD

def cal_EMA(close_prices,length):
    return close_prices.ewm(span=length).mean() 

def cal_MA(close_prices,length):
    return close_prices.rolling(length).mean()

def cal_BOLL(close_prices):
    return close_prices.rolling(20).mean()

def add_feature():
    data = pd.read_csv(file)
    df = pd.DataFrame(data)
    df.sort_values(by=['Date'],inplace=True)
    df['MACD']=cal_MACD(df['Adj Close'])
    df['EMA20']=cal_EMA(df['Adj Close'],20)
    df['EMA50']=cal_EMA(df['Adj Close'],50)
    df['BOLL']=cal_BOLL(df['Adj Close'])
    df['MA10']=cal_MA(df['Adj Close'],10)
    df.dropna(inplace=True)
    #name_order=['Open','High','Low','Close','Volume','MACD'
    #          ,'EMA20','EMA50','BOLL','MA10','Adj Close']
    name_order=['Open','High','Low','Close','Volume','MACD','EMA20','BOLL','Adj Close']
    df=df[name_order]
    return df,name_order

def data_normalization():
    df,name_order=add_feature()
    min_max_scaler = preprocessing.MinMaxScaler()
    newdf= df.copy()
    for i in range(0,len(name_order)):
        newdf[name_order[i]]=min_max_scaler.fit_transform(df[name_order[i]].values.reshape(-1,1))
    return newdf

def data_split(data,sample):
    #convert df into array
    dimension=len(data.columns)
    print(dimension)
    data = data.as_matrix()
    x_data = []
    y_data = []
    
    #we set batch size as sample
    for i in range(len(data)-sample):
        x_data.append(data[i:(i+sample)])
        y_data.append(data[i+sample,-1:])
    
    #use 70% as train data    
    n = int(round(0.8*len(x_data)))
    x_data = np.array(x_data)
    
    x_train = x_data[:n] 
    y_train = y_data[:n]
    x_test = x_data[n:]
    y_test = y_data[n:]
    
    x_train = x_train.reshape(-1, sample, dimension)
    x_test = x_test.reshape(-1, sample, dimension)
    return x_train,np.array(y_train), x_test, np.array(y_test)

def data_process(sample_size):
    df=data_normalization()
    df=df.astype('float32')
    x_train,y_train,x_test,y_test= data_split(df,sample_size)
    return x_train,y_train,x_test,y_test

