
import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

# prices
def MACD(close_prices):
    EMA12 = close_prices.ewm(span=12).mean() 
    EMA26 = close_prices.ewm(span=26).mean() 
    res = EMA12-EMA26
    return res

def EMA(close_prices, length):
    return close_prices.ewm(span=length).mean() 

def MA(close_prices, length):
    return close_prices.rolling(length).mean()

def BOLL(close_prices):
    return close_prices.rolling(20).mean()

def add_feature(df):
    df.sort_values(by=['Date'], inplace=True)
    df['MACD'] = MACD(df['Adj Close'])
    df['EMA20'] = EMA(df['Adj Close'], 20)
    df['EMA50'] = EMA(df['Adj Close'], 50)
    df['BOLL'] = BOLL(df['Adj Close'])
    df['MA10'] = MA(df['Adj Close'], 10)
    df.dropna(inplace=True)
    name_order = ['Open','High','Low','Close','Volume','MACD','EMA20','BOLL','Adj Close']
    df = df[name_order]
    return df, name_order

def normalize(origin):
	origin["Open"] /= 20000
	origin["High"] /= 20000
	origin["Low"] /= 20000
	origin["Close"] /= 20000
	origin["Adj Close"] /= 20000
	origin["Volume"] /= 100000000
	origin["MACD"] /= 100
	origin["EMA20"] /= 20000
	origin["BOLL"] /= 20000
	return origin

def save_expanded_price_data()
	origin = pd.read_csv('../Data/DJIA_table.csv', index_col=0)
	origin, _ = add_feature(origin)
	origin = normalize(origin)
	origin.to_csv("../Data/expanded_price")








