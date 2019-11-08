import data_preprocess
import model
import torch
import numpy as np
import matplotlib.pyplot as plt 
from torch.autograd import Variable

def calculate_MAPE(x,y,drop=20):
    error=0
    for i in range(drop,len(x)):
        if y[i]>=0.1:
            error+=abs(x[i]-y[i])/x[i]
    return 1/(len(x)-drop)*error

def calculate_R(x,y,drop=20):
    x_m=np.mean(x[drop:])
    y_m=np.mean(y[drop:])
    a=0
    b=0
    c=0
    for i in range(drop,len(x)):
        a+=(x[i]-x_m)*(y[i]-y_m)
        b+=(x[i]-x_m)**2
        c+=(y[i]-y_m)**2
    return a/(np.sqrt(b*c))
        
def calculate_TheilU(x,y,drop=20):
    a=0
    b=0
    c=0
    for i in range(drop,len(x)):
        a+=(x[i]-y[i])**2
        b+=x[i]**2
        c+=y[i]**2
    return np.sqrt(a)/(np.sqrt(b)+np.sqrt(c))
    
    
if __name__ == '__main__':
    x_train,y_train,x_test,y_test=data_preprocess.data_process(10)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    lstm = model.LSTM(9,64,10)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
    for e in range(100):
        var_x = Variable(x_train)
        var_y = Variable(y_train)
        out = lstm(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: {}, Loss:{:.5f}'.format(e+1, loss.data.item()))
    
    var_data = Variable(x_train)
    pred_train = lstm(var_data) 
    pred_train = pred_train.view(-1).data.numpy()
    y_train = y_train.view(-1).data.numpy()
    MAPE=calculate_MAPE(pred_train,y_train)
    R=calculate_R(pred_train,y_train)
    TheilU=calculate_TheilU(pred_train,y_train)
    print(MAPE,R,TheilU)
    plt.figure(figsize=(12,8))
    plt.plot(pred_train, 'r', label='prediction')
    plt.plot(y_train, 'b', label='real')
    plt.legend(loc='best')
    plt.show()
    
    var_data = Variable(x_test)
    pred_test = lstm(var_data) 
    pred_test = pred_test.view(-1).data.numpy()
    y_test = y_test.view(-1).data.numpy()
    MAPE=calculate_MAPE(pred_test,y_test)
    R=calculate_R(pred_test,y_test)
    TheilU=calculate_TheilU(pred_test,y_test)
    print(MAPE,R,TheilU)
    plt.figure(figsize=(12,8))
    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(y_test, 'b', label='real')
    plt.legend(loc='best')
    plt.show()
