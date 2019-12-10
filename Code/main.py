
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn import metrics



def trend_train(model, train_loader, num_epochs=10)
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    for epoch in range(num_epochs):
        print('\nEpoch %d' % (epoch+1))
        label_arr = []
        pred_arr = []
        for i, (price, news, y) in enumerate(train_loader):
            price, news, y = Variable(price), Variable(news), Variable(y)

            optimizer.zero_grad()
            outputs = model(price, news)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            label_arr += list(y.detach().numpy())
            pred_arr += list(predicted.detach().numpy())
            
            if (i+1) % 20 == 0:
                acc = metrics.accuracy_score(label_arr, pred_arr)
                f1 = metrics.f1_score(label_arr, pred_arr)
                print ('Iter %d Loss: %.4f, acc = %.4f, f1 = %.4f' % (i+1, loss.item(), acc, f1))
        acc = metrics.accuracy_score(label_arr, pred_arr)
        f1 = metrics.f1_score(label_arr, pred_arr)
       	print('Train acc: %.4f, f1: %.4f' % (acc, f1))
    return model 


def trend_val(model, val_loader):   
    model.eval()
    label_arr = []
    pred_arr = []
    for price, news, y in val_loader:
        price, news = Variable(price), Variable(news)
        outputs = model(price, news)

        _, predicted = torch.max(outputs.data, 1)
        label_arr += list(y.detach().numpy())
        pred_arr += list(predicted.detach().numpy())
    
    acc = metrics.accuracy_score(label_arr, pred_arr)
    f1 = metrics.f1_score(label_arr, pred_arr)
    print('Val acc: %.4f, f1: %.4f' % (acc, f1))


def price_train(model, train_loader, num_epochs=10)
    num_epochs = 10
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())

    for epoch in range(num_epochs):
        print('\nEpoch %d' % (epoch+1))
        label_arr = []
        pred_arr = []
        for i, (price, news, y) in enumerate(train_loader):
            price, news, y = Variable(price), Variable(news), Variable(y)

            optimizer.zero_grad()
            outputs = model(price, news)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            label_arr += list(y.detach().numpy())
            pred_arr += list(outputs.detach().numpy())
            
            if (i+1) % 20 == 0:
                mae = metrics.mean_absolute_error(label_arr, pred_arr)
                mse = metrics.mean_squared_error(label_arr, pred_arr)
                print ('Iter %d Loss: %.4f, mse = %.4f, mae = %.4f' % (i+1, loss.item(), mse, mae))
	    mae = metrics.mean_absolute_error(label_arr, pred_arr)
        mse = metrics.mean_squared_error(label_arr, pred_arr)
	   	print('Train mse: %.4f, mae: %.4f' % (mse, mae))
	return model 


def price_val(model, val_loader):   
    model.eval()
    label_arr = []
    pred_arr = []
    for price, news, y in val_loader:
    	price, news = Variable(price), Variable(news)
        outputs = model(price, news)

        _, predicted = torch.max(outputs.data, 1)
        label_arr += list(y.detach().numpy())
        pred_arr += list(predicted.detach().numpy())
    
    mae = metrics.mean_absolute_error(label_arr, pred_arr)
    mse = metrics.mean_squared_error(label_arr, pred_arr)
   	print('Val mse: %.4f, mae: %.4f' % (mse, mae))
    
















