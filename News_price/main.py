
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from time import gmtime, strftime
from news_model import *
from dataloader import *


def train_val(epochs = 1, lr=3e-4):
    print("############# Train Model #############")
    # criterion = nn.SmoothL1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_mse, all_preds = val()
    best_model = deepcopy(model)
    best_preds = all_preds[:]

    for epoch in range(epochs):
        model.train()
        accumulate_loss = 0.0
        print("\nEpoch %d" % (epoch+1))
        for batch_idx, (X, Y) in tqdm(enumerate(train_loader)):
            if torch.cuda.is_available():
                X, Y = Variable(X).cuda(), Variable(Y).cuda()

            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print statistics
            accumulate_loss += loss.item()
            if batch_idx % base == 0:
                print('[batch: %3d] loss: %.5f' %
                      (batch_idx + 1, accumulate_loss / base))
                accumulate_loss = 0.0

        mse, all_preds = val()
        if mse < best_mse:
            best_mse = mse
            best_model = deepcopy(model)
            best_preds = all_preds[:]
    print("\nBest MSE = %.4f" % best_mse)

    np.save("val_preds-{}.npy".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime())), best_preds)
    return best_model


def val():
    print("\nValidate: ")
    model.eval()
    with torch.no_grad():
        mse = n = 0.0 
        all_pred = []
        for batch_idx, (X, Y) in tqdm(enumerate(val_loader)):
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            pred = model(X)
            
            all_pred.extend(list(np.squeeze(pred.cpu().data.numpy())))

            loss = criterion(pred, Y)
            mse += loss.item() * Y.size(0)
            n += Y.size(0)
        print("MSE = %.4f" % (mse / n))
    return mse/n, all_pred



if __name__ == '__main__':
    train_loader, val_loader = get_loader()
    
    model = NewsEmbedding(seq_length=5)

    base = 20

    # pretrain_model_path = ""
    # model.load_state_dict(torch.load(pretrain_model_path))
    # print("Load model from {}".format(pretrain_model_path))

    model = model.cuda()
    
    lr = 3e-4
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n**************  lr = 3e-3  ***************")
    best_model = train_val(epochs=30, lr=lr)
    
    # lr = 1e-3
    # print("\n**************  lr = 1e-3  ***************")
    # train_val(epochs=2)

    # lr = 3e-4
    # print("\n**************  lr = 3e-4  ***************")
    # train_val(epochs=2)

    # test_full(test_loader, model)
    
    model_path = "model-{}.pkl".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    torch.save(best_model.state_dict(), model_path)
    print("model saved as {}".format(model_path))


