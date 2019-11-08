
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import gmtime, strftime
from news_model import *
from dataloader import *


def train_val(epochs = 1, lr=3e-4):
    print("############# Train Model #############")
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    val()

    for epoch in range(epochs):
        model.train()
        accumulate_loss = 0.0
        acc_num = total_num = 0
        print("\nEpoch %d" % (epoch+1))
        for batch_idx, (X, Y) in tqdm(enumerate(train_loader)):
            if torch.cuda.is_available():
                X, Y = Variable(X).cuda(), Variable(Y).cuda()

            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print("pred: {}".format(pred))
            # print("Y: {}".format(Y))
            # print statistics
            accumulate_loss += loss.item()
            acc_num += torch.sum(torch.floor(pred+0.5) == Y.float()).item()
            total_num += Y.size(0)
            if 1+batch_idx % base == 0:
                print('[batch: %3d] loss: %.5f' %
                      (batch_idx + 1, accumulate_loss / base))
                accumulate_loss = 0.0
        print("\nTraining Accuracy = %.4f" % (acc_num / total_num))

        val()


def val():
    print("")
    model.eval()
    with torch.no_grad():
        acc_num = total_num = 0
        for batch_idx, (X, Y) in tqdm(enumerate(val_loader)):
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()
            pred = model(X)
            
            acc_num += torch.sum(torch.floor(pred+0.5) == Y.float()).item()
            total_num += Y.size(0)
        print("Validate Accuracy = %.4f" % (acc_num / total_num))




if __name__ == '__main__':
    train_loader, val_loader = get_loader()
    
    model = NewsEmbedding(seq_length=5)

    base = 20

    # pretrain_model_path = ""
    # model.load_state_dict(torch.load(pretrain_model_path))
    # print("Load model from {}".format(pretrain_model_path))

    model = model.cuda()

    lr = 3e-4
    print("\n**************  lr = 3e-3  ***************")
    train_val(epochs=50, lr=lr)
    
    # lr = 1e-3
    # print("\n**************  lr = 1e-3  ***************")
    # train_val(epochs=2)

    # lr = 3e-4
    # print("\n**************  lr = 3e-4  ***************")
    # train_val(epochs=2)

    # test_full(test_loader, model)
    
    model_path = "model-{}.pkl".format(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    torch.save(model.state_dict(), model_path)
    print("model saved as {}".format(model_path))


