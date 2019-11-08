import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch as t
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sample_size, output_size=1, num_layers=2,dropout=0.5):
            super(LSTM, self).__init__()
            self.Lstm = nn.LSTM(input_size, hidden_size, num_layers,dropout=dropout)
            self.Regression = nn.Linear(sample_size*hidden_size, output_size)
          
    def forward(self, x):
        r_out, (h_n, h_c) = self.Lstm(x, None)   
        r_out = r_out.reshape(r_out.shape[0], r_out.shape[1]*r_out.shape[2])
        out = self.Regression(r_out)
        return out
