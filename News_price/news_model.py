
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class NewsEmbedding(nn.Module):
	def __init__(self, batch_size=32, seq_length=5):
		super(NewsEmbedding, self).__init__()
		self.input_size = 768
		self.batch_size = batch_size
		self.seq_length = 5
		self.hidden_size = 768
		self.num_layers = 1
		self.num_directions = 2
		batch_first = True
		bidirectional = (self.num_directions==2)
		self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.input_size,
                            num_layers=self.num_layers,
                            batch_first=batch_first,
                            bidirectional=bidirectional)
		
		self.classifier = nn.Sequential(
							nn.Linear(self.hidden_size, 1024),
							nn.ReLU(),
							nn.Linear(1024, 128),
							nn.ReLU(),
							nn.Linear(128, 1),
							# nn.Sigmoid()
						)

	def init_hidden(self, batch_size):
		h0 = Variable(torch.rand(self.num_layers * self.num_directions, batch_size, self.hidden_size))
		c0 = Variable(torch.rand(self.num_layers * self.num_directions, batch_size, self.hidden_size))
		if torch.cuda.is_available():
			h0 = h0.cuda()
			c0 = c0.cuda()
		return h0, c0


	def forward(self, x):
                # x = (batch_size, seq_len, input_size)
                h0, c0 = self.init_hidden(x.size(0))
                output, _ = self.lstm(x, (h0, c0))
                normal_out = output[:, -1, :self.hidden_size]
                reverse_out = output[:, 0, self.hidden_size:]
                combine_out = (normal_out + reverse_out).squeeze(1)	# daily news (5) embedding
                pred = self.classifier(combine_out)
                
                # pred = self.classifier(torch.mean(x, axis=1))
                return pred


if __name__ == '__main__':
	inp = Variable(torch.rand(5, 6, 768))
	model = NewsEmbedding(batch_size=8, seq_length=5)
	res = model(inp)
	print(res)








