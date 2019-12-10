
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FullModel(nn.Module):
	def __init__(self, p_input_size = 9, e_input_size = 768, pattern = "m2o", task = "classification"):
		super(FullModel, self).__init__()
		self.task = task
		self.pattern = pattern

		# price pipeline
		self.p_hidden_size = 32
		self.fc1 = nn.Linear(p_input_size, self.p_hidden_size)
		self.gru = nn.GRU(input_size=self.p_hidden_size,
						hidden_size=self.p_hidden_size,
						num_layers=1,
						batch_first=True,
						bidirectional=False)
		# event pipeline
		self.e_hidden_size = 32
		self.fc2 = nn.Linear(e_input_size, self.e_hidden_size)
		self.bilstm = nn.LSTM(input_size=self.e_hidden_size,
							hidden_size=self.e_hidden_size,
							num_layers=1,
							batch_first=True,
							bidirectional=True)
		# feature fusion
		fusion_size = self.e_hidden_size + self.p_hidden_size
		num_output = 2 if self.task == "classification" else 1
		self.feature_fusion = nn.Sequential(
								nn.Linear(fusion_size, fusion_size // 2),
								nn.ReLU(), 
								nn.Linear(fusion_size // 2, num_output)
								)


	def init_price_hidden(self, batch_size = 1):
		if self.pattern == 'm2o':
			h0 = Variable(torch.zeros(1, batch_size, self.p_hidden_size))
			if torch.cuda.is_available():
				h0 = h0.cuda()
			return h0
		else:
			self.hn = Variable(torch.zeros(1, batch_size, self.p_hidden_size))
			if torch.cuda.is_available():
				self.hn = self.hn.cuda()
			

	def init_event_hidden(self, batch_size = 1):
		h0 = Variable(torch.zeros(1 * 2, batch_size, self.e_hidden_size))
		c0 = Variable(torch.zeros(1 * 2, batch_size, self.e_hidden_size))
		if torch.cuda.is_available():
			h0 = h0.cuda()
			c0 = c0.cuda()
		return h0, c0


	def forward(self, price, event):
		# price forward
		price = F.relu(self.fc1(price))
		if self.pattern == 'm2o':
			p_h0 = self.init_price_hidden(batch_size = price.size(0))
			price_out, _ = self.gru(price, p_h0)
		else:
			price_out, self.hn = self.gru(price, self.hn)
		price_feature = price_out[:, -1, :]
		print(price_feature.size())

		# event forward
		event = F.relu(self.fc2(event))
		e_h0, e_c0 = self.init_event_hidden(batch_size = event.size(0))
		event_out, _ = self.bilstm(event, (e_h0, e_c0))
		# hidden states from forward LSTM and backward LSTM
		forward_hidden = event_out[:, -1, :self.e_hidden_size]
		backward_hidden = event_out[:, 0, self.e_hidden_size:]
		event_feature = (forward_hidden + backward_hidden) / 2
		print(event_feature.size())

		# feature fusion
		concat_feature = torch.cat((price_feature, event_feature), axis=1)
		print(concat_feature.size())
		pred = self.feature_fusion(concat_feature)
		print(pred.size())

		return pred




if __name__ == '__main__':
	model = FullModel(p_input_size=9, e_input_size=768, pattern="m2o", task="classification")
	price = torch.randn(12, 7, 9)
	event = torch.randn(12, 5, 768)
	model(price, event)


	model = FullModel(p_input_size=9, e_input_size=768, pattern="m2o", task="regression")
	price = torch.randn(12, 7, 9)
	event = torch.randn(12, 5, 768)
	model(price, event)

	model = FullModel(p_input_size=9, e_input_size=768, pattern="seq2seq", task="classification")
	model.init_price_hidden()
	price = torch.randn(1, 1, 9)
	event = torch.randn(1, 5, 768)
	model(price, event)

	model = FullModel(p_input_size=9, e_input_size=768, pattern="seq2seq", task="regression")
	model.init_price_hidden()
	price = torch.randn(1, 1, 9)
	event = torch.randn(1, 5, 768)
	model(price, event)







