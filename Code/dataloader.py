import torch
import torch.utils.data as data


class BothData(data.Dataset):
	def __init__(self, price_data, event_data):
		self.price_data = price_data
		self.event_data = event_data

	def __getitem__(self, index):
		price, y = self.price_data.__getitem__(index)
		event, _ = self.event_data.__getitem__(index)
		return price, event, y

	def __len__(self):
		return len(self.price_data)


class PriceData(data.Dataset):
	def __init__(self, price_dataset, labels):
		self.price_dataset = price_dataset
		self.labels = labels

	def __getitem__(self, index):
		price = self.price_dataset[index]
		y = self.labels[index]
		return price, y

	def __len__(self):
		return len(self.labels)


class EventData(data.Dataset):
	def __init__(self, event_dataset, labels):
		self.event_dataset = event_dataset
		self.labels = labels

	def __getitem__(self, index):
		event = self.event_dataset[index]
		y = self.labels[index]
		return event, y

	def __len__(self):
		return len(self.labels)


