import torch.utils.data as data
import torch
import numpy as np
from tqdm import tqdm
import os

data_path = "../Data/"

class NewsDataset(data.Dataset):
	def __init__(self, stage='train'):
		if stage == 'train':
			self.X = np.load(os.path.join(data_path, "train_data.npy"))
			self.Y = np.load(os.path.join(data_path, "train_label.npy"))
		else:
			self.X = np.load(os.path.join(data_path, "val_data.npy"))
			self.Y = np.load(os.path.join(data_path, "val_label.npy"))

	def __getitem__(self, index):
		return self.X[index], self.Y[index]

	def __len__(self):
		return len(self.Y)


def get_loader():
	# batch_size = 32
	train_data = NewsDataset("train")
	val_data = NewsDataset("val")

	train_loader = data.DataLoader(dataset=train_data,
									batch_size=32, 
									shuffle = False,
									# collate_fn = my_collate_fn,
									num_workers = 4)
	val_loader = data.DataLoader(dataset=val_data,
								batch_size=64, 
								shuffle = False,
								# collate_fn = my_collate_fn,
								num_workers = 4)
	return train_loader, val_loader


