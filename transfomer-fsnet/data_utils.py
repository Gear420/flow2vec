#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/21 3:08 PM
# @Author  : Gear

# @Author  : Gear


import random

import numpy as np
from torch.utils.data.dataset import Dataset
import torch
import pandas as pd


class FlowDataset(Dataset):
	def __init__(self, num_pkt_features, file_path):
		super(FlowDataset, self).__init__()
		self.num_pkt_features = num_pkt_features
		self.data = pd.read_csv(file_path, sep='\t', header=None)
		# find the maximum value of each feature dimension
		arr = np.arange(len(self.data.columns)) % num_pkt_features
		
		# self.data = self.data.drop([256 * 6 + 1], axis=1)
		
		arr[-1] = -1
		
		sizes = []  # embedding_sizes
		out_sizes = []
		self.flow_len = self.data.shape[1] - 1
		self.labels = np.asarray(self.data.iloc[:, -1])
		for i in range(num_pkt_features):
			arrt = (arr == i)
			elements = []
			for j in range(self.flow_len):
				if arrt[j] == True:
					col = list(self.data.iloc[:, j])
					elements += col
					elements = list(set(np.asarray(elements)))
			print("#elements_sizes:", len(elements))
			elements_size = len(elements)
			out_sizes.append(elements)
			sizes.append(elements_size)
		self.out_sizes = out_sizes
		self.sizes = sizes
		
		self.categories = list(set(self.labels))
		print("#Label_index:")
		print(self.categories)
		self.num_classes = len(self.categories)
		self.seq_len = (self.data.shape[1] - 1) // num_pkt_features
		print("#examples:", self.data.shape[0], "#classes:", self.num_classes)
	
	def elements2index(self, filename):
		file = open(filename, 'w+')
		for i in range(self.data.shape[0]):
			col = list(self.data.iloc[i])
			col = col[:-1]
			for j in range(len(col)):
				k = j % self.num_pkt_features
				file.write(str(self.out_sizes[k].index(col[j])))
				file.write('\t')
			file.write(str(self.labels[i]))
			file.write("\n")
	
	def __getitem__(self, index):
		label = self.labels[index]
		label_id = self.categories.index(label)
		vec = np.asarray(self.data.iloc[index][:-1])
		vec = torch.from_numpy(vec).reshape(-1, self.num_pkt_features)
		return vec, label_id
	
	def __len__(self):
		return self.data.shape[0]
	
	def transform_dataset(self):
		pass


if __name__ == '__main__':
	from torch.utils.data import DataLoader
	
	dataset = FlowDataset(6, "../fsnet_use_tcpc30_features")
	dataset.elements2index("fsnet_usetcpc30_index")
	# params = {'batch_size': 2, 'shuffle': True}
	# test_loader = DataLoader(dataset, **params)
	# for flows, c
	print(dataset.sizes)