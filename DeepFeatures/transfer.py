#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/2 12:46 AM
# @Author  : Gear


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os


import argparse

import torch
import torch.nn
from torch.utils.data import DataLoader, random_split

from dataset import FlowDataset
from model import Transformer

import argparse

import torch
import torch.nn
from torch.utils.data import DataLoader, random_split

from dataset import FlowDataset
from model import Transformer

argparser = argparse.ArgumentParser()
argparser.add_argument('dataset')
argparser.add_argument('--model')
argparser.add_argument('--batch_size',
                       type=int,
                       default=64,
                       help="batch size (default 10)")
argparser.add_argument('--classes',
                       type=int,
                       default=64,
                       help="transfer classes")
args = argparser.parse_args()

dataset = FlowDataset(args.num_pkt_features, args.dataset)
model_ft = torch.load(args.model)

num_ftrs = model_ft.fc.in_features             # 全连接层的输入的特征数

model_ft.fc = nn.Linear(num_ftrs, args.classes)           # 利用线性映射将原来的num_ftrs转换为2（蚂蚁和密封）
                                               # 将最后一个全连接由（512， 1000）改为(512, 2)   因为原网络是在1000类的ImageNet数据集上训练的                # 设置计算采用的设备，GPU还是CPU

criterion = nn.CrossEntropyLoss()              # 交叉熵损失函数

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)   # 优化器，对加载的模型中所有参数都进行优化

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

params = {'batch_size': args.batch_size, 'shuffle': True}

# 分割数据80:20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(0)
train_set, test_set = random_split(dataset, (train_size, test_size))
train_loader = DataLoader(train_set, **params)
test_loader = DataLoader(test_set, **params)

iter = 0
cur_loss = 0
all_losses = []

device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
print("using device:", device)
model = model.to(device)
# 训练模型
timer.tick()


def evaluate(model):
	model.eval()
	with torch.no_grad():
		correct = total = 0
		for flows, categories in test_loader:
			flows = flows.to(device)
			categories = categories.to(device)
			outputs = model(flows)
			_, predicted = torch.max(outputs.data, 1)
			total += categories.size(0)
			correct += (predicted == categories).sum().item()
	print('Accuracy: {:.3f}%'.format(100 * correct / total))
	acc = 100 * correct / total
	return acc


best_acc = float('-inf')

try:
	for epoch in range(1, args.num_epochs + 1):
		model.train()
		for flows, categories in train_loader:
			# 正向计算
			flows = flows.to(device)
			categories = categories.to(device)
			outputs = model(flows)
			loss = criterion(outputs, categories)
			
			# 反向传播和优化
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss = loss.item()
			
			cur_loss += loss
			
			iter += 1
			if iter % 100 == 0:
				print('epoch {} of {} ({}), loss: {:.4f}'.format(
					epoch, args.num_epochs, timer.tmstr(), loss))
				# Add current loss avg to list of losses
				all_losses.append(cur_loss / 100)
				cur_loss = 0
		acc = evaluate(model)
		if acc > best_acc:
			save()
			best_acc = acc
except KeyboardInterrupt:
	print("Save before quit...")

try:
	from matplotlib import pyplot as plt
	
	plt.figure()
	plt.plot(all_losses)
	plt.show()
	plt.savefig("result_loss.jpg")
except ImportError:
	pass
