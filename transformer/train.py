#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 2:45 AM
# @Author  : Gear

# ! /usr/bin/env python3
# -*- coding: utf-8 -*-


import myutils
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader, random_split

from dataset import FlowDataset
from model import Transformer

import argparse
import os

import myutils
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader, random_split

from dataset import FlowDataset
from model import Transformer

timer = myutils.Timer()

argparser = argparse.ArgumentParser()
argparser.add_argument('dataset')
argparser.add_argument('--num_epochs',
                       type=int,
                       default=100,
                       help="epochs (default 10)")
argparser.add_argument('--learning_rate',
                       type=float,
                       default=0.0001,
                       help="learning rate (default 0.0001)")
argparser.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help="batch size (default 10)")
argparser.add_argument("--use_cuda", action='store_true', help='run prepare_data or not')
argparser.add_argument('-O', default='data/model.pt', help='path ot save model')

args = argparser.parse_args()

outdir = os.path.dirname(args.O)
if not os.path.exists(outdir): os.mkdir(outdir)

dataset = FlowDataset(args.num_pkt_features, args.dataset)
model = Transformer(dataset.sizes, dataset.num_classes, args.use_cuda)


def save():
	torch.save(model, args.O)
	print('Saved to', args.O)


def weigth_init(m):
	if isinstance(m, torch.nn.BatchNorm2d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	elif isinstance(m, torch.nn.Linear):
		m.weight.data.normal_(0, 0.01)
		m.bias.data.zero_()


model.apply(weigth_init)
# 损失和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Parameters
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
