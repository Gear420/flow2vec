#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 11:09 PM
# @Author  : Gear
import torch
import torch.nn as nn


class CNN(nn.moudle):
	def __init__(self,num_classes):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc = nn.Linear(7 * 7 * 32, num_classes)
	def forward(self,x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out



