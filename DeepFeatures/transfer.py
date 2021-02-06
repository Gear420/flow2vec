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
args = argparser.parse_args()


