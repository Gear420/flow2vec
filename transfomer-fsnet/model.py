#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/20 3:03 PM
# @Author  : Gear

import torch
import torch.nn as nn
from data_utils import *


class TfNet(nn.Module):
	def __init__(self, out_size, num_classes, num_pkt_features, embedding_dim=128, hidden_dim=32,
	             num_layers=1,
	             ):
		super(TfNet, self).__init__()
		self.seq_len = 0
		self.embedding_size = out_size
		
		
		self.out_sizes = out_size
		self.num_pkt_features = num_pkt_features
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.num_classes = num_classes
		
		# 构建模型。
		self.embedding = self._embedding()  # embedding数目为layer数
		self.encoder_layer = self._encoder()
		self.decoder_layer = self._decoder()
		self.cls_layer = self._classify()
		self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
	
	def _embedding(self):
		embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
		return embedding
	
	def _encoder(self):
		num_heads = 1
		num_layers = 2
		encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, num_heads)
		encoder = nn.TransformerEncoder(encoder_layer, num_layers)
		return encoder
	
	def _decoder_input(self, seq_len, encoder_feats):
		encoder_feats = encoder_feats.unsqueeze(1)
		decoder_input = encoder_feats.repeat(1, seq_len, 1)
		return decoder_input
	
	def _decoder(self):
		num_heads = 1
		num_layers = 2
		decoder_layer = nn.TransformerDecoderLayer(self.embedding_dim, num_heads)
		# stack `num_layers` layers to form an encoder
		decoder = nn.TransformerEncoder(decoder_layer, num_layers)
		return decoder
	
	def fusion(self, encoder_feats, decoder_feats):
		element_wise_product = encoder_feats * decoder_feats
		element_wise_absolute = torch.abs(encoder_feats - decoder_feats)
		cls_feats = torch.cat([encoder_feats, decoder_feats, element_wise_product, element_wise_absolute], axis=-1)
		return cls_feats
	
	def _classify(self):
		classifier = nn.Sequential(
			nn.Linear(self.hidden_dim * self.num_layers * self.num_pkt_features * 2 * 4, self.num_classes),
			nn.SELU(),
			nn.Linear(self.num_classes, self.num_classes),
			nn.SELU()
		)
		return classifier
	
	def forward(self, x):
		self.seq_len = x.size(1)
		x = torch.squeeze(x)  # 输出两个结果，一个是AutoDecoder结构的重建结果，与输入进行loss计算，一个是分类后的结果与label进行loss计算
		x2 = self.embedding(x)
		_, encoder_feats = self.encoder_layer(x2)
		decoder_inputs = self._decoder_input(self.seq_len, encoder_feats)
		re, decoder_feats = self.decoder_layer(decoder_inputs)
		cls_feats = self.fusion(encoder_feats, decoder_feats)
		y = self.cls_layer(cls_feats)
		return re, y, x


class FsNetLoss(nn.Module):
	# 重新设计算法Loss
	def __init__(self, alpah):
		super(FsNetLoss, self).__init__()
		self.alpah = alpah
		self.CLS = torch.nn.CrossEntropyLoss()
		self.RE = torch.nn.CrossEntropyLoss()
	
	def forward(self, re, cls, label, inputs):
		cls_loss = self.CLS(cls, label)
		inputs = torch.squeeze(inputs)
		# print(inputs.size())
		re = re.transpose(1, 2)
		re_loss = self.RE(re, inputs)
		loss = cls_loss + self.alpah * re_loss
		return loss


if __name__ == "__main__":
    #模型测试
    test_input = torch.tensor([[1,2,3,4,5],[3,4,6,7,2]])
    test_out_put = torch.tensor(1)
    cls = TfNet(10,10,10,2,4)
    re,y = cls(test_input)
    print(re)
    print(y)


# if __name__ == "__main__":
#     #BIGRU 测试。
#     gru_model = BiGRU(10,20,2)
#     print(gru_model)
#
#     # data = DatasetFromCSV("data/10k_time.txt")
#     # train_loader = DataLoader(data,batch_size=2)
#     input = torch.

# if __name__ == "__main__":
# 	# Loss函数测试
# 	alpah = 1
# 	inputs = torch.tensor([[0, 2, 3], [3, 4, 2]])
# 	print(inputs)
# 	re = torch.tensor([[[1.9843, 1.3084, 0.3485, 0.4179, -1.1200],
# 	                    [0.1801, 0.7200, -0.3634, -1.1379, 2.7537],
# 	                    [-2.0973, -0.4166, 1.1812, -1.6127, 0.8896]],
# 	                   [[-1.6348, -0.6409, -0.0389, -0.5553, 1.1713],
# 	                    [0.0656, -0.6620, -0.5782, -0.3181, -0.8497],
# 	                    [-0.0292, -0.8111, 2.3627, -1.2034, -0.7236]]])
# if __name__ == "__main__":
# 	# Loss函数测试
# 	alpah = 1
# 	inputs = torch.tensor([[0, 2, 3], [3, 4, 2]])
# 	print(inputs)
# 	re = torch.tensor([[[1.9843, 1.3084, 0.3485, 0.4179, -1.1200],
# 	                    [0.1801, 0.7200, -0.3634, -1.1379, 2.7537],
# 	                    [-2.0973, -0.4166, 1.1812, -1.6127, 0.8896]],
# 	                   [[-1.6348, -0.6409, -0.0389, -0.5553, 1.1713],
# 	                    [0.0656, -0.6620, -0.5782, -0.3181, -0.8497],
# 	                    [-0.0292, -0.8111, 2.3627, -1.2034, -0.7236]]])