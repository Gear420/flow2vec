#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 11:07 PM
# @Author  : Gear


#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self,
                 embedding_sizes,
                 num_classes,
                 embedding_dim=128,
                 num_layers=1,
                 num_heads=4):
        '''`embedding_sizes` is a list specifying the size of each embedding space.
        '''
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # embedding look-up tables with sizes given by `embedding_sizes`
        self.embeddings = [
            nn.Embedding(size, embedding_dim) for size in embedding_sizes
        ]
        # a layer of transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads)
        # stack `num_layers` layers to form an encoder
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # a FC as the final classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        ''' Shape of x: (batch_size, seq_len, dim). '''
        # Calculate the embedding of each dim, and sum them together as the
        # final embedding, which has shape (batch_size, seq_len).
        e = sum([
            embedding(x[:, :, i]) for i, embedding in enumerate(self.embeddings)
        ])
        x = self.encoder(e)
        x = torch.mean(x, 1)
        self.feature_map = x.detach()
        return self.classifier(x)
    
