#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/24 11:11 PM
# @Author  : Gear

import numpy as np

class FlowDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, num_pkt_features, file_path):
        super(FlowDataset, self).__init__()
        self.num_pkt_features = num_pkt_features
        self.data = pd.read_csv(file_path, sep='\t', header=None)

        # find the maximum value of each feature dimension
        arr = np.arange(len(self.data.columns)) % num_pkt_features
        arr[-1] = -1
        self.sizes = [
            self.data.iloc[:, arr == i].to_numpy().max() + 1
            for i in range(num_pkt_features)
        ]

        self.labels = np.asarray(self.data.iloc[:, -1])
        self.categories = list(set(self.labels))
        self.num_classes = len(self.categories)
        self.seq_len = (self.data.shape[1] - 1) // num_pkt_features
        print("#examples:", self.data.shape[0], "#classes:", self.num_classes)

    def __getitem__(self, index):
        label = self.labels[index]
        label_id = self.categories.index(label)
        # label_vec = torch.zeros(self.num_classes, dtype=int)
        # label_vec[label_id] = 1
        vec = np.asarray(self.data.iloc[index][:-1])
        # shape (num_pkt_features, seq_len)
        vec = torch.from_numpy(vec).reshape(-1, self.num_pkt_features)
        return vec, label_id

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = FlowDataset(2, "../data/test.dat")
    loader = DataLoader(dataset, batch_size=2)
    for X, Y in loader:
        print(X.shape)
        break