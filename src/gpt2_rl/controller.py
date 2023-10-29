

''' controller '''

import collections
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, hidden_size,
                 num_hidden_layers,
                 dropout_ratio=0.1,
                 mode="train",
                 tanh_c=2.5):
        super(Controller, self).__init__()

        self.mode = mode
        self.tanh_c = tanh_c

        # 目前controller
        #   - 以instruction/prompt的最后一个token的表征作为输入
        #   - controller网络结构为linear + activation + linear + activation + linear(dim num_hidden_layers)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(p=dropout_ratio),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Dropout(p=dropout_ratio),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_hidden_layers * 2)
        )

    def forward(self, input_tensor, ):
        logits = self.net(input_tensor)

        if self.mode == 'train':
            logits = (self.args.tanh_c * F.tanh(logits))

        return logits

    def sample(self, batch_size=1, ):