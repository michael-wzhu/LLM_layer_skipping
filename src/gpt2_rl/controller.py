

''' controller '''

import collections
import os

import torch
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, args):
        super(Controller, self).__init__()

        
