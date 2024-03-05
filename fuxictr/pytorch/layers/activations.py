# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class Dice(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + self.alpha * (1 - p) * X
        return output


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


""" LeLeLU: Leaky Learnable ReLU """
class LeLeLU(nn.Module):
    __constants__ = ['inplace', 'init_parms']
    inplace: bool
    init_parms: float

    def __init__(self, init_parms: float = 1e-1, inplace: bool = False) -> None:
        super(LeLeLU, self).__init__()
        self.learnable_shared_params = nn.Parameter(torch.Tensor([init_parms]))
        self.inplace = inplace  # can optionally do the operation in-place. Default: False

    def forward(self, x):
        return self.learnable_shared_params * torch.where(x >= 0, x, 0.01 * x)


""" Swish """
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class PBMish_Avazu(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(PBMish_Avazu, self).__init__()
        self.temp = nn.Parameter(torch.empty(input_dim).fill_(1.0))
        self.alpha = nn.Parameter(torch.zeros(input_dim))
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        return p * X + self.alpha * (1 - p) * self.bn(X) * torch.tanh(F.softplus(X / self.temp))


class PBMish_Criteo(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(PBMish_Criteo, self).__init__()
        self.temp = nn.Parameter(torch.empty(input_dim).fill_(1.0))
        self.alpha = nn.Parameter(torch.ones(input_dim))
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        return p * X + self.alpha * (1 - p) * self.bn(X) * torch.tanh(F.softplus(X / self.temp))
