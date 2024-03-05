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


""" My Lift: the variant of Dice """


class Lift(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(Lift, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))
        output = p * X + self.alpha * (1 - p) * (-X)
        return output


""" My SimLift: the variant of Dice, remove BatchNorm1d """


class SimLift(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(SimLift, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        p = torch.sigmoid(X)
        output = p * X + self.alpha * (1 - p) * (-X)
        return output


""" My VODice: the variant of Dice, remove E(s) only """


# TODO 按照 Dice 源码进行修改，然后修改成只有方差的版本
class VODice(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(VODice, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        # mean = torch.mean(X, dim=0, keepdim=True)
        std = torch.std(X, dim=0, keepdim=True) + self.eps
        X_normd = X / std
        p = torch.sigmoid(X_normd)
        return p * X + self.alpha * (1 - p) * X


""" x * tanh(x) after normalize """


class XTan(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(XTan, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        # self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        return X * torch.tanh(self.bn(X))


""" x * x * tanh(x) after normalize """


class XXTan(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(XXTan, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        # self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        return X * X * torch.tanh(self.bn(X))


""" bx * tanh(x) after normalize """


class BXTan(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(BXTan, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        # self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        return self.bn(X) * torch.tanh(X)


""" bxbx * tanh(x) after normalize """


class BXXTan(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(BXXTan, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        # self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        return self.bn(X) * X * torch.tanh(X)


""" x * tanh(x) """


class SimXTan(nn.Module):
    def __init__(self):
        super(SimXTan, self).__init__()

    def forward(self, X):
        return X * torch.tanh(X)


""" parameter every x * tanh(x) after normalize """


class PXTan(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(PXTan, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, X):
        return self.alpha * X * torch.tanh(self.bn(X))


""" [结果和 PXTan 一样] parameter every x * tanh(x) """
# class SimPXTan(nn.Module):
#     def __init__(self, input_dim):
#         super(SimPXTan, self).__init__()
#         self.alpha = nn.Parameter(torch.zeros(input_dim))
#
#     def forward(self, X):
#         return self.alpha * X * torch.tanh(X)

""" Tanh(x) * Softplus(x) """


class TanHS(nn.Module):
    def __init__(self):
        super(TanHS, self).__init__()

    def forward(self, X):
        return torch.tanh(X) * F.softplus(X)


""" mish: adaptive temperature, when temperature close to 0.01, it looks like ReLU """


class PMish(nn.Module):
    def __init__(self, input_dim, param_type="correct"):
        super(PMish, self).__init__()
        if param_type == "correct":
            # init = 0.9
            init = 0.1
        elif param_type == "stabilize":
            init = 0.5
        elif param_type == "accelerate":
            # init = 0.1
            init = 0.9
        else:
            raise ValueError("Invalid param_type. Choose from 'correct', 'stable', or 'accelerate'.")

        # self.temp = nn.Parameter(torch.empty(input_dim).fill_(init))
        self.temp = init

    def forward(self, X, min_temp=0.1, max_temp=1.0):
        # temp = torch.clamp(self.temp, min=min_temp, max=max_temp)
        # return X * torch.tanh(F.softplus(X / self.temp))
        return X * torch.tanh(F.softplus(X / self.temp))
        # return X * torch.tanh(F.softplus(X)) # [已测试，可复现] 测试 Mish 是否能复现


""" bmish: adaptive temperature, when temperature close to 0.01, it looks like ReLU """


class BMish(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(BMish, self).__init__()
        self.temp = nn.Parameter(torch.ones(input_dim))
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)

    def forward(self, X, min_temp=0.01, max_temp=1.0):
        temp = torch.clamp(self.temp, min=min_temp, max=max_temp)
        # return X * torch.tanh(F.softplus(X/temp))
        return self.bn(X) * torch.tanh(F.softplus(X / temp))  # DCN:81.4618, DNN:81.4542, PNN:81.3512， DeepFM:81.4369
        # return X * torch.tanh(F.softplus(self.bn(X)/temp)) # DCN:81.3327, DNN:81.3924, PNN:81.2926, DeepFM: 没跑
        # X_norm = self.bn(X)
        # return X_norm * torch.tanh(F.softplus(X_norm / temp)) # DCN: 81.4264, DNN: 81.3992, PNN: 81.3762, DeepFM: 没跑


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


""" pbmish: adaptive temperature bmish for HAF-DNN """
class PBMish(nn.Module):
    def __init__(self, input_dim, param_type="correct", eps=1e-9, min_temp=0.1, max_temp=2.0):
        super(PBMish, self).__init__()
        if param_type == "correct":
            init = 1.0
        elif param_type == "stabilize":
            init = 1.0
        elif param_type == "accelerate":
            init = 1.0
        else:
            raise ValueError("Invalid param_type. Choose from 'correct', 'stable', or 'accelerate'.")

        self.temp = nn.Parameter(torch.empty(input_dim).fill_(init))  # 可学习
        # self.temp = torch.empty(input_dim).fill_(init) # 不可学习, for Avazu
        self.alpha = nn.Parameter(torch.zeros(input_dim))  # 可学习, for Avazu
        # self.alpha = nn.Parameter(torch.ones(input_dim)) # 可学习, for Criteo
        # self.alpha = nn.Parameter(torch.empty(input_dim).fill_(0.5))  # 可学习, 初始化成 0.5

        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.min_temp = min_temp
        self.max_temp = max_temp

    def forward(self, X):
        p = torch.sigmoid(self.bn(X))  # 加通道的前置组件
        # self.temp.data.clamp_(min=self.min_temp, max=self.max_temp) # 限制了上下限

        # return X * torch.tanh(F.softplus(X / self.temp)) # 不加通道, 不加 self.bn(X)
        # return self.bn(X) * torch.tanh(F.softplus(X / self.temp)) # 不加通道, 加 self.bn(X)
        # return p * X + (1-p) * X * torch.tanh(F.softplus(X / self.temp)) # 加通道, 不加 alpha, 不加 self.bn(X)
        # return p * X + (1-p) * self.bn(X) * torch.tanh(F.softplus(X / self.temp)) # 加通道, 不加 alpha, 加 self.bn(X)
        # return p * X + self.alpha * (1-p) * X * torch.tanh(F.softplus(X / self.temp)) # 加通道, 加 alpha, 不加 self.bn(X)

        # 加通道, 加 alpha, 加 self.bn(X)
        return p * X + self.alpha * (1 - p) * self.bn(X) * torch.tanh(F.softplus(X / self.temp))


""" AC_PNN """
# Criteo, alpha 初始化 0.5
#   加通道, 加 alpha, 加 self.bn(X): 1.0, 1.0, 1.0: 0.805782, when training of epoch 1-> logloss: 0.439583 - AUC: 0.813263

# Criteo, alpha 初始化 1
#   加通道, 加 alpha, 加 self.bn(X): 1.0, 1.0, 1.0: 0.804364, when training of epoch 1-> logloss: 0.437667 - AUC: 0.814381

""" AC_DCN """
# Criteo, alpha 初始化 0
#   加通道, 加 alpha, 加 self.bn(X): 1.7, 0.7, 0.1: 0.805544, when training of epoch 1-> logloss: 0.439413 - AUC: 0.812375
#   加通道, 加 alpha, 加 self.bn(X): 1.0, 1.0, 1.0:

# Criteo, alpha 初始化 0.5
#   加通道, 加 alpha, 加 self.bn(X): 1.0, 1.0, 1.0:

# Criteo, alpha 初始化 1
#   加通道, 加 alpha, 加 self.bn(X): 1.7, 0.7, 0.1: 0.804241, when training of epoch 1-> logloss: 0.438301 - AUC: 0.813863
#   加通道, 加 alpha, 加 self.bn(X): 1.0, 1.0, 1.0: 0.804298, when training of epoch 1-> logloss: 0.437738 - AUC: 0.814490

# Avazu, alpha 初始化 0
# 不可学习
#   不加通道, 加 self.bn(X): 1.0, 1.0, 1.0: , when training of epoch 1
# 可学习, self.alpha 初始化为 0, 限制了上下限
#   不加通道, 加 self.bn(X): 1.0, 1.0, 1.0: 0.792323, when training of epoch 1
#   不加通道, 加 self.bn(X): 1.5, 1.0, 0.5: 0.792495, when training of epoch 1
#   加通道, 不加 alpha, 不加 self.bn(X): 1.0, 1.0, 1.0: 0.787962, when training of epoch 1
#   加通道, 不加 alpha, 加 self.bn(X): 1.0, 1.0, 1.0: 0.792104, when training of epoch 1
#   加通道, 加 alpha, 不加 self.bn(X): 1.0, 1.0, 1.0: 0.793338, when training of epoch 1
#   加通道, 加 alpha, 不加 self.bn(X): 1.8, 1.0, 0.2: 0.793350, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.0, 1.0, 1.0: 0.794190, when training of epoch 1 -> 0.794420
#   加通道, 加 alpha, 加 self.bn(X): 1.1, 1.0, 0.9: 0.794188, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.2, 1.0, 0.8: 0.794193, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.3, 1.0, 0.7: 0.794203, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.4, 1.0, 0.6: 0.794208, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.5, 1.0, 0.5: 0.794216, when training of epoch 1 -> 0.794436
#   加通道, 加 alpha, 加 self.bn(X): 1.6, 1.0, 0.4: 0.794228, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.7, 0.7, 0.1: 0.794401, when training of epoch 1 -> logloss: 0.371164 - AUC: 0.794599
#   加通道, 加 alpha, 加 self.bn(X): 1.7, 0.8, 0.1: 0.794402, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.7, 0.9, 0.1: 0.794399, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.7, 1.0, 0.1: 0.794397, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.7, 1.4, 0.1: 0.794397, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.9, 1.0, 0.1: 0.794396, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 1.9, 1.0, 0.1: 0.794395, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 2.0, 1.0, 0.1: 0.794395, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 2.0, 1.0, 0.05: 0.794395, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 2.0, 1.0, 0.01: 0.794395, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 3.0, 1.0, 0.1: 0.794395, when training of epoch 1
#   加通道, 加 alpha, 加 self.bn(X): 0.1, 0.7, 1.7: 0.794277, when training of epoch 1

# class PBMish(nn.Module):
#     def __init__(self, input_dim, param_type="correct", eps=1e-9, min_temp=0.1, max_temp=2.0):
#         super(PBMish, self).__init__()
#         if param_type == "correct":
#             init = 1.5
#         elif param_type == "stabilize":
#             init = 1.0
#         elif param_type == "accelerate":
#             init = 0.5
#         else:
#             raise ValueError("Invalid param_type. Choose from 'correct', 'stable', or 'accelerate'.")
#
#         self.temp = nn.Parameter(torch.empty(input_dim).fill_(init))
#         self.alpha = nn.Parameter(torch.zeros(input_dim))
#         self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
#         self.min_temp = min_temp
#         self.max_temp = max_temp
#
#     def forward(self, X):
#         p = torch.sigmoid(self.bn(X))
#         self.temp.data.clamp_(min=self.min_temp, max=self.max_temp)
#         return p * X + self.alpha * (1-p) * self.bn(X) * torch.tanh(F.softplus(X / self.temp))

""" The imitation of VODice """


class VOMish(nn.Module):
    def __init__(self, input_dim, eps=1e-9):
        super(VOMish, self).__init__()
        self.temp = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, X, min_temp=0.01, max_temp=1.0):
        temp = torch.clamp(self.temp, min=min_temp, max=max_temp)
        std = torch.std(X, dim=0, keepdim=True) + self.eps
        X_normd = X / std
        return X * torch.tanh(F.softplus(X_normd / temp))


""" MiLU: the part of Mish multiple the part of gelu """


class MiLU(nn.Module):
    def __init__(self):
        super(MiLU, self).__init__()

    def forward(self, X):
        return 0.6 * X * torch.tanh(F.softplus(X)) \
               * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / math.pi)) * (X + 0.044715 * torch.pow(X, 3))))


""" My IdLU: the variant of Leaky_relu, using fixed value to replace positive/negtive slope """


class IdLU(nn.Module):
    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, positive_slope=1.) -> None:
        super(IdLU, self).__init__()
        self.negative_slope = negative_slope
        self.positive_slope = positive_slope

    def forward(self, X):
        # return self.positive_slope * torch.max(x, torch.zeros_like(x)) \
        #        + self.negative_slope * torch.min(x, torch.zeros_like(x))
        # -->(LReLU) 1. LogLoss: 0.372401 AUC: 79.2213

        # return torch.where(X >= 0, F.leaky_relu(X, self.negative_slope), F.leaky_relu(X, self.negative_slope))
        # -->(LReLU) 1. 效果好像也很差

        # return torch.where(X >= 0, X, X/100)
        # -->(LReLU) 1. LogLoss: 0.372366 AUC: 79.2259

        # return torch.where(X >= 0, self.alpha * X, self.negative_slope * self.alpha * X)
        # -->(LReLU) 1. logloss: 0.376578 - AUC: 0.785641

        return torch.where(X >= 0, self.positive_slope * X, self.negative_slope * X)
        # -->(LReLU) 1. LogLoss: 0.372366 AUC: 79.2259

        # return torch.where(X >= 0, X, F.leaky_relu(X, self.negative_slope))
        # -->(LReLU) 1. LogLoss: 0.372431 AUC: 79.2129

        # return torch.where(X >= 0, self.positive_slope * X, F.leaky_relu(X, self.negative_slope))
        # -->(LReLU) 1. LogLoss: 0.372366 AUC: 79.2259 2.logloss: 0.372366 - AUC: 0.792259

        # return X * (X >= 0).float() + self.negative_slope * X * (X < 0).float()
        # -->(LReLU) 1. LogLoss: 0.372431 AUC: 79.2129


class LSig(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self) -> None:
        super(LSig, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, 2 * (F.sigmoid(x) - 0.4), 0.01 * x + 0.2)
