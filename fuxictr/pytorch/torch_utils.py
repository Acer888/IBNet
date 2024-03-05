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

import sys
import os
import numpy as np
import torch
from torch import nn
import random
from functools import partial
import h5py
import re



def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    return device


def get_optimizer(optimizer, params, lr, kwargs=None):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
        elif optimizer.lower() == "adamw":
            optimizer = "AdamW"
        # 自定义优化器
        if optimizer.upper() == "GRDA":
            from fuxictr.pytorch.layers.GRDA import GRDA
            optimizer = GRDA(params, c=kwargs["GRDA_c"], mu=kwargs["GRDA_mu"], lr=lr)
        else:
            try:
                optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
            except:
                raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer


def get_loss(loss):
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try:
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))
    return loss_fn


def get_regularizer(reg):
    reg_pair = []  # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair


def get_activation(activation, hidden_units=None, negative_slope=0.01, positive_slope=1.05):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice", "lift", "simlift", "vodice", "xtan", "pxtan", "simpxtan", "pmish", "pbmish"]:
            assert type(hidden_units) == int
        if activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "elu":
            return nn.ELU(alpha=1)
        elif activation.lower() == "hardshrink":
            return nn.Hardshrink(lambd=0.5)
        elif activation.lower() == "hardsigmoid":
            return nn.Hardsigmoid()
        elif activation.lower() == "hardtanh":
            return nn.Hardtanh(-2, 20) # default:(-2, 2)
        elif activation.lower() == "swish":
            from fuxictr.pytorch.layers.activations import Swish
            return Swish()
        elif activation.lower() == "hardswish":
            return nn.Hardswish()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        # elif activation.lower() == "lrelu":
        #     return nn.LeakyReLU(negative_slope=0.5)
        elif activation.lower() == "logsigmoid":
            return nn.LogSigmoid()
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.01)
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "relu6":
            return nn.ReLU6()
        elif activation.lower() == "rrelu":
            return nn.RReLU(0.1, 0.3)
        elif activation.lower() == "selu":
            return nn.SELU()
        elif activation.lower() == "celu":
            return nn.CELU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "silu":
            return nn.SiLU()
        elif activation.lower() == "mish":
            return nn.Mish()
        elif activation.lower() == "softplus":
            return nn.Softplus(beta=1, threshold=20)
        elif activation.lower() == "softshrink":
            return nn.Softshrink(lambd=0.5)
        elif activation.lower() == "softsign":
            return nn.Softsign()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "tanhshrink":
            return nn.Tanhshrink()
        elif activation.lower() == "threshold":
            return nn.Threshold(0.1, 20)
        elif activation.lower() == "glu":
            return nn.GLU()
        elif activation.lower() == "dice":
            from fuxictr.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        elif activation.lower() == "lelelu":
            from fuxictr.pytorch.layers.activations import LeLeLU
            return LeLeLU()
        elif activation.lower() == "lift":
            from fuxictr.pytorch.layers.activations import Lift
            return Lift(hidden_units)
        elif activation.lower() == "simlift":
            from fuxictr.pytorch.layers.activations import SimLift
            return SimLift(hidden_units)
        elif activation.lower() == "vodice":
            from fuxictr.pytorch.layers.activations import VODice
            return VODice(hidden_units)
        elif activation.lower() == "xtan":
            from fuxictr.pytorch.layers.activations import XTan
            return XTan(hidden_units)
        elif activation.lower() == "xxtan":
            from fuxictr.pytorch.layers.activations import XXTan
            return XXTan(hidden_units)
        elif activation.lower() == "bxtan":
            from fuxictr.pytorch.layers.activations import BXTan
            return BXTan(hidden_units)
        elif activation.lower() == "bxxtan":
            from fuxictr.pytorch.layers.activations import BXXTan
            return BXXTan(hidden_units)
        elif activation.lower() == "simxtan":
            from fuxictr.pytorch.layers.activations import SimXTan
            return SimXTan()
        elif activation.lower() == "pxtan":
            from fuxictr.pytorch.layers.activations import PXTan
            return PXTan(hidden_units)
        # elif activation.lower() == "simpxtan":
        #     from fuxictr.pytorch.layers.activations import SimPXTan
        #     return SimPXTan(hidden_units)
        elif activation.lower() == "tanhs":
            from fuxictr.pytorch.layers.activations import TanHS
            return TanHS()
        # elif activation.lower() == "pmish_correct":
        #     from fuxictr.pytorch.layers.activations import PMish
        #     return PMish(hidden_units, param_type="correct")
        # elif activation.lower() == "pmish_stabilize":
        #     from fuxictr.pytorch.layers.activations import PMish
        #     return PMish(hidden_units, param_type="stabilize")
        # elif activation.lower() == "pmish_accelerate":
        #     from fuxictr.pytorch.layers.activations import PMish
        #     return PMish(hidden_units, param_type="accelerate")
        elif activation.lower() == "bmish":
            from fuxictr.pytorch.layers.activations import BMish
            return BMish(hidden_units)
        elif activation.lower() == "pbmish_avazu":
            from fuxictr.pytorch.layers.activations import PBMish_Avazu
            return PBMish_Avazu(hidden_units)
        elif activation.lower() == "pbmish_criteo":
            from fuxictr.pytorch.layers.activations import PBMish_Criteo
            return PBMish_Criteo(hidden_units)
        elif activation.lower() == "pbmish_correct":
            from fuxictr.pytorch.layers.activations import PBMish
            return PBMish(hidden_units, "correct")
        elif activation.lower() == "pbmish_stabilize":
            from fuxictr.pytorch.layers.activations import PBMish
            return PBMish(hidden_units, "stabilize")
        elif activation.lower() == "pbmish_accelerate":
            from fuxictr.pytorch.layers.activations import PBMish
            return PBMish(hidden_units, "accelerate")
        elif activation.lower() == "vomish":
            from fuxictr.pytorch.layers.activations import VOMish
            return VOMish(hidden_units)
        elif activation.lower() == "milu":
            from fuxictr.pytorch.layers.activations import MiLU
            return MiLU()
        # elif activation.lower() == "idlu":
        #     from fuxictr.pytorch.layers.activations import IdLU
        #     return IdLU(negative_slope=negative_slope, positive_slope=positive_slope)
        # TODO 添加新的激活函数
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation


def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported." \
                             .format(initializer))
    return initializer


def save_init_embs(model, data_path="init_embs.h5"):
    emb_dict = dict()
    for k, v in model.state_dict().items():
        if "embedding_layers" in k:
            if v.size(-1) > 1:
                f_name = re.findall(r"embedding_layers.(.*).weight", k)[0]
                emb_dict[f_name] = v.cpu().numpy()
    with h5py.File(data_path, 'w') as hf:
        for key, arr in emb_dict.items():
            hf.create_dataset(key, data=arr)


def load_init_embs(model, data_path="init_embs.h5"):
    state_dict = model.state_dict()
    f_name_dict = dict()
    for k in state_dict.keys():
        if "embedding_layers" in k and state_dict[k].size(-1) > 1:
            f_name = re.findall(r"embedding_layers.(.*).weight", k)[0]
            f_name_dict[f_name] = k
    with h5py.File(data_path, 'r') as hf:
        for key in hf.keys():
            if key in f_name_dict:
                state_dict[f_name_dict[key]] = torch.from_numpy(hf[key][:])
    model.load_state_dict(state_dict)
