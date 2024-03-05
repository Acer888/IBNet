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
from fuxictr.pytorch.torch_utils import get_activation


class MLP_Block(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False,
                 layer_norm=False,
                 norm_before_activation=True,
                 use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)

        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
                elif layer_norm:
                    dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if not norm_before_activation:
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
                elif layer_norm:
                    dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers)  # * used to unpack list

    def forward(self, inputs):
        return self.mlp(inputs)


class FinalBlock(nn.Module):
    def __init__(self, input_dim, hidden_units=[], hidden_activations=None,
                 dropout_rates=[], batch_norm=True, residual_type="sum"):
        # Replacement of MLP_Block, identical when order=1
        super(FinalBlock, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(FinalLinear(hidden_units[idx], hidden_units[idx + 1],
                                          residual_type=residual_type))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i


class FinalLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, residual_type="sum"):
        """ A replacement of nn.Linear to enhance multiplicative feature interactions.
            `residual_type="concat"` uses the same number of parameters as nn.Linear
            while `residual_type="sum"` doubles the number of parameters.
        """
        super(FinalLinear, self).__init__()
        self.residual_type = residual_type
        if residual_type == "sum":
            output_dim = output_dim * 2
        else:
            assert output_dim % 2 == 0, "output_dim should be divisible by 2."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        if self.residual_type == "concat":
            h = torch.cat([h2, h1 * h2], dim=-1)
        elif self.residual_type == "sum":
            h = h2 + h1 * h2
        return h

