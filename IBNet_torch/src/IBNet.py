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
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossInteraction

class IBNet(BaseModel):
    def __init__(self, 
                 feature_map,
                 model_id="IBNet",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 gate_temp=0.1,
                 cl_temp=0.2,
                 cl_weight=0.01,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=1,
                 net_dropout=0,
                 ssl_mode=False,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(IBNet, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None,  # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) \
            if dnn_hidden_units else None  # in case of only crossing net used
        self.crossnet = CrossNet(input_dim, num_cross_layers, gate_temp)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit
        self.cl_weight = cl_weight
        self.cl_temp = cl_temp
        self.ssl_mode = ssl_mode

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, dynamic_emb_dim=True)  # Batch x Feature x Dimension
        flat_input = feature_emb.flatten(start_dim=1)  # Batch x Feature * Dimension

        cross_emb, cross_emb_list, cross_emb_mask_concat_list = self.crossnet(flat_input)

        if self.dnn is not None:
            dnn_out = self.dnn(flat_input)
            final_out = torch.cat([cross_emb_mask_concat_list[-1], dnn_out], dim=-1)
        else:
            final_out = cross_emb
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred,
                       "cross_emb_list": cross_emb_list,
                       "cross_emb_mask_concat_list": cross_emb_mask_concat_list}
        return return_dict

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        loss_main = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')

        cross_emb_list = return_dict["cross_emb_list"]
        cross_emb_mask_concat_list = return_dict["cross_emb_mask_concat_list"]
        cross_emb_all = torch.cat(cross_emb_list, dim=-1)
        cross_emb_mask_concat_all = torch.cat(cross_emb_mask_concat_list, dim=-1)
        loss_cl = self.InfoNCE(cross_emb_all=cross_emb_all,
                               cross_emb_mask_all=cross_emb_mask_concat_all,
                               temperature=self.cl_temp)

        if not self.ssl_mode:
            loss = loss_main
        else:
            loss = loss_main + self.cl_weight * loss_cl

        return loss

    def InfoNCE(self, cross_emb_all, cross_emb_mask_all, temperature):
        cross_emb_all = torch.nn.functional.normalize(cross_emb_all)
        cross_emb_mask_all = torch.nn.functional.normalize(cross_emb_mask_all)

        pos_score = torch.exp(torch.tensor(1.0) / temperature)

        all_score = torch.matmul(cross_emb_all, cross_emb_mask_all.transpose(0, 1))
        all_score = torch.exp(all_score / temperature).sum(dim=1)

        loss = - torch.log(pos_score / all_score + 10e-6)
        return torch.mean(loss)


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers, gate_temp):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteraction(input_dim)
                                       for _ in range(self.num_layers))
        self.mask = Masker(cross_dim=input_dim, gate_temp=gate_temp)
        self.fc_compress = nn.Linear(input_dim * 2, input_dim)

    def forward(self, X_0):
        X_i = X_0  # b x dim
        cross_emb_list = []
        cross_emb_mask_concat_list = []
        # cross_emb_mask_list = []

        # input_mask = self.mask(X_0)

        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)  # 1. get cross_emb
            X_i_mask = self.mask(X_i)  # 2.reparametriztion & aggregate

            X_i_compress = torch.cat([X_i, X_i_mask], dim=-1)
            X_i_compress = self.fc_compress(X_i_compress)

            cross_emb_list.append(X_i)
            cross_emb_mask_concat_list.append(X_i_compress)
            # cross_emb_mask_list.append(self.mask(X_i))

        return X_i, cross_emb_list, cross_emb_mask_concat_list

class Masker(nn.Module):
    def __init__(self, cross_dim, gate_temp):
        super(Masker, self).__init__()
        self.linear = nn.Linear(cross_dim, cross_dim)
        self.gate_temp = gate_temp

    def forward(self, cross_emb):
        eps = torch.rand(cross_emb.size()[-1])
        gate_inputs = (torch.log(eps) - torch.log(1 - eps)).to(cross_emb.device)
        gate_inputs = self.linear(cross_emb) + gate_inputs
        gate_inputs = gate_inputs / self.gate_temp
        mask = torch.sigmoid(gate_inputs)
        cross_emb_mask = mask * cross_emb + (1 - mask) * torch.mean(cross_emb)
        return cross_emb_mask

