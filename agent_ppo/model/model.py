#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CNN + MLP dual-branch Actor-Critic for Gorge Chase PPO.
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_ch, out_ch, kernel=3, stride=1, padding=1):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
    nn.init.orthogonal_(conv.weight.data)
    nn.init.zeros_(conv.bias.data)
    return conv


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn_mlp"
        self.device = device

        c = Config.MAP_CHANNELS
        v = Config.MAP_VIEW
        s = Config.SYM_FEATURE_LEN
        a = Config.ACTION_NUM
        vn = Config.VALUE_NUM

        self._map_shape = (c, v, v)

        self.cnn = nn.Sequential(
            make_conv_layer(c, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            make_conv_layer(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            make_conv_layer(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            make_fc_layer(s, 128),
            nn.ReLU(inplace=True),
            make_fc_layer(128, 64),
            nn.ReLU(inplace=True),
        )

        fused_dim = 32 + 64
        self.fusion = nn.Sequential(
            make_fc_layer(fused_dim, 128),
            nn.ReLU(inplace=True),
            make_fc_layer(128, 64),
            nn.ReLU(inplace=True),
        )

        self.actor_head = make_fc_layer(64, a)
        self.critic_head = make_fc_layer(64, vn)

    def _reshape_map(self, map_tensor):
        if map_tensor.dim() == 2:
            return map_tensor.view(-1, *self._map_shape)
        return map_tensor

    def forward(self, map_tensor, sym_feat, inference=False):
        m = self._reshape_map(map_tensor)
        cnn_feat = self.cnn(m)
        mlp_feat = self.mlp(sym_feat)
        fused = torch.cat([cnn_feat, mlp_feat], dim=1)
        h = self.fusion(fused)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
