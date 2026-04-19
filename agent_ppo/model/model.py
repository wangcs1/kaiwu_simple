#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(x + identity)


class SEGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class Model(nn.Module):
    """Compact residual CNN + scalar fusion + dual-value critic.

    轻量残差 CNN + 标量融合 + 双价值分支 Critic。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_resnet_fusion"
        self.device = device

        scalar_dim = Config.SCALAR_FEATURE_DIM
        trunk_dim = 192

        self.map_encoder = nn.Sequential(
            ConvBlock(Config.VIEW_CHANNELS, 32),
            ConvBlock(32, 64),
            SEGate(64),
            ConvBlock(64, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.scalar_encoder = nn.Sequential(
            make_fc_layer(scalar_dim, 64),
            nn.SiLU(),
            make_fc_layer(64, 64),
            nn.SiLU(),
        )

        self.shared = nn.Sequential(
            make_fc_layer(64 + 64, trunk_dim),
            nn.SiLU(),
            make_fc_layer(trunk_dim, trunk_dim),
            nn.SiLU(),
        )

        self.actor_head = make_fc_layer(trunk_dim, Config.ACTION_NUM)
        self.value_survival = make_fc_layer(trunk_dim, Config.VALUE_NUM)
        self.value_treasure = make_fc_layer(trunk_dim, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        batch_size = obs.shape[0]
        view_flat_dim = Config.VIEW_CHANNELS * Config.VIEW_SIZE * Config.VIEW_SIZE

        view_flat = obs[:, :view_flat_dim]
        scalar = obs[:, view_flat_dim:]

        view = view_flat.view(batch_size, Config.VIEW_CHANNELS, Config.VIEW_SIZE, Config.VIEW_SIZE)
        map_feat = self.map_encoder(view).flatten(1)
        scalar_feat = self.scalar_encoder(scalar)

        trunk = self.shared(torch.cat([map_feat, scalar_feat], dim=1))
        logits = self.actor_head(trunk)
        value = 0.6 * self.value_survival(trunk) + 0.4 * self.value_treasure(trunk)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
