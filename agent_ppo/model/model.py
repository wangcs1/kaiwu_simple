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
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """CNN + self-attention backbone + Actor/Critic dual heads.

    CNN + 自注意力骨干 + Actor/Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn_attn"
        self.device = device

        token_dim = 64
        scalar_dim = Config.SCALAR_FEATURE_DIM
        trunk_dim = 128
        action_num = Config.ACTION_NUM
        value_num = Config.VALUE_NUM

        # Visual CNN encoder / 视觉 CNN 编码器
        self.cnn = nn.Sequential(
            nn.Conv2d(Config.VIEW_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, token_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Token projection + self-attention / token 投影 + 自注意力
        self.token_proj = make_fc_layer(token_dim, token_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(token_dim)

        # Shared MLP after attention / 注意力后的共享 MLP
        self.shared_mlp = nn.Sequential(
            make_fc_layer(token_dim + scalar_dim, trunk_dim),
            nn.ReLU(),
            make_fc_layer(trunk_dim, trunk_dim),
            nn.ReLU(),
        )

        # Actor head / 策略头
        self.actor_head = make_fc_layer(trunk_dim, action_num)

        # Critic head / 价值头
        self.critic_head = make_fc_layer(trunk_dim, value_num)

    def forward(self, obs, inference=False):
        batch_size = obs.shape[0]
        view_flat_dim = Config.VIEW_CHANNELS * Config.VIEW_SIZE * Config.VIEW_SIZE

        view_flat = obs[:, :view_flat_dim]
        scalar = obs[:, view_flat_dim:]

        view = view_flat.view(batch_size, Config.VIEW_CHANNELS, Config.VIEW_SIZE, Config.VIEW_SIZE)
        feat = self.cnn(view)
        token = feat.flatten(2).transpose(1, 2)
        token = self.token_proj(token)

        attn_out, _ = self.self_attn(token, token, token, need_weights=False)
        token = self.attn_norm(token + attn_out)
        pooled = token.mean(dim=1)

        hidden = self.shared_mlp(torch.cat([pooled, scalar], dim=1))
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
