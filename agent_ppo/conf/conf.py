#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:
    # Observation setting / 观测设置
    VIEW_SIZE = 21
    VIEW_CHANNELS = 4  # obstacle, treasure, monster, self
    SCALAR_FEATURE_DIM = 8

    # Feature dimensions / 特征维度
    FEATURES = [VIEW_CHANNELS * VIEW_SIZE * VIEW_SIZE, SCALAR_FEATURE_DIM]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间
    ACTION_NUM = 8

    # Value head / 价值头
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # Reward shaping / 奖励塑形
    REWARD_STEP_PENALTY = -0.002
    REWARD_TREASURE_PROGRESS = 0.06
    REWARD_TREASURE_PICKUP = 0.8
    REWARD_ESCAPE_PROGRESS = 0.08
    REWARD_ESCAPE_DANGER_PENALTY = -0.04
    DANGER_DISTANCE = 3.0

    # Extra light-weight rewards / 额外轻量奖励
    REWARD_SAFE_BONUS = 0.003
    REWARD_DANGER_ESCAPE_BONUS = 0.05
    REWARD_STAGNATION_PENALTY = -0.01
    STAGNATION_STEPS = 6
