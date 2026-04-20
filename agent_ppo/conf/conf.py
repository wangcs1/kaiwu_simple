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
    VIEW_SIZE = 9
    VIEW_CHANNELS = 6  # passable, treasure, buff, monster, danger, visited
    SCALAR_FEATURE_DIM = 14

    # Feature dimensions / 特征维度
    FEATURES = [VIEW_CHANNELS * VIEW_SIZE * VIEW_SIZE, SCALAR_FEATURE_DIM]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间
    ACTION_NUM = 16  # 0-7 move, 8-15 flash

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
    REWARD_STEP_SURVIVE = 0.006
    REWARD_STEP_PENALTY = -0.002
    REWARD_TREASURE_PROGRESS_BASE = 0.03
    REWARD_TREASURE_PROGRESS_SAFE_GAIN = 0.10
    REWARD_TREASURE_NEAR_PROGRESS = 0.25
    REWARD_TREASURE_NEAR_BONUS = 0.06
    REWARD_TREASURE_PICKUP = 1.2
    REWARD_BUFF_PICKUP = 0.45
    REWARD_ESCAPE_PROGRESS_BASE = 0.03
    REWARD_ESCAPE_PROGRESS_DANGER_GAIN = 0.22
    REWARD_MONSTER_PROXIMITY_PENALTY = -0.12
    REWARD_EXPLORATION = 0.006
    REWARD_STAGNATION_PENALTY = -0.012
    REWARD_FLASH_ESCAPE = 0.18
    REWARD_FLASH_WASTE_SAFE = -0.025
    REWARD_STUCK_PENALTY = -0.03
    REWARD_REPEAT_ACTION_STUCK = -0.01

    REWARD_ENV_MIX = 0.2

    # Obstacle-aware action mask / 障碍物动作掩码
    ENABLE_OBSTACLE_ACTION_MASK = True
    # soft: attenuate blocked actions, hard: directly mask blocked actions
    OBSTACLE_MASK_MODE = "soft"
    OBSTACLE_MASK_STRENGTH = 0.65
    FLASH_OBSTACLE_MASK_SCALE = 0.6
    # Action direction offsets for actions 0..7 (dx, dz), configured for quick remap if env differs.
    ACTION_MOVE_DIRS = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ]

    DANGER_DISTANCE = 3.0
    TREASURE_NEAR_RADIUS = 3.0
    EXPLORATION_DECAY = 0.92
    STAGNATION_STEPS = 8
    STUCK_STEPS = 3
