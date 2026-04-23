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
    # Training profile / 训练配置档位
    SIMPLE_TRAINING_MODE = True

    # Observation setting / 观测设置
    VIEW_SIZE = 9
    VIEW_CHANNELS = 6  # passable, treasure, buff, monster, danger, visited
    SCALAR_FEATURE_DIM = 24
    HERO_FEATURE_DIM = 5
    TREASURE_FEATURE_DIM = 6
    MONSTER_FEATURE_DIM = 5
    MOBILITY_FEATURE_DIM = 4
    STAGE_FEATURE_DIM = 4

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
    PPO_EPOCHS = 4
    PPO_MINIBATCH_SIZE = 128
    BETA_END = 0.0002
    BETA_DECAY_STEPS = 200000

    # Reward shaping / 奖励塑形
    REWARD_STEP_SURVIVE = 0.006
    REWARD_STEP_PENALTY = -0.002
    REWARD_TREASURE_PROGRESS_BASE = 0.06
    REWARD_TREASURE_PROGRESS_SAFE_GAIN = 0.18
    REWARD_TREASURE_NEAR_PROGRESS = 0.40
    REWARD_TREASURE_NEAR_BONUS = 0.16
    REWARD_TREASURE_PICKUP = 2.0
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

    # Final reward = (1 - REWARD_ENV_MIX) * shaped + REWARD_ENV_MIX * env_reward
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

    # Monster acceleration phase / 怪物加速阶段
    MONSTER_ACCEL_STEP = 500
    ESCAPE_WEIGHT_AFTER_ACCEL = 1.25
    MONSTER_PENALTY_AFTER_ACCEL = 1.20

    # Treasure top-k tracking / 宝箱 top-k 追踪
    TREASURE_TARGET_TOPK = 1
    TREASURE_MEMORY_TTL = 12
    TREASURE_NEW_SEEN_COUNT = 1
    TREASURE_RANK_NEW_BONUS = 1.2
    TREASURE_RANK_DIST_WEIGHT = 0.10
    TREASURE_RANK_RISK_WEIGHT = 0.14
    TREASURE_RANK_RECENCY_WEIGHT = 0.04
    TREASURE_RANK_VALUE_WEIGHT = 0.02

    # Direct target-seeking shaping / 目标直追奖励
    REWARD_TARGET_TRACK_PROGRESS = 0.22
    REWARD_TARGET_TRACK_AWAY_PENALTY = -0.08
    REWARD_TARGET_LOCK_BONUS = 0.04
    REWARD_SIMPLE_TARGET_PROGRESS = 0.30
    REWARD_SIMPLE_TARGET_AWAY_PENALTY = -0.12
    REWARD_SIMPLE_DANGER_PENALTY = -0.10
    REWARD_SIMPLE_ESCAPE_PROGRESS = 0.06
    REWARD_SIMPLE_STAGNATION = -0.02

    # Treasure direction action bias / 朝宝箱方向动作偏置
    ENABLE_TREASURE_DIRECTION_BIAS = True
    TREASURE_DIRECTION_BIAS_STRENGTH = 0.35
    TREASURE_DIRECTION_BIAS_DANGER_GATE = 0.72
