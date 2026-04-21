#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions, GAE computation for Gorge Chase PPO.
峡谷追猎 PPO 数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config


# ObsData: flattened map+scalar feature, legal_action mask / 展平特征与合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: action, d_action(greedy), prob, value / 动作、贪心动作、概率、价值
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: single-frame sample with int dims / 单帧样本（整数表示维度）
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    for i in range(len(list_sample_data) - 1):
        done_flag = float(np.asarray(list_sample_data[i].done, dtype=np.float32).reshape(-1)[0])
        if done_flag > 0.5:
            list_sample_data[i].next_value = np.zeros_like(list_sample_data[i].value, dtype=np.float32)
        else:
            list_sample_data[i].next_value = np.asarray(list_sample_data[i + 1].value, dtype=np.float32)

    if list_sample_data:
        list_sample_data[-1].next_value = np.zeros_like(np.asarray(list_sample_data[-1].value, dtype=np.float32))

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    """
    gae = np.zeros((Config.VALUE_NUM,), dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        reward = np.asarray(sample.reward, dtype=np.float32)
        value = np.asarray(sample.value, dtype=np.float32)
        next_value = np.asarray(sample.next_value, dtype=np.float32)
        done = float(np.asarray(sample.done, dtype=np.float32).reshape(-1)[0])
        non_terminal = 1.0 - np.clip(done, 0.0, 1.0)

        delta = reward + gamma * next_value * non_terminal - value
        gae = delta + gamma * lamda * non_terminal * gae
        sample.advantage = gae
        sample.reward_sum = gae + value
