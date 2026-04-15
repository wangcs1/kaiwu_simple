#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

from agent_ppo.conf.conf import Config


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.prev_treasure_dist = None
        self.prev_monster_dist = None
        self.prev_score = 0.0
        self.stagnation_steps = 0

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        self.step_no += 1

        observation = env_obs.get("observation", env_obs)
        frame_state = observation.get("frame_state", {})
        legal_act_raw = observation.get("legal_action", None)

        obstacle_map = self._extract_view_layer(
            observation,
            frame_state,
            candidate_keys=["obstacle_map", "obstacle", "wall_map", "block_map", "obstacle_mask", "map_info"],
        )
        treasure_map = self._extract_view_layer(
            observation,
            frame_state,
            candidate_keys=["treasure_map", "treasure", "chest_map", "chest", "treasure_mask"],
        )
        monster_map = self._extract_view_layer(
            observation,
            frame_state,
            candidate_keys=["monster_map", "monster", "enemy_map", "enemy", "threat_map"],
        )
        self_map = self._extract_view_layer(
            observation,
            frame_state,
            candidate_keys=["self_map", "agent_map", "player_map", "hero_map"],
        )

        if float(self_map.sum()) <= 0:
            center = Config.VIEW_SIZE // 2
            self_map[center, center] = 1.0

        stacked_view = np.stack([obstacle_map, treasure_map, monster_map, self_map], axis=0)

        nearest_treasure_dist = self._nearest_distance_to_center(treasure_map)
        nearest_monster_dist = self._nearest_distance_to_center(monster_map)
        reward_value = self._shape_reward(observation, nearest_treasure_dist, nearest_monster_dist)

        scalar_feature = self._build_scalar_feature(
            last_action=last_action,
            nearest_treasure_dist=nearest_treasure_dist,
            nearest_monster_dist=nearest_monster_dist,
            treasure_map=treasure_map,
            monster_map=monster_map,
        )

        feature = np.concatenate([stacked_view.reshape(-1), scalar_feature], axis=0).astype(np.float32)
        legal_action = self._extract_legal_action(legal_act_raw)
        reward = [float(reward_value)]

        return feature, legal_action, reward

    def _shape_reward(self, observation, nearest_treasure_dist, nearest_monster_dist):
        reward = Config.REWARD_STEP_PENALTY

        if nearest_treasure_dist is not None and self.prev_treasure_dist is not None:
            reward += Config.REWARD_TREASURE_PROGRESS * (self.prev_treasure_dist - nearest_treasure_dist)

        if nearest_monster_dist is not None and self.prev_monster_dist is not None:
            reward += Config.REWARD_ESCAPE_PROGRESS * (nearest_monster_dist - self.prev_monster_dist)

        if nearest_monster_dist is not None and nearest_monster_dist <= Config.DANGER_DISTANCE:
            reward += Config.REWARD_ESCAPE_DANGER_PENALTY * (Config.DANGER_DISTANCE - nearest_monster_dist + 1.0)

        if nearest_monster_dist is not None and nearest_monster_dist > Config.DANGER_DISTANCE:
            reward += Config.REWARD_SAFE_BONUS

        if (
            self.prev_monster_dist is not None
            and self.prev_monster_dist <= Config.DANGER_DISTANCE
            and nearest_monster_dist is not None
            and nearest_monster_dist > Config.DANGER_DISTANCE
        ):
            reward += Config.REWARD_DANGER_ESCAPE_BONUS

        score = float(self._scalar_from_obs(observation, ["score", "total_score", "game_score"], default=self.prev_score))
        score_delta = max(0.0, score - self.prev_score)
        if score_delta > 0:
            reward += Config.REWARD_TREASURE_PICKUP * score_delta
        self.prev_score = score

        moved_on_target = False
        if nearest_treasure_dist is not None and self.prev_treasure_dist is not None:
            moved_on_target = abs(nearest_treasure_dist - self.prev_treasure_dist) > 1e-6
        elif nearest_monster_dist is not None and self.prev_monster_dist is not None:
            moved_on_target = abs(nearest_monster_dist - self.prev_monster_dist) > 1e-6

        if moved_on_target:
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
            if self.stagnation_steps >= Config.STAGNATION_STEPS:
                reward += Config.REWARD_STAGNATION_PENALTY

        self.prev_treasure_dist = nearest_treasure_dist
        self.prev_monster_dist = nearest_monster_dist
        return reward

    def _build_scalar_feature(
        self,
        last_action,
        nearest_treasure_dist,
        nearest_monster_dist,
        treasure_map,
        monster_map,
    ):
        max_view_dist = float(Config.VIEW_SIZE - 1) * 2.0
        has_treasure = 1.0 if float(treasure_map.sum()) > 0 else 0.0
        has_monster = 1.0 if float(monster_map.sum()) > 0 else 0.0
        last_action_norm = 0.0 if last_action < 0 else float(last_action) / float(max(1, Config.ACTION_NUM - 1))
        treasure_dist_norm = 1.0 if nearest_treasure_dist is None else min(1.0, nearest_treasure_dist / max_view_dist)
        monster_dist_norm = 1.0 if nearest_monster_dist is None else min(1.0, nearest_monster_dist / max_view_dist)

        scalar = np.array(
            [
                min(1.0, self.step_no / 1000.0),
                last_action_norm,
                treasure_dist_norm,
                monster_dist_norm,
                has_treasure,
                has_monster,
                float(self.prev_treasure_dist is not None),
                float(self.prev_monster_dist is not None),
            ],
            dtype=np.float32,
        )
        return scalar

    def _extract_legal_action(self, legal_act_raw):
        if legal_act_raw is None:
            return [1] * Config.ACTION_NUM

        arr = np.asarray(legal_act_raw, dtype=np.float32).reshape(-1)
        if arr.size >= Config.ACTION_NUM:
            if np.all((arr[: Config.ACTION_NUM] == 0) | (arr[: Config.ACTION_NUM] == 1)):
                mask = arr[: Config.ACTION_NUM]
            else:
                valid_set = {int(a) for a in arr if 0 <= int(a) < Config.ACTION_NUM}
                mask = np.array([1 if i in valid_set else 0 for i in range(Config.ACTION_NUM)], dtype=np.float32)
        else:
            return [1] * Config.ACTION_NUM

        if float(mask.sum()) <= 0:
            mask = np.ones(Config.ACTION_NUM, dtype=np.float32)
        return mask.tolist()

    def _extract_view_layer(self, observation, frame_state, candidate_keys):
        raw = self._find_first(observation, candidate_keys)
        if raw is None:
            raw = self._find_first(frame_state, candidate_keys)
        if raw is None:
            return np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)

        arr = self._to_2d_array(raw)
        if arr is None:
            return np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)

        arr = self._center_crop_or_pad(arr.astype(np.float32), Config.VIEW_SIZE)
        return np.where(arr > 0, 1.0, 0.0).astype(np.float32)

    def _to_2d_array(self, data):
        if isinstance(data, dict):
            nested = self._find_first(data, ["map", "grid", "data", "matrix", "mask", "view"])
            if nested is None:
                return None
            return self._to_2d_array(nested)

        arr = np.asarray(data)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            if arr.shape[0] <= 4:
                return np.max(arr, axis=0)
            return np.max(arr, axis=-1)
        if arr.ndim == 1:
            length = arr.shape[0]
            side = int(np.sqrt(length))
            if side * side == length:
                return arr.reshape(side, side)
        return None

    def _center_crop_or_pad(self, arr, target_size):
        h, w = arr.shape
        out = np.zeros((target_size, target_size), dtype=np.float32)

        src_top = max(0, (h - target_size) // 2)
        src_left = max(0, (w - target_size) // 2)
        src_bottom = min(h, src_top + target_size)
        src_right = min(w, src_left + target_size)

        cropped = arr[src_top:src_bottom, src_left:src_right]
        ch, cw = cropped.shape
        dst_top = max(0, (target_size - ch) // 2)
        dst_left = max(0, (target_size - cw) // 2)
        out[dst_top : dst_top + ch, dst_left : dst_left + cw] = cropped
        return out

    def _nearest_distance_to_center(self, layer):
        ys, xs = np.where(layer > 0)
        if ys.size == 0:
            return None
        center = Config.VIEW_SIZE // 2
        dist = np.abs(ys - center) + np.abs(xs - center)
        return float(np.min(dist))

    def _scalar_from_obs(self, env_obs, keys, default=0.0):
        value = self._find_first(env_obs, keys)
        if value is None:
            return default
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _find_first(self, obj, candidate_keys):
        if obj is None:
            return None

        keys = {str(k).lower() for k in candidate_keys}
        queue = [obj]
        max_nodes = 256
        visited = 0
        while queue and visited < max_nodes:
            cur = queue.pop(0)
            visited += 1
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if str(k).lower() in keys:
                        return v
                for v in cur.values():
                    if isinstance(v, (dict, list, tuple)):
                        queue.append(v)
            elif isinstance(cur, (list, tuple)):
                for v in cur:
                    if isinstance(v, (dict, list, tuple)):
                        queue.append(v)
        return None
