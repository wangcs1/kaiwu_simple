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
        self.prev_treasure_score = 0.0
        self.prev_treasure_count = 0
        self.prev_buff_count = 0
        self.stagnation_steps = 0
        self.visit_counts = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)

    def feature_process(self, env_obs, last_action):
        self.step_no += 1
        observation = env_obs.get("observation", env_obs)
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})

        hero = self._extract_hero(frame_state)
        hero_pos = self._extract_pos(hero)

        passable = self._extract_map_passable(observation)
        treasures, buffs = self._extract_organs(frame_state)
        monsters = self._extract_monsters(frame_state)

        treasure_layer, nearest_treasure_dist = self._entity_layer(treasures, hero_pos, decay=0.0)
        buff_layer, nearest_buff_dist = self._entity_layer(buffs, hero_pos, decay=0.0)
        monster_layer, nearest_monster_dist = self._entity_layer(monsters, hero_pos, decay=0.0)
        danger_layer, _ = self._entity_layer(monsters, hero_pos, decay=0.20)

        self.visit_counts *= Config.EXPLORATION_DECAY
        self.visit_counts[Config.VIEW_SIZE // 2, Config.VIEW_SIZE // 2] += 1.0
        visited_layer = np.clip(self.visit_counts / 4.0, 0.0, 1.0)

        stacked = np.stack([passable, treasure_layer, buff_layer, monster_layer, danger_layer, visited_layer], axis=0)

        legal_action = self._extract_legal_action(observation.get("legal_action", None))
        scalar_feature = self._build_scalar_feature(
            env_info=env_info,
            hero=hero,
            legal_action=legal_action,
            last_action=last_action,
            nearest_treasure_dist=nearest_treasure_dist,
            nearest_buff_dist=nearest_buff_dist,
            nearest_monster_dist=nearest_monster_dist,
            has_treasure=float(treasure_layer.sum()) > 0,
            has_buff=float(buff_layer.sum()) > 0,
            has_monster=float(monster_layer.sum()) > 0,
        )

        feature = np.concatenate([stacked.reshape(-1), scalar_feature], axis=0).astype(np.float32)
        reward = [
            float(
                self._shape_reward(
                    env_info=env_info,
                    hero=hero,
                    last_action=last_action,
                    nearest_treasure_dist=nearest_treasure_dist,
                    nearest_monster_dist=nearest_monster_dist,
                    legal_action=legal_action,
                )
            )
        ]
        return feature, legal_action, reward

    def _shape_reward(self, env_info, hero, last_action, nearest_treasure_dist, nearest_monster_dist, legal_action):
        reward = Config.REWARD_STEP_SURVIVE + Config.REWARD_STEP_PENALTY

        treasure_score = float(self._safe_get(env_info, ["treasure_score", "score"], self.prev_treasure_score))
        treasure_score_delta = max(0.0, treasure_score - self.prev_treasure_score)
        self.prev_treasure_score = treasure_score

        treasure_count = int(self._safe_get(env_info, ["treasures_collected"], self.prev_treasure_count))
        treasure_count_delta = max(0, treasure_count - self.prev_treasure_count)
        self.prev_treasure_count = treasure_count

        buff_count = int(self._safe_get(env_info, ["collected_buff"], self.prev_buff_count))
        buff_count_delta = max(0, buff_count - self.prev_buff_count)
        self.prev_buff_count = buff_count

        reward += Config.REWARD_TREASURE_PICKUP * float(treasure_count_delta)
        reward += Config.REWARD_TREASURE_PICKUP * 0.25 * treasure_score_delta
        reward += Config.REWARD_BUFF_PICKUP * float(buff_count_delta)

        moved_on_target = False
        if self.prev_treasure_dist is not None and nearest_treasure_dist is not None:
            delta_t = self.prev_treasure_dist - nearest_treasure_dist
            reward += Config.REWARD_TREASURE_PROGRESS * delta_t
            moved_on_target = moved_on_target or abs(delta_t) > 1e-6

        if self.prev_monster_dist is not None and nearest_monster_dist is not None:
            delta_m = nearest_monster_dist - self.prev_monster_dist
            reward += Config.REWARD_ESCAPE_PROGRESS * delta_m
            moved_on_target = moved_on_target or abs(delta_m) > 1e-6

        if nearest_monster_dist is not None:
            if nearest_monster_dist <= Config.DANGER_DISTANCE:
                danger_depth = Config.DANGER_DISTANCE - nearest_monster_dist + 1.0
                reward += Config.REWARD_DANGER_PENALTY * danger_depth
            else:
                reward += Config.REWARD_SAFE_BONUS

        if moved_on_target:
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
            if self.stagnation_steps >= Config.STAGNATION_STEPS:
                reward += Config.REWARD_STAGNATION_PENALTY

        novelty = 1.0 - np.clip(self.visit_counts[Config.VIEW_SIZE // 2, Config.VIEW_SIZE // 2], 0.0, 1.0)
        reward += Config.REWARD_EXPLORATION * novelty

        # Encourage flash for danger escape, punish blind flash.
        used_flash = last_action is not None and last_action >= 8 and last_action < Config.ACTION_NUM
        flash_legal = bool(np.sum(np.asarray(legal_action[8:16], dtype=np.float32)) > 0)
        if used_flash:
            in_danger = nearest_monster_dist is not None and nearest_monster_dist <= Config.DANGER_DISTANCE
            if in_danger and self.prev_monster_dist is not None and nearest_monster_dist > self.prev_monster_dist:
                reward += Config.REWARD_FLASH_ESCAPE
            elif not in_danger:
                reward += Config.REWARD_FLASH_WASTE
        elif flash_legal and nearest_monster_dist is not None and nearest_monster_dist <= Config.DANGER_DISTANCE - 1.0:
            reward += Config.REWARD_FLASH_WASTE * 0.4

        self.prev_treasure_dist = nearest_treasure_dist
        self.prev_monster_dist = nearest_monster_dist
        return reward

    def _build_scalar_feature(
        self,
        env_info,
        hero,
        legal_action,
        last_action,
        nearest_treasure_dist,
        nearest_buff_dist,
        nearest_monster_dist,
        has_treasure,
        has_buff,
        has_monster,
    ):
        max_view_dist = float(Config.VIEW_SIZE - 1) * 2.0

        total_treasure = max(1.0, float(self._safe_get(env_info, ["total_treasure"], 1.0)))
        treasures_collected = float(self._safe_get(env_info, ["treasures_collected"], 0.0))
        total_buff = max(1.0, float(self._safe_get(env_info, ["total_buff"], 1.0)))
        buffs_collected = float(self._safe_get(env_info, ["collected_buff"], 0.0))

        flash_count = float(self._safe_get(env_info, ["flash_count"], 0.0))
        flash_cooldown = float(self._safe_get(env_info, ["flash_cooldown"], 0.0))
        max_step = max(1.0, float(self._safe_get(env_info, ["max_step"], 1.0)))

        speed_left = float(self._safe_get(hero, ["speed_up_buff_left", "speed_buff_left", "buff_left"], 0.0))

        last_action_norm = 0.0 if last_action is None or last_action < 0 else float(last_action) / float(Config.ACTION_NUM - 1)
        treasure_dist_norm = 1.0 if nearest_treasure_dist is None else min(1.0, nearest_treasure_dist / max_view_dist)
        buff_dist_norm = 1.0 if nearest_buff_dist is None else min(1.0, nearest_buff_dist / max_view_dist)
        monster_dist_norm = 1.0 if nearest_monster_dist is None else min(1.0, nearest_monster_dist / max_view_dist)

        legal_move_ratio = float(np.mean(np.asarray(legal_action[:8], dtype=np.float32)))
        legal_flash_ratio = float(np.mean(np.asarray(legal_action[8:16], dtype=np.float32)))

        scalar = np.array(
            [
                min(1.0, self.step_no / max_step),
                treasures_collected / total_treasure,
                buffs_collected / total_buff,
                min(1.0, flash_count / 20.0),
                min(1.0, flash_cooldown / 50.0),
                min(1.0, speed_left / 50.0),
                treasure_dist_norm,
                buff_dist_norm,
                monster_dist_norm,
                1.0 if has_treasure else 0.0,
                1.0 if has_buff else 0.0,
                1.0 if has_monster else 0.0,
                last_action_norm,
                0.5 * (legal_move_ratio + legal_flash_ratio),
            ],
            dtype=np.float32,
        )
        return scalar

    def _extract_map_passable(self, observation):
        map_info = np.asarray(observation.get("map_info", np.ones((Config.VIEW_SIZE, Config.VIEW_SIZE))), dtype=np.float32)
        map_info = self._center_crop_or_pad(map_info, Config.VIEW_SIZE)
        return np.where(map_info > 0, 1.0, 0.0).astype(np.float32)

    def _extract_hero(self, frame_state):
        heroes = frame_state.get("heroes", []) if isinstance(frame_state, dict) else []
        if isinstance(heroes, list) and heroes:
            return heroes[0]
        if isinstance(heroes, dict):
            return heroes
        return {}

    def _extract_organs(self, frame_state):
        treasures = []
        buffs = []
        organs = frame_state.get("organs", []) if isinstance(frame_state, dict) else []
        if not isinstance(organs, list):
            return treasures, buffs

        for organ in organs:
            if int(self._safe_get(organ, ["status"], 0)) != 1:
                continue
            sub_type = int(self._safe_get(organ, ["sub_type"], 0))
            if sub_type == 1:
                treasures.append(organ)
            elif sub_type == 2:
                buffs.append(organ)
        return treasures, buffs

    def _extract_monsters(self, frame_state):
        monsters = frame_state.get("monsters", []) if isinstance(frame_state, dict) else []
        return monsters if isinstance(monsters, list) else []

    def _entity_layer(self, entities, hero_pos, decay=0.0):
        layer = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        nearest_dist = None
        center = Config.VIEW_SIZE // 2
        for entity in entities:
            pos = self._extract_pos(entity)
            if pos is None or hero_pos is None:
                continue
            dx = int(pos[0] - hero_pos[0])
            dz = int(pos[1] - hero_pos[1])
            ix, iz = center + dx, center + dz
            if 0 <= ix < Config.VIEW_SIZE and 0 <= iz < Config.VIEW_SIZE:
                manhattan = float(abs(dx) + abs(dz))
                nearest_dist = manhattan if nearest_dist is None else min(nearest_dist, manhattan)
                if decay > 0:
                    layer[iz, ix] = max(layer[iz, ix], np.exp(-decay * manhattan))
                else:
                    layer[iz, ix] = 1.0
        return layer, nearest_dist

    def _extract_pos(self, obj):
        if not isinstance(obj, dict):
            return None
        pos = obj.get("pos", None)
        if isinstance(pos, dict):
            x = pos.get("x", None)
            z = pos.get("z", None)
            if x is None or z is None:
                return None
            return int(x), int(z)
        return None

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
            mask = np.ones(Config.ACTION_NUM, dtype=np.float32)

        if float(mask.sum()) <= 0:
            mask = np.ones(Config.ACTION_NUM, dtype=np.float32)
        return mask.tolist()

    def _safe_get(self, obj, keys, default):
        if not isinstance(obj, dict):
            return default
        for key in keys:
            if key in obj:
                return obj[key]
        return default

    def _center_crop_or_pad(self, arr, target_size):
        if arr.ndim != 2:
            arr = np.asarray(arr).reshape(-1)
            side = int(np.sqrt(arr.shape[0]))
            if side * side == arr.shape[0]:
                arr = arr.reshape(side, side)
            else:
                return np.zeros((target_size, target_size), dtype=np.float32)

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
