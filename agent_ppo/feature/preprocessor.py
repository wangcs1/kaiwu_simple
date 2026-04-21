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
    def __init__(self, treasure_topk=None):
        self.treasure_topk = int(treasure_topk if treasure_topk is not None else Config.TREASURE_TARGET_TOPK)
        self.reset()

    def reset(self):
        self.step_no = 0
        self.prev_treasure_dist = None
        self.prev_monster_dist = None
        self.prev_treasure_score = 0.0
        self.prev_treasure_count = 0
        self.prev_buff_count = 0
        self.stagnation_steps = 0
        self.global_visit_counts = {}
        self.current_novelty = 1.0
        self.prev_monster_positions = {}
        self.prev_hero_pos = None
        self.same_pos_steps = 0
        self.prev_last_action = -1
        self.treasure_memory = {}
        self.current_target_dist = None

    def feature_process(self, env_obs, last_action):
        self.step_no += 1
        observation = env_obs.get("observation", env_obs)
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})

        hero = self._extract_hero(frame_state)
        hero_pos = self._extract_pos(hero)

        passable = self._extract_map_passable(observation)
        treasures, buffs, organ_debug = self._extract_organs(frame_state)
        monsters = self._extract_monsters(frame_state)

        ranked_treasures, nearest_treasure_dist = self._rank_topk_treasures(treasures, hero_pos)

        treasure_layer, _ = self._entity_layer(ranked_treasures, hero_pos, decay=0.0)
        buff_layer, nearest_buff_dist = self._entity_layer(buffs, hero_pos, decay=0.0)
        monster_layer, nearest_monster_dist = self._entity_layer(monsters, hero_pos, decay=0.0)
        danger_layer, _ = self._entity_layer(monsters, hero_pos, decay=0.20)
        predicted_danger = self._predict_monster_threat(monsters, hero_pos)
        danger_layer = np.maximum(danger_layer, predicted_danger)

        self._update_stuck_state(hero_pos)
        visited_layer, self.current_novelty = self._build_visited_layer(hero_pos)

        stacked = np.stack([passable, treasure_layer, buff_layer, monster_layer, danger_layer, visited_layer], axis=0)

        legal_action = self._extract_legal_action(observation.get("legal_action", None))
        legal_action = self._apply_obstacle_action_mask(legal_action=legal_action, passable=passable)
        scalar_feature = self._build_scalar_feature(
            env_info=env_info,
            hero=hero,
            passable=passable,
            legal_action=legal_action,
            last_action=last_action,
            nearest_treasure_dist=nearest_treasure_dist,
            nearest_buff_dist=nearest_buff_dist,
            nearest_monster_dist=nearest_monster_dist,
            second_monster_dist=self._second_nearest_dist(monsters, hero_pos),
            has_treasure=float(treasure_layer.sum()) > 0,
            has_buff=float(buff_layer.sum()) > 0,
            has_monster=float(monster_layer.sum()) > 0,
        )

        self.last_feature_debug = {
            "treasure_source": organ_debug.get("source", "unknown"),
            "treasure_count": len(tracked_treasures),
            "visible_treasure_count": len(treasures),
            "buff_count": len(buffs),
            "nearest_treasure_dist": nearest_treasure_dist,
            "treasure_layer_sum": float(treasure_layer.sum()),
            "treasure_topk": [
                {
                    "id": t.get("memory_id", -1),
                    "dist": t.get("rank_dist", None),
                    "value": t.get("rank_value", 0.0),
                }
                for t in tracked_treasures
            ],
        }

        feature = np.concatenate([stacked.reshape(-1), scalar_feature], axis=0).astype(np.float32)
        reward = [
            float(
                self._shape_reward(
                    env_info=env_info,
                    last_action=last_action,
                    nearest_treasure_dist=nearest_treasure_dist,
                    nearest_monster_dist=nearest_monster_dist,
                    legal_action=legal_action,
                )
            )
        ]
        self._update_monster_memory(monsters)
        return feature, legal_action, reward

    def _rank_topk_treasures(self, treasures, hero_pos):
        if hero_pos is None:
            self.current_target_dist = None
            return treasures, None

        current_step = int(self.step_no)
        visible = []
        for t in treasures:
            pos = self._extract_pos(t)
            if pos is None:
                continue

            key = (int(pos[0]), int(pos[1]))
            item = self.treasure_memory.get(key)
            if item is None:
                item = {
                    "first_seen": current_step,
                    "last_seen": current_step,
                    "seen_count": 0,
                }
            item["seen_count"] = int(item.get("seen_count", 0)) + 1
            item["last_seen"] = current_step
            self.treasure_memory[key] = item

            dist = float(abs(pos[0] - hero_pos[0]) + abs(pos[1] - hero_pos[1]))
            is_new = 1 if item["seen_count"] <= Config.TREASURE_NEW_SEEN_COUNT else 0
            visible.append((is_new, dist, -item["last_seen"], t))

        ttl = int(Config.TREASURE_MEMORY_TTL)
        stale_keys = [
            k for k, v in self.treasure_memory.items() if (current_step - int(v.get("last_seen", current_step))) > ttl
        ]
        for key in stale_keys:
            self.treasure_memory.pop(key, None)

        if not visible:
            self.current_target_dist = None
            return treasures, None

        visible.sort(key=lambda x: (-x[0], x[1], x[2]))
        topk = max(1, int(self.treasure_topk))
        selected = [x[3] for x in visible[:topk]]
        self.current_target_dist = float(visible[0][1])
        return selected, self.current_target_dist

    def _shape_reward(self, env_info, last_action, nearest_treasure_dist, nearest_monster_dist, legal_action):
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

        danger_level = self._danger_level(nearest_monster_dist)
        safe_level = 1.0 - danger_level

        moved_on_target = False
        if self.prev_treasure_dist is not None and nearest_treasure_dist is not None:
            delta_t = self.prev_treasure_dist - nearest_treasure_dist
            treasure_progress_weight = (
                Config.REWARD_TREASURE_PROGRESS_BASE + Config.REWARD_TREASURE_PROGRESS_SAFE_GAIN * safe_level
            )
            reward += treasure_progress_weight * delta_t
            prev_potential = 1.0 / (self.prev_treasure_dist + 1.0)
            curr_potential = 1.0 / (nearest_treasure_dist + 1.0)
            reward += Config.REWARD_TREASURE_NEAR_PROGRESS * safe_level * (curr_potential - prev_potential)
            if nearest_treasure_dist <= Config.TREASURE_NEAR_RADIUS:
                reward += Config.REWARD_TREASURE_NEAR_BONUS * safe_level * (
                    Config.TREASURE_NEAR_RADIUS - nearest_treasure_dist + 1.0
                )
            moved_on_target = moved_on_target or abs(delta_t) > 1e-6

        if self.prev_monster_dist is not None and nearest_monster_dist is not None:
            delta_m = nearest_monster_dist - self.prev_monster_dist
            escape_progress_weight = Config.REWARD_ESCAPE_PROGRESS_BASE + Config.REWARD_ESCAPE_PROGRESS_DANGER_GAIN * danger_level
            reward += escape_progress_weight * delta_m
            moved_on_target = moved_on_target or abs(delta_m) > 1e-6

        if nearest_monster_dist is not None:
            danger_depth = max(0.0, Config.DANGER_DISTANCE - nearest_monster_dist + 1.0)
            reward += Config.REWARD_MONSTER_PROXIMITY_PENALTY * danger_depth * (0.5 + danger_level)

        if moved_on_target:
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
            if self.stagnation_steps >= Config.STAGNATION_STEPS:
                reward += Config.REWARD_STAGNATION_PENALTY

        reward += Config.REWARD_EXPLORATION * self.current_novelty * (0.5 + 0.5 * safe_level)

        if self.same_pos_steps >= Config.STUCK_STEPS:
            stuck_scale = min(3.0, float(self.same_pos_steps) / float(Config.STUCK_STEPS))
            reward += Config.REWARD_STUCK_PENALTY * stuck_scale
            if 0 <= int(last_action) < 8 and int(last_action) == int(self.prev_last_action):
                reward += Config.REWARD_REPEAT_ACTION_STUCK

        # Encourage flash for danger escape, punish blind flash.
        used_flash = last_action is not None and last_action >= 8 and last_action < Config.ACTION_NUM
        flash_legal = bool(np.sum(np.asarray(legal_action[8:16], dtype=np.float32)) > 0)
        if used_flash:
            if danger_level >= 0.4 and self.prev_monster_dist is not None and nearest_monster_dist > self.prev_monster_dist:
                reward += Config.REWARD_FLASH_ESCAPE * (1.0 + 0.5 * danger_level)
            elif danger_level <= 0.2:
                reward += Config.REWARD_FLASH_WASTE_SAFE
        elif flash_legal and danger_level >= 0.7:
            reward += 0.4 * Config.REWARD_FLASH_WASTE_SAFE

        self.prev_last_action = -1 if last_action is None else int(last_action)
        self.prev_treasure_dist = nearest_treasure_dist
        self.prev_monster_dist = nearest_monster_dist
        return reward

    def _update_stuck_state(self, hero_pos):
        if hero_pos is None:
            self.same_pos_steps = 0
            self.prev_hero_pos = None
            return

        if self.prev_hero_pos is not None and hero_pos == self.prev_hero_pos:
            self.same_pos_steps += 1
        else:
            self.same_pos_steps = 0
        self.prev_hero_pos = hero_pos

    def _build_visited_layer(self, hero_pos):
        layer = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        if hero_pos is None:
            return layer, 1.0

        center = Config.VIEW_SIZE // 2
        hero_x, hero_z = int(hero_pos[0]), int(hero_pos[1])

        for dz in range(-center, center + 1):
            for dx in range(-center, center + 1):
                key = (hero_x + dx, hero_z + dz)
                count = float(self.global_visit_counts.get(key, 0.0))
                layer[center + dz, center + dx] = np.clip(count / 4.0, 0.0, 1.0)

        current_key = (hero_x, hero_z)
        current_count = float(self.global_visit_counts.get(current_key, 0.0))
        novelty = 1.0 / np.sqrt(current_count + 1.0)
        self.global_visit_counts[current_key] = current_count + 1.0

        return layer, float(novelty)

    def _danger_level(self, nearest_monster_dist):
        if nearest_monster_dist is None:
            return 0.0
        level = (Config.DANGER_DISTANCE - nearest_monster_dist) / max(1e-6, Config.DANGER_DISTANCE)
        return float(np.clip(level, 0.0, 1.0))

    def _build_scalar_feature(
        self,
        env_info,
        hero,
        passable,
        legal_action,
        last_action,
        nearest_treasure_dist,
        nearest_buff_dist,
        nearest_monster_dist,
        second_monster_dist,
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
        second_monster_dist_norm = (
            1.0 if second_monster_dist is None else min(1.0, second_monster_dist / max_view_dist)
        )
        danger_level = self._danger_level(nearest_monster_dist)
        corridor_score = self._corridor_score(passable)
        alive_space = self._alive_space_ratio(passable)
        is_speed_stage = 1.0 if danger_level > 0.45 or second_monster_dist_norm < 0.5 else 0.0

        legal_move_ratio = float(np.mean(np.asarray(legal_action[:8], dtype=np.float32)))
        legal_flash_ratio = float(np.mean(np.asarray(legal_action[8:16], dtype=np.float32)))
        flash_ready = 1.0 if float(flash_cooldown) <= 1e-5 and legal_flash_ratio > 0 else 0.0
        recent_novelty = 1.0 - np.clip(self.visit_counts[Config.VIEW_SIZE // 2, Config.VIEW_SIZE // 2], 0.0, 1.0)

        hero_state = np.array(
            [
                min(1.0, self.step_no / max_step),
                min(1.0, speed_left / 50.0),
                min(1.0, flash_count / 20.0),
                min(1.0, flash_cooldown / 50.0),
                flash_ready,
            ],
            dtype=np.float32,
        )
        treasure_state = np.array(
            [
                treasures_collected / total_treasure,
                buffs_collected / total_buff,
                treasure_dist_norm,
                buff_dist_norm,
                1.0 if has_treasure else 0.0,
                1.0 if has_buff else 0.0,
            ],
            dtype=np.float32,
        )
        monster_state = np.array(
            [
                monster_dist_norm,
                second_monster_dist_norm,
                danger_level,
                1.0 if has_monster else 0.0,
                1.0 if second_monster_dist_norm < 0.45 else 0.0,
            ],
            dtype=np.float32,
        )
        mobility_state = np.array(
            [
                legal_move_ratio,
                legal_flash_ratio,
                corridor_score,
                alive_space,
            ],
            dtype=np.float32,
        )
        stage_state = np.array(
            [
                last_action_norm,
                recent_novelty,
                is_speed_stage,
                0.5 * (legal_move_ratio + legal_flash_ratio),
            ],
            dtype=np.float32,
        )
        scalar = np.concatenate([hero_state, treasure_state, monster_state, mobility_state, stage_state], axis=0)
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
        debug = {"source": "none"}
        organ_sources = []
        if isinstance(frame_state, dict):
            for key in ("organs", "treasures", "buffs", "items"):
                raw = frame_state.get(key, [])
                entities = self._as_entity_list(raw)
                if entities:
                    organ_sources.append((key, entities))
        if not organ_sources:
            return treasures, buffs, debug

        for source, entities in organ_sources:
            debug["source"] = source
            for organ in entities:
                status = int(self._safe_get(organ, ["status", "alive", "active"], 1))
                if status not in (1, True):
                    continue
                sub_type = int(self._safe_get(organ, ["sub_type", "type", "organ_type"], 0))
                if source == "treasures" and sub_type == 0:
                    sub_type = 1
                elif source == "buffs" and sub_type == 0:
                    sub_type = 2
                name = str(self._safe_get(organ, ["name"], "")).lower()
                if sub_type == 1 or "treasure" in name or "chest" in name or "宝箱" in name:
                    treasures.append(organ)
                elif sub_type == 2 or "buff" in name:
                    buffs.append(organ)
        return treasures, buffs, debug

    def _as_entity_list(self, raw):
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        if isinstance(raw, dict):
            values = [v for v in raw.values() if isinstance(v, dict)]
            return values if values else [raw]
        return []

    def _second_nearest_dist(self, entities, hero_pos):
        dists = []
        for entity in entities:
            pos = self._extract_pos(entity)
            if pos is None or hero_pos is None:
                continue
            dx = int(pos[0] - hero_pos[0])
            dz = int(pos[1] - hero_pos[1])
            dists.append(float(abs(dx) + abs(dz)))
        if len(dists) < 2:
            return None
        dists.sort()
        return dists[1]

    def _extract_monsters(self, frame_state):
        monsters = frame_state.get("monsters", []) if isinstance(frame_state, dict) else []
        return monsters if isinstance(monsters, list) else []

    def _update_and_rank_treasures(self, visible_treasures, hero_pos):
        for mem in self.treasure_memory.values():
            mem["visible"] = False

        for treasure in visible_treasures:
            memory_id = self._treasure_memory_id(treasure)
            pos = self._extract_pos(treasure)
            if pos is None:
                continue
            prev = self.treasure_memory.get(
                memory_id,
                {
                    "memory_id": memory_id,
                    "last_seen": -1,
                    "value": 0.0,
                    "pos": pos,
                },
            )
            prev["pos"] = pos
            prev["last_seen"] = self.step_no
            prev["visible"] = True
            prev["value"] = max(prev.get("value", 0.0), self._treasure_value(treasure))
            self.treasure_memory[memory_id] = prev

        stale_ids = []
        for memory_id, mem in self.treasure_memory.items():
            if self.step_no - int(mem.get("last_seen", -1)) > Config.TREASURE_MEMORY_TTL:
                stale_ids.append(memory_id)
        for memory_id in stale_ids:
            self.treasure_memory.pop(memory_id, None)

        ranked = []
        for mem in self.treasure_memory.values():
            mem_pos = mem.get("pos", None)
            if mem_pos is None or hero_pos is None:
                rank_dist = 1e6
            else:
                rank_dist = float(abs(mem_pos[0] - hero_pos[0]) + abs(mem_pos[1] - hero_pos[1]))
            ranked.append(
                {
                    "memory_id": int(mem.get("memory_id", -1)),
                    "pos": {"x": int(mem_pos[0]), "z": int(mem_pos[1])},
                    "rank_dist": rank_dist,
                    "rank_value": float(mem.get("value", 0.0)),
                    "visible": 1.0 if bool(mem.get("visible", False)) else 0.0,
                    "recency": float(self.step_no - int(mem.get("last_seen", -1))),
                }
            )

        ranked.sort(
            key=lambda x: (
                -x["visible"],
                -x["rank_value"],
                x["rank_dist"],
                x["recency"],
            )
        )
        return ranked[: Config.TREASURE_TRACK_TOPK]

    def _treasure_memory_id(self, treasure):
        fixed_id = self._safe_get(treasure, ["organ_id", "id", "treasure_id"], None)
        if fixed_id is not None:
            return int(fixed_id)
        pos = self._extract_pos(treasure)
        if pos is None:
            return -1
        return int(pos[0] * 1000 + pos[1])

    def _treasure_value(self, treasure):
        score = self._safe_get(treasure, ["score", "value", "reward"], 0.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

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

    def _predict_monster_threat(self, monsters, hero_pos):
        layer = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        if hero_pos is None:
            return layer

        center = Config.VIEW_SIZE // 2
        for monster in monsters:
            monster_id = int(self._safe_get(monster, ["monster_id"], -1))
            cur_pos = self._extract_pos(monster)
            if cur_pos is None:
                continue

            prev_pos = self.prev_monster_positions.get(monster_id, cur_pos)
            vx = int(cur_pos[0] - prev_pos[0])
            vz = int(cur_pos[1] - prev_pos[1])
            pred_pos = (cur_pos[0] + vx, cur_pos[1] + vz)

            dx = int(pred_pos[0] - hero_pos[0])
            dz = int(pred_pos[1] - hero_pos[1])
            ix, iz = center + dx, center + dz
            if 0 <= ix < Config.VIEW_SIZE and 0 <= iz < Config.VIEW_SIZE:
                dist = float(abs(dx) + abs(dz))
                layer[iz, ix] = max(layer[iz, ix], np.exp(-0.55 * dist))
        return layer

    def _update_monster_memory(self, monsters):
        updated = {}
        for monster in monsters:
            monster_id = int(self._safe_get(monster, ["monster_id"], -1))
            pos = self._extract_pos(monster)
            if monster_id >= 0 and pos is not None:
                updated[monster_id] = pos
        self.prev_monster_positions = updated

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

    def _corridor_score(self, passable):
        center = Config.VIEW_SIZE // 2
        if passable.shape != (Config.VIEW_SIZE, Config.VIEW_SIZE):
            return 0.0
        rays = []
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            length = 0
            x, z = center, center
            while True:
                x += dx
                z += dz
                if not (0 <= x < Config.VIEW_SIZE and 0 <= z < Config.VIEW_SIZE):
                    break
                if passable[z, x] <= 0.5:
                    break
                length += 1
            rays.append(length / float(Config.VIEW_SIZE - 1))
        return float(np.mean(rays))

    def _alive_space_ratio(self, passable):
        if passable.shape != (Config.VIEW_SIZE, Config.VIEW_SIZE):
            return 0.0
        return float(np.mean(passable > 0.5))

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

    def _apply_obstacle_action_mask(self, legal_action, passable):
        env_mask = np.asarray(legal_action, dtype=np.float32).reshape(-1)
        if env_mask.size < Config.ACTION_NUM:
            pad = np.ones(Config.ACTION_NUM - env_mask.size, dtype=np.float32)
            env_mask = np.concatenate([env_mask, pad], axis=0)
        env_mask = env_mask[: Config.ACTION_NUM]

        if not Config.ENABLE_OBSTACLE_ACTION_MASK:
            return env_mask.tolist()

        move_mask, flash_mask = self._build_obstacle_masks(passable)
        mode = str(Config.OBSTACLE_MASK_MODE).lower().strip()

        if mode == "hard":
            merged = env_mask.copy()
            merged[:8] = merged[:8] * move_mask
            merged[8:16] = merged[8:16] * flash_mask
        else:
            strength = float(np.clip(Config.OBSTACLE_MASK_STRENGTH, 0.0, 1.0))
            flash_scale = float(np.clip(Config.FLASH_OBSTACLE_MASK_SCALE, 0.0, 1.0))

            merged = env_mask.copy()
            move_gate = (1.0 - strength) + strength * move_mask
            flash_gate = (1.0 - strength * flash_scale) + strength * flash_scale * flash_mask
            merged[:8] = merged[:8] * move_gate
            merged[8:16] = merged[8:16] * flash_gate

        if float(np.sum(merged)) <= 1e-6:
            merged = env_mask
        return merged.tolist()

    def _build_obstacle_masks(self, passable):
        move_mask = np.ones(8, dtype=np.float32)
        flash_mask = np.ones(8, dtype=np.float32)

        grid = np.asarray(passable, dtype=np.float32)
        if grid.ndim != 2:
            return move_mask, flash_mask

        h, w = grid.shape
        cz = h // 2
        cx = w // 2

        dirs = list(getattr(Config, "ACTION_MOVE_DIRS", []))
        if len(dirs) < 8:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

        for i in range(8):
            dx, dz = int(dirs[i][0]), int(dirs[i][1])
            nz = cz + dz
            nx = cx + dx
            if 0 <= nz < h and 0 <= nx < w:
                move_mask[i] = 1.0 if grid[nz, nx] > 0.5 else 0.0
            else:
                move_mask[i] = 0.0

            fz = cz + 2 * dz
            fx = cx + 2 * dx
            if 0 <= fz < h and 0 <= fx < w:
                near_ok = 1.0 if grid[nz, nx] > 0.5 else 0.0
                far_ok = 1.0 if grid[fz, fx] > 0.5 else 0.0
                flash_mask[i] = 1.0 if (near_ok > 0.5 and far_ok > 0.5) else 0.0
            else:
                flash_mask[i] = 0.0

        if float(np.sum(move_mask)) <= 1e-6:
            move_mask[:] = 1.0
        if float(np.sum(flash_mask)) <= 1e-6:
            flash_mask[:] = 1.0

        return move_mask, flash_mask

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
