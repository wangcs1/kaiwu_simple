#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessing and reward shaping for Gorge Chase PPO.
"""

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature import bfs as bfs_mod


def _clip01(v):
    return float(max(0.0, min(1.0, v)))


def _norm(v, v_max, v_min=0.0):
    if v_max - v_min <= 1e-6:
        return 0.0
    return _clip01((float(v) - v_min) / (v_max - v_min))


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000

        self.last_treasures_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0

        self.last_hero_pos = None
        self.last_monster_pos = {}
        self.trajectory = []
        self.position_history = []

        self.last_potential = 0.0
        self.last_min_monster_bfs = Config.BFS_MAX_STEPS

        self.view_h = Config.MAP_VIEW
        self.view_w = Config.MAP_VIEW
        self.wall_bump_count = 0

        self.seen_treasure_positions = set()
        self.cached_treasure_positions = set()

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info_raw = observation.get("map_info", None)
        legal_act_raw = observation.get("legal_action", None)

        self.step_no = int(observation.get("step_no", env_info.get("step_no", 0)))
        self.max_step = int(env_info.get("max_step", 1000))

        map_grid = self._normalize_map(map_info_raw)
        h, w = map_grid.shape
        self.view_h, self.view_w = h, w
        center_x = w // 2
        center_z = h // 2

        hero = frame_state.get("heroes", {}) or {}
        if isinstance(hero, list):
            hero = hero[0] if hero else {}
        hero_pos = hero.get("pos", {}) or {}
        hero_global_x = int(hero_pos.get("x", 0))
        hero_global_z = int(hero_pos.get("z", 0))

        organs = frame_state.get("organs", []) or []
        treasure_local = []
        buff_local = []
        for o in organs:
            sub_type = int(o.get("sub_type", 0))
            status = int(o.get("status", 0))
            bucket = int(o.get("hero_l2_distance", 5))
            dir_e = int(o.get("hero_relative_direction", 0))
            opos = o.get("pos", {}) or {}
            ox, oz = int(opos.get("x", 0)), int(opos.get("z", 0))
            lx = ox - hero_global_x + center_x
            lz = oz - hero_global_z + center_z
            entry = (lx, lz, bucket, dir_e, status, ox, oz)
            if sub_type == 1 and status == 1:
                treasure_local.append(entry)
            elif sub_type == 2 and status == 1:
                buff_local.append(entry)

        treasure_in_view_set = {
            (e[0], e[1]) for e in treasure_local if 0 <= e[0] < w and 0 <= e[1] < h
        }
        buff_in_view_set = {
            (e[0], e[1]) for e in buff_local if 0 <= e[0] < w and 0 <= e[1] < h
        }

        monsters_raw = frame_state.get("monsters", []) or []
        monsters = [monsters_raw[i] if i < len(monsters_raw) else None for i in range(2)]

        rays = bfs_mod.compute_obstacle_rays(map_grid, center_x, center_z)
        flash_landings = bfs_mod.compute_flash_landing(
            map_grid, center_x, center_z, treasure_in_view_set, buff_in_view_set
        )
        movable_mask = bfs_mod.compute_movable_mask(map_grid, center_x, center_z)
        flash_useful_mask = bfs_mod.compute_flash_useful_mask(flash_landings)
        hero_dist_map, hero_first_dir_map = bfs_mod.local_bfs(map_grid, center_x, center_z)

        reachable_treasure_set = set()
        reachable_buff_set = set()
        reachable_visible_treasure_globals = set()
        has_unreachable_treasure_in_view = False
        nearest_treasure_bfs = Config.BFS_MAX_STEPS
        for (lx, lz, _bucket, _dir_e, _status, gx, gz) in treasure_local:
            if not (0 <= lx < w and 0 <= lz < h):
                continue
            d = int(hero_dist_map[lz, lx])
            if d >= 0:
                reachable_treasure_set.add((lx, lz))
                reachable_visible_treasure_globals.add((gx, gz))
                nearest_treasure_bfs = min(nearest_treasure_bfs, d)
            else:
                has_unreachable_treasure_in_view = True
        for (lx, lz) in buff_in_view_set:
            if 0 <= lx < w and 0 <= lz < h and int(hero_dist_map[lz, lx]) >= 0:
                reachable_buff_set.add((lx, lz))

        map_tensor = self._build_map_tensor(
            map_grid,
            reachable_treasure_set,
            reachable_buff_set,
            monsters,
            hero_global_x,
            hero_global_z,
            center_x,
            center_z,
        )

        sym_feat, dead_end_score = self._build_sym_feat(
            hero=hero,
            env_info=env_info,
            monsters=monsters,
            treasure_local=treasure_local,
            buff_local=buff_local,
            remembered_treasures=self.cached_treasure_positions,
            hero_global=(hero_global_x, hero_global_z),
            center=(center_x, center_z),
            map_grid=map_grid,
            hero_dist_map=hero_dist_map,
            rays=rays,
            flash_landings=flash_landings,
            movable_mask=movable_mask,
            flash_useful_mask=flash_useful_mask,
            last_action=last_action,
        )

        flash_cd = int(hero.get("flash_cooldown", 0))
        flash_ready = 1 if flash_cd <= 0 else 0
        env_flash_mask = self._parse_env_flash_mask(legal_act_raw)
        legal_action_16d = self._build_legal_16(
            movable_mask, flash_ready, flash_useful_mask, env_flash_mask
        )

        cur_treasures_collected = int(env_info.get("treasures_collected", 0))
        cur_collected_buff = int(env_info.get("collected_buff", 0))
        cur_flash_count = int(env_info.get("flash_count", 0))
        picked_treasure = cur_treasures_collected > self.last_treasures_collected
        picked_buff = cur_collected_buff > self.last_collected_buff
        used_flash = cur_flash_count > self.last_flash_count

        newly_seen_treasures = reachable_visible_treasure_globals - self.seen_treasure_positions
        first_seen_treasure_count = len(newly_seen_treasures)

        bumped_wall = False
        if (
            self.last_hero_pos is not None
            and last_action is not None
            and 0 <= int(last_action) <= 7
            and (hero_global_x, hero_global_z) == self.last_hero_pos
        ):
            bumped_wall = True
            self.wall_bump_count += 1

        min_monster_bfs = Config.BFS_MAX_STEPS
        monster_bfs_dists = [Config.BFS_MAX_STEPS, Config.BFS_MAX_STEPS]
        visible_monster_count = 0
        for mi, m in enumerate(monsters):
            if m is None or int(m.get("is_in_view", 0)) == 0:
                continue
            visible_monster_count += 1
            mpos = m.get("pos", {}) or {}
            mx, mz = int(mpos.get("x", 0)), int(mpos.get("z", 0))
            lx = mx - hero_global_x + center_x
            lz = mz - hero_global_z + center_z
            if 0 <= lx < w and 0 <= lz < h:
                d = int(hero_dist_map[lz, lx])
                if d >= 0:
                    monster_bfs_dists[mi] = d
                    min_monster_bfs = min(min_monster_bfs, d)

        opening_treasure_shield = (
            self.step_no <= Config.OPENING_TREASURE_SHIELD_STEPS
            and (
                visible_monster_count == 0
                or min_monster_bfs <= Config.OPENING_TREASURE_MONSTER_SAFE_DIST
            )
        )

        max_monster_speed = 1
        for m in monsters:
            if m is None:
                continue
            max_monster_speed = max(max_monster_speed, int(m.get("speed", 1)))
        late_factor = _clip01((max_monster_speed - 1.0) / 1.0)
        is_late_phase = late_factor > 0.5

        flash_path_had_item = False
        if used_flash and last_action is not None and 8 <= int(last_action) <= 15:
            dir_idx = int(last_action) - 8
            flash_path_had_item = flash_landings[dir_idx][2] > 0

        monster_bfs_gain = int(min_monster_bfs - self.last_min_monster_bfs)
        dead_end = dead_end_score > Config.DEAD_END_RATIO_THRESHOLD

        is_lingering = False
        if (
            self.step_no <= Config.LINGER_STEP_LIMIT
            and len(self.position_history) >= Config.LINGER_WINDOW
        ):
            old_x, old_z = self.position_history[-Config.LINGER_WINDOW]
            linger_dist = float(np.hypot(hero_global_x - old_x, hero_global_z - old_z))
            is_lingering = linger_dist < Config.LINGER_L2_THRESHOLD

        flash_escape_scores = self._compute_flash_escape_scores(
            map_grid=map_grid,
            monsters=monsters,
            hero_global=(hero_global_x, hero_global_z),
            center=(center_x, center_z),
            flash_landings=flash_landings,
        )

        reward = self._compute_reward(
            picked_treasure=picked_treasure,
            picked_buff=picked_buff,
            used_flash=used_flash,
            flash_path_had_item=flash_path_had_item,
            monster_bfs_gain=monster_bfs_gain,
            bumped_wall=bumped_wall,
            dead_end=dead_end,
            late_factor=late_factor,
            nearest_treasure_bfs=nearest_treasure_bfs,
            min_monster_bfs=min_monster_bfs,
            first_seen_treasure_count=first_seen_treasure_count,
            opening_treasure_shield=opening_treasure_shield,
            has_unreachable_treasure_in_view=has_unreachable_treasure_in_view,
            is_lingering=is_lingering,
        )

        self.last_treasures_collected = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_flash_count = cur_flash_count
        self.last_hero_pos = (hero_global_x, hero_global_z)
        self.last_min_monster_bfs = min_monster_bfs

        self.seen_treasure_positions.update(newly_seen_treasures)
        self.cached_treasure_positions.update(reachable_visible_treasure_globals)
        if picked_treasure:
            self._consume_cached_treasure(hero_global_x, hero_global_z)

        for mi, m in enumerate(monsters):
            if m is None:
                continue
            mpos = m.get("pos", {}) or {}
            self.last_monster_pos[mi] = (int(mpos.get("x", 0)), int(mpos.get("z", 0)))

        self.trajectory.append((hero_global_x, hero_global_z))
        self.position_history.append((hero_global_x, hero_global_z))
        max_hist = max(Config.TRAJECTORY_LEN, Config.LINGER_WINDOW, Config.POSITION_HISTORY_WINDOW)
        if len(self.trajectory) > max_hist:
            self.trajectory.pop(0)
        if len(self.position_history) > max_hist:
            self.position_history.pop(0)

        monster_dirs = []
        for m in monsters:
            if m is None or int(m.get("is_in_view", 0)) == 0:
                monster_dirs.append(-1)
                continue
            mpos = m.get("pos", {}) or {}
            mx, mz = int(mpos.get("x", 0)), int(mpos.get("z", 0))
            lx = mx - hero_global_x + center_x
            lz = mz - hero_global_z + center_z
            first_dir = -1
            if 0 <= lx < w and 0 <= lz < h:
                first_dir = int(hero_first_dir_map[lz, lx])
            monster_dirs.append(first_dir)

        adjacent_treasure_dir = -1
        for (lx, lz, _b, _d, _s, _gx, _gz) in treasure_local:
            if not (0 <= lx < w and 0 <= lz < h):
                continue
            if int(hero_dist_map[lz, lx]) < 0:
                continue
            dx = lx - center_x
            dz = lz - center_z
            if max(abs(dx), abs(dz)) == 1 and (dx != 0 or dz != 0):
                for k, (odx, odz) in enumerate(Config.DIR_OFFSETS):
                    if odx == dx and odz == dz and movable_mask[k]:
                        adjacent_treasure_dir = k
                        break
                if adjacent_treasure_dir != -1:
                    break

        rule_hints = {
            "movable_mask": movable_mask,
            "flash_useful_mask": flash_useful_mask,
            "flash_ready": bool(flash_ready),
            "adjacent_treasure_dir": adjacent_treasure_dir,
            "monster_bfs_dists": monster_bfs_dists,
            "monster_dirs": monster_dirs,
            "is_late_phase": is_late_phase,
            "flash_landings": flash_landings,
            "flash_escape_scores": flash_escape_scores,
            "opening_treasure_shield": opening_treasure_shield,
            "step_no": self.step_no,
        }

        return {
            "map_tensor": map_tensor.astype(np.float32),
            "sym_feat": sym_feat.astype(np.float32),
            "legal_action_16d": legal_action_16d,
            "movable_mask_8d": movable_mask,
            "flash_useful_mask_8d": flash_useful_mask,
            "reward": [float(reward)],
            "rule_hints": rule_hints,
        }

    def _consume_cached_treasure(self, hero_x, hero_z):
        if not self.cached_treasure_positions:
            return
        nearest = min(
            self.cached_treasure_positions,
            key=lambda p: (np.hypot(hero_x - p[0], hero_z - p[1]), p[0], p[1]),
        )
        if np.hypot(hero_x - nearest[0], hero_z - nearest[1]) <= 2.0:
            self.cached_treasure_positions.discard(nearest)

    def _compute_flash_escape_scores(self, map_grid, monsters, hero_global, center, flash_landings):
        hgx, hgz = hero_global
        cx, cz = center
        scores = []
        for (dx, dz, _cnt) in flash_landings:
            if dx == 0 and dz == 0:
                scores.append(0)
                continue
            land_x = cx + dx
            land_z = cz + dz
            dist_map, _ = bfs_mod.local_bfs(map_grid, land_x, land_z)
            min_dist = Config.BFS_MAX_STEPS
            has_visible_monster = False
            for m in monsters:
                if m is None or int(m.get("is_in_view", 0)) == 0:
                    continue
                has_visible_monster = True
                mpos = m.get("pos", {}) or {}
                lx = int(mpos.get("x", 0)) - hgx + cx
                lz = int(mpos.get("z", 0)) - hgz + cz
                if 0 <= lx < self.view_w and 0 <= lz < self.view_h:
                    d = int(dist_map[lz, lx])
                    if d >= 0:
                        min_dist = min(min_dist, d)
            scores.append(min_dist if has_visible_monster else 0)
        return scores

    def _normalize_map(self, map_info_raw):
        if map_info_raw is None:
            return np.ones((Config.MAP_VIEW, Config.MAP_VIEW), dtype=np.int32)
        arr = np.asarray(map_info_raw, dtype=np.int32)
        if arr.ndim != 2:
            return np.ones((Config.MAP_VIEW, Config.MAP_VIEW), dtype=np.int32)
        return (arr != 0).astype(np.int32)

    def _build_map_tensor(self, map_grid, treasure_set, buff_set, monsters, hero_gx, hero_gz, cx, cz):
        h, w = map_grid.shape
        tensor = np.zeros((Config.MAP_CHANNELS, h, w), dtype=np.float32)
        tensor[0] = (map_grid == 0).astype(np.float32)
        for (lx, lz) in treasure_set:
            tensor[1, lz, lx] = 1.0
        for (lx, lz) in buff_set:
            tensor[2, lz, lx] = 1.0
        for m in monsters:
            if m is None or int(m.get("is_in_view", 0)) == 0:
                continue
            mpos = m.get("pos", {}) or {}
            lx = int(mpos.get("x", 0)) - hero_gx + cx
            lz = int(mpos.get("z", 0)) - hero_gz + cz
            if 0 <= lx < w and 0 <= lz < h:
                tensor[3, lz, lx] = 1.0
        tensor[4, cz, cx] = 1.0
        for i, (gx, gz) in enumerate(reversed(self.trajectory)):
            decay = 1.0 - i / max(Config.TRAJECTORY_LEN, 1)
            lx = gx - hero_gx + cx
            lz = gz - hero_gz + cz
            if 0 <= lx < w and 0 <= lz < h:
                tensor[5, lz, lx] = max(tensor[5, lz, lx], decay)
        if h == Config.MAP_VIEW and w == Config.MAP_VIEW:
            return tensor
        out = np.zeros((Config.MAP_CHANNELS, Config.MAP_VIEW, Config.MAP_VIEW), dtype=np.float32)
        ocx, ocz = Config.MAP_VIEW // 2, Config.MAP_VIEW // 2
        for zz in range(Config.MAP_VIEW):
            for xx in range(Config.MAP_VIEW):
                sx = cx + (xx - ocx)
                sz = cz + (zz - ocz)
                if 0 <= sx < w and 0 <= sz < h:
                    out[:, zz, xx] = tensor[:, sz, sx]
        return out

    def _build_sym_feat(
        self,
        hero,
        env_info,
        monsters,
        treasure_local,
        buff_local,
        remembered_treasures,
        hero_global,
        center,
        map_grid,
        hero_dist_map,
        rays,
        flash_landings,
        movable_mask,
        flash_useful_mask,
        last_action,
    ):
        hgx, hgz = hero_global
        cx, cz = center
        h, w = map_grid.shape

        flash_cd = int(hero.get("flash_cooldown", 0))
        flash_ready = 1.0 if flash_cd <= 0 else 0.0
        buff_remain = float(hero.get("buff_remaining_time", 0))
        has_speed_buff = 1.0 if buff_remain > 0 else 0.0
        hero_feat = np.array(
            [
                _norm(hgx, Config.MAP_SIZE),
                _norm(hgz, Config.MAP_SIZE),
                _norm(flash_cd, Config.FLASH_COOLDOWN_DEFAULT),
                flash_ready,
                _norm(buff_remain, Config.MAX_BUFF_DURATION),
                has_speed_buff,
            ],
            dtype=np.float32,
        )

        monster_feats = []
        for mi in range(2):
            m = monsters[mi]
            if m is None:
                monster_feats.append(np.zeros(10, dtype=np.float32))
                continue
            is_in_view = int(m.get("is_in_view", 0))
            mpos = m.get("pos", {}) or {}
            mx, mz = int(mpos.get("x", 0)), int(mpos.get("z", 0))
            speed = int(m.get("speed", 1))
            bucket = int(m.get("hero_l2_distance", 5))
            dir_e = int(m.get("hero_relative_direction", 0))

            pos_x_norm = 0.0
            pos_z_norm = 0.0
            euclid_dist = 1.0
            bfs_or_bucket = bucket / 5.0
            delta_dist = 0.0

            if is_in_view:
                pos_x_norm = _norm(mx, Config.MAP_SIZE)
                pos_z_norm = _norm(mz, Config.MAP_SIZE)
                raw = np.hypot(hgx - mx, hgz - mz)
                euclid_dist = _norm(raw, Config.MAP_SIZE * 1.4143)
                lx = mx - hgx + cx
                lz = mz - hgz + cz
                if 0 <= lx < w and 0 <= lz < h:
                    d = int(hero_dist_map[lz, lx])
                    if d >= 0:
                        bfs_or_bucket = d / float(Config.BFS_MAX_STEPS)
                last_mp = self.last_monster_pos.get(mi)
                if last_mp is not None:
                    delta_dist = np.hypot(hgx - mx, hgz - mz) - np.hypot(hgx - last_mp[0], hgz - last_mp[1])
                    delta_dist = float(np.clip(delta_dist / 4.0, -1.0, 1.0))

            cos_d, sin_d, unk_d = bfs_mod.bucket_direction_to_cos_sin(dir_e)
            monster_feats.append(
                np.array(
                    [
                        float(is_in_view),
                        pos_x_norm,
                        pos_z_norm,
                        _norm(speed, Config.MAX_MONSTER_SPEED),
                        euclid_dist,
                        bfs_or_bucket,
                        cos_d,
                        sin_d,
                        unk_d,
                        delta_dist,
                    ],
                    dtype=np.float32,
                )
            )

        treasure_entries = []
        for (lx, lz, bucket, dir_e, _st, _gx, _gz) in treasure_local:
            if 0 <= lx < w and 0 <= lz < h:
                d = int(hero_dist_map[lz, lx])
                if d >= 0:
                    treasure_entries.append((d, bucket, dir_e))
        treasure_entries.sort(key=lambda t: (t[0], t[1]))
        treasure_feat = []
        for i in range(Config.TOP_K_TREASURE):
            if i < len(treasure_entries):
                d, _bucket, dir_e = treasure_entries[i]
                cos_d, sin_d, _unk = bfs_mod.bucket_direction_to_cos_sin(dir_e)
                treasure_feat.extend([d / float(Config.BFS_MAX_STEPS), cos_d, sin_d])
            else:
                treasure_feat.extend([1.0, 0.0, 0.0])
        treasure_feat = np.array(treasure_feat, dtype=np.float32)

        if buff_local:
            best = None
            for (lx, lz, bucket, dir_e, status, _gx, _gz) in buff_local:
                d = Config.BFS_MAX_STEPS
                if 0 <= lx < w and 0 <= lz < h:
                    dd = int(hero_dist_map[lz, lx])
                    if dd >= 0:
                        d = dd
                if best is None or d < best[0]:
                    best = (d, bucket, dir_e, status)
            d, bucket, dir_e, status = best
            dist_norm = d / float(Config.BFS_MAX_STEPS) if d < Config.BFS_MAX_STEPS else bucket / 5.0
            cos_d, sin_d, _unk = bfs_mod.bucket_direction_to_cos_sin(dir_e)
            buff_feat = np.array([dist_norm, cos_d, sin_d, float(status == 1)], dtype=np.float32)
        else:
            buff_feat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        rays_feat = np.array([r / float(Config.FLASH_RANGE_STRAIGHT) for r in rays], dtype=np.float32)

        flash_feat = []
        for (dx, dz, cnt) in flash_landings:
            flash_feat.extend(
                [
                    dx / float(Config.FLASH_RANGE_STRAIGHT),
                    dz / float(Config.FLASH_RANGE_STRAIGHT),
                    min(cnt, 5) / 5.0,
                ]
            )
        flash_feat = np.array(flash_feat, dtype=np.float32)

        last_oh = np.zeros(16, dtype=np.float32)
        if last_action is not None and 0 <= int(last_action) < 16:
            last_oh[int(last_action)] = 1.0

        movable_feat = np.array(movable_mask, dtype=np.float32)
        flash_useful_feat = np.array(flash_useful_mask, dtype=np.float32)
        flash_ready_feat = np.array([flash_ready], dtype=np.float32)

        max_monster_speed = 1
        for m in monsters:
            if m is None:
                continue
            max_monster_speed = max(max_monster_speed, int(m.get("speed", 1)))
        late_factor = _clip01((max_monster_speed - 1.0) / 1.0)
        phase_feat = np.array(
            [
                max_monster_speed / Config.MAX_MONSTER_SPEED,
                1.0 if late_factor > 0.5 else 0.0,
                flash_ready * (1.0 if late_factor > 0.5 else 0.0),
                _clip01(1.0 - self.step_no / max(self.max_step, 1)),
            ],
            dtype=np.float32,
        )

        total_tre = max(int(env_info.get("total_treasure", 1)), 1)
        got_tre = int(env_info.get("treasures_collected", 0))
        progress_feat = np.array(
            [_norm(self.step_no, self.max_step), got_tre / float(total_tre)],
            dtype=np.float32,
        )

        if len(self.position_history) >= Config.POSITION_HISTORY_WINDOW:
            old_x, old_z = self.position_history[-Config.POSITION_HISTORY_WINDOW]
            pos10_feat = np.array(
                [
                    np.clip((old_x - hgx) / float(Config.FLASH_RANGE_STRAIGHT), -1.0, 1.0),
                    np.clip((old_z - hgz) / float(Config.FLASH_RANGE_STRAIGHT), -1.0, 1.0),
                ],
                dtype=np.float32,
            )
        else:
            pos10_feat = np.zeros(2, dtype=np.float32)

        remembered_feat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        remembered_candidates = [(tx, tz) for (tx, tz) in remembered_treasures if (tx, tz) != (hgx, hgz)]
        if remembered_candidates:
            tx, tz = min(remembered_candidates, key=lambda p: np.hypot(p[0] - hgx, p[1] - hgz))
            dx = tx - hgx
            dz = tz - hgz
            dist = np.hypot(dx, dz)
            if dist > 1e-6:
                remembered_feat = np.array(
                    [
                        _norm(dist, Config.MAP_SIZE * 1.4143),
                        float(np.clip(dx / dist, -1.0, 1.0)),
                        float(np.clip(dz / dist, -1.0, 1.0)),
                    ],
                    dtype=np.float32,
                )

        sym_feat = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                buff_feat,
                rays_feat,
                flash_feat,
                last_oh,
                movable_feat,
                flash_useful_feat,
                flash_ready_feat,
                phase_feat,
                progress_feat,
                pos10_feat,
                remembered_feat,
            ]
        ).astype(np.float32)
        assert sym_feat.shape[0] == Config.SYM_FEATURE_LEN

        blocked = sum(1 for r in rays if r <= Config.DEAD_END_RAY_THRESHOLD)
        dead_end_score = blocked / 8.0
        return sym_feat, dead_end_score

    def _parse_env_flash_mask(self, legal_act_raw):
        if legal_act_raw is None:
            return [1] * 8
        if isinstance(legal_act_raw, np.ndarray):
            legal_act_raw = legal_act_raw.tolist()
        if not isinstance(legal_act_raw, (list, tuple)) or len(legal_act_raw) == 0:
            return [1] * 8
        if isinstance(legal_act_raw[0], (bool, np.bool_, int, np.integer)):
            if len(legal_act_raw) >= 16:
                return [int(bool(legal_act_raw[8 + j])) for j in range(8)]
        return [1] * 8

    def _build_legal_16(self, movable_mask, flash_ready, flash_useful_mask, env_flash_mask):
        mask = [0] * 16
        for k in range(8):
            mask[k] = int(movable_mask[k])
        if flash_ready:
            for k in range(8):
                if env_flash_mask[k] and flash_useful_mask[k]:
                    mask[8 + k] = 1
        if sum(mask) == 0 and flash_ready:
            for k in range(8):
                if flash_useful_mask[k]:
                    mask[8 + k] = 1
        if sum(mask) == 0:
            mask = [1] * 16
        return mask

    def _compute_reward(
        self,
        picked_treasure,
        picked_buff,
        used_flash,
        flash_path_had_item,
        monster_bfs_gain,
        bumped_wall,
        dead_end,
        late_factor,
        nearest_treasure_bfs,
        min_monster_bfs,
        first_seen_treasure_count=0,
        opening_treasure_shield=False,
        has_unreachable_treasure_in_view=False,
        is_lingering=False,
    ):
        r_treasure = _lerp(Config.R_TREASURE_EARLY, Config.R_TREASURE_LATE, late_factor) if picked_treasure else 0.0
        r_buff = _lerp(Config.R_BUFF_EARLY, Config.R_BUFF_LATE, late_factor) if picked_buff else 0.0
        r_survive = _lerp(Config.R_SURVIVE_EARLY, Config.R_SURVIVE_LATE, late_factor)
        r_first_seen = Config.R_FIRST_SEEN_TREASURE * float(first_seen_treasure_count)
        r_opening_blind_treasure = (
            Config.R_OPENING_BLIND_TREASURE if picked_treasure and opening_treasure_shield else 0.0
        )

        r_flash_path_item = 0.0
        r_flash_escape = 0.0
        r_flash_cost = 0.0
        r_flash_waste = 0.0
        if used_flash:
            if flash_path_had_item:
                r_flash_path_item = Config.R_FLASH_TO_REACH * (1.0 - late_factor)
            if monster_bfs_gain >= Config.FLASH_ESCAPE_MIN_GAIN:
                gain_scale = min(monster_bfs_gain / 4.0, 1.5)
                r_flash_escape = Config.R_FLASH_ESCAPE * late_factor * gain_scale
            r_flash_cost = Config.R_FLASH_COST_EARLY * (1.0 - late_factor)
            if late_factor > 0.5 and monster_bfs_gain < Config.FLASH_ESCAPE_MIN_GAIN and not flash_path_had_item:
                stage = Config.CURRICULUM_STAGE
                if stage == 3:
                    r_flash_waste = Config.R_FLASH_WASTE_STAGE3
                elif stage >= 4:
                    r_flash_waste = Config.R_FLASH_WASTE_STAGE4

        if bumped_wall and has_unreachable_treasure_in_view:
            r_bump = Config.R_BUMP_WALL_TOWARD_UNREACHABLE
        else:
            r_bump = Config.R_BUMP_WALL if bumped_wall else 0.0
        r_dead = Config.R_DEAD_END if dead_end else 0.0

        r_monster_proximity = 0.0
        if min_monster_bfs < Config.MONSTER_DANGER_THRESHOLD:
            ratio = 1.0 - min_monster_bfs / float(Config.MONSTER_DANGER_THRESHOLD)
            r_monster_proximity = Config.R_MONSTER_PROXIMITY_COEF * (ratio ** 2)

        r_linger = Config.R_LINGER if is_lingering else 0.0

        n = float(Config.BFS_MAX_STEPS)
        phi = (
            Config.POT_ALPHA_TREASURE * (1.0 - min(nearest_treasure_bfs / n, 1.0))
            + Config.POT_BETA_MONSTER * min(min_monster_bfs / n, 1.0)
        )
        r_pot = Config.GAMMA * phi - self.last_potential
        self.last_potential = phi

        return (
            r_treasure
            + r_buff
            + r_survive
            + r_first_seen
            + r_opening_blind_treasure
            + r_flash_path_item
            + r_flash_escape
            + r_flash_cost
            + r_flash_waste
            + r_bump
            + r_dead
            + r_monster_proximity
            + r_linger
            + r_pot
        )
