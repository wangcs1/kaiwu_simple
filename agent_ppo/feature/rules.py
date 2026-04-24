#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Rule override layer for Gorge Chase PPO.
"""

from agent_ppo.conf.conf import Config


def adjacent_treasure_action(rule_hints):
    d = rule_hints.get("adjacent_treasure_dir", -1)
    if d is not None and 0 <= d < 8:
        return int(d)
    return None


def _pick_safe_flash(
    flash_useful_mask, flash_landings, monster_dirs, flash_ready, flash_escape_scores=None
):
    if not flash_ready:
        return None
    useful_dirs = [k for k in range(8) if flash_useful_mask[k]]
    if not useful_dirs:
        return None

    if flash_escape_scores is None:
        flash_escape_scores = [0] * 8

    m_dir = -1
    for d in monster_dirs:
        if d is not None and 0 <= d <= 7:
            m_dir = d
            break

    if m_dir == -1:
        best = max(
            useful_dirs,
            key=lambda k: (
                flash_escape_scores[k],
                abs(flash_landings[k][0]) + abs(flash_landings[k][1]),
            ),
        )
        return 8 + best

    def angle_diff(a, b):
        d = abs(a - b) % 8
        return min(d, 8 - d)

    best = max(
        useful_dirs,
        key=lambda k: (
            flash_escape_scores[k],
            angle_diff(k, m_dir),
            abs(flash_landings[k][0]) + abs(flash_landings[k][1]),
        ),
    )
    return 8 + best


def panic_flash_action(rule_hints):
    if not Config.RULE_PANIC_FLASH:
        return None
    if not rule_hints.get("flash_ready", False):
        return None
    dists = rule_hints.get("monster_bfs_dists", [])
    if not dists:
        return None
    thr = (
        Config.PANIC_FLASH_THRESHOLD_LATE
        if rule_hints.get("is_late_phase", False)
        else Config.PANIC_FLASH_THRESHOLD_EARLY
    )
    if min(dists) > thr:
        return None
    return _pick_safe_flash(
        rule_hints["flash_useful_mask"],
        rule_hints["flash_landings"],
        rule_hints.get("monster_dirs", []),
        rule_hints["flash_ready"],
        rule_hints.get("flash_escape_scores", []),
    )


def apply_rule_override(rl_action, legal_action_16d, rule_hints):
    if rl_action < 0 or rl_action >= 16 or legal_action_16d[rl_action] == 0:
        for k, v in enumerate(legal_action_16d):
            if v == 1:
                rl_action = k
                break

    if Config.RULE_ADJACENT_TREASURE and not rule_hints.get("opening_treasure_shield", False):
        a = adjacent_treasure_action(rule_hints)
        if a is not None and legal_action_16d[a] == 1:
            return a, True

    a = panic_flash_action(rule_hints)
    if a is not None and legal_action_16d[a] == 1:
        return a, True

    if rule_hints.get("is_late_phase", False) and rule_hints.get("flash_ready", False):
        dists = rule_hints.get("monster_bfs_dists", [])
        scores = rule_hints.get("flash_escape_scores", [])
        if dists and scores and min(dists) <= Config.CHASE_FLASH_THRESHOLD_LATE:
            best_dir = max(range(8), key=lambda k: scores[k])
            if (scores[best_dir] - min(dists)) >= Config.FLASH_ESCAPE_MIN_GAIN:
                a = 8 + best_dir
                if legal_action_16d[a] == 1:
                    return a, True

    return rl_action, False
