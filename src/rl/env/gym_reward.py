#!/usr/bin/env python3
"""Rewards for the SSBU gym environment.

Two modes (chosen by ``SSBUGymEnv(reward_mode=...)``):

``"none"`` (default)
    Always 0.0.  This is the BC/recording default: the dataset stores raw
    transitions and the reward is irrelevant.

``"combat"``
    The shaped fighting reward, controlled by :class:`RewardConfig`:

    * damage delta: ``damage_scale * (damage_dealt * dealt - damage_taken * taken)``
      where ``dealt``/``taken`` are positive damage increases on the opponent
      / on the ego fighter (damage resets at a KO therefore contribute 0, not
      a large negative);
    * KO/stock delta: ``ko_bonus`` for each opponent stock lost,
      ``stock_loss`` (negative by default) for each own stock lost;
    * match result: ``win_bonus`` when the opponent is eliminated (or the
      fight ends with more stocks), ``loss_penalty`` when the ego fighter is
      eliminated, ``draw_reward`` on a double KO/tie.

Damage is in percent, so the default ``damage_scale=0.01`` makes 100 % of
damage worth 1.0 reward.

Snapshot protocol: any object with ``fighters`` (a sequence of fighter dicts)
and ``fight_on`` works (the env passes :class:`gym_env.RawSnapshot`); tests can
use :class:`types.SimpleNamespace`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_EGO_SLOT = 0


@dataclass(frozen=True)
class RewardConfig:
    """Weights for ``reward_mode="combat"``."""

    damage_scale: float = 0.01      # percent of damage -> reward units
    damage_dealt: float = 1.0       # multiplier on damage dealt to opponent
    damage_taken: float = 1.0       # multiplier on damage taken
    ko_bonus: float = 1.0           # per opponent stock lost
    stock_loss: float = -1.0        # per own stock lost (negative penalty)
    win_bonus: float = 1.0
    loss_penalty: float = -1.0
    draw_reward: float = 0.0
    ego_slot: int = DEFAULT_EGO_SLOT
    #: Small per-decision penalty while the ego is offstage (not on the
    #: revival platform).  It discourages walking/drifting off the edge
    #: without matching the full stock penalty.
    offstage_penalty: float = 0.0
    #: Positive per-decision reward while the opponent is below the stage,
    #: mirroring ``offstage_penalty``.  This pays for edgeguarding: keeping the
    #: opponent offstage is progress, whether or not the KO lands.
    opponent_offstage_reward: float = 0.0
    #: Potential-based reward for reducing distance below the stage: moving
    #: upward/toward recovery gets a bonus, falling gets a matching penalty.
    recovery_weight: float = 0.0
    stage_half_width: float = 100.0
    stage_top: float = 0.0

    def as_dict(self) -> dict:
        return {
            "damage_scale": self.damage_scale,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "ko_bonus": self.ko_bonus,
            "stock_loss": self.stock_loss,
            "win_bonus": self.win_bonus,
            "loss_penalty": self.loss_penalty,
            "draw_reward": self.draw_reward,
            "ego_slot": self.ego_slot,
            "offstage_penalty": self.offstage_penalty,
            "opponent_offstage_reward": self.opponent_offstage_reward,
            "recovery_weight": self.recovery_weight,
            "stage_half_width": self.stage_half_width,
            "stage_top": self.stage_top,
        }


#: Short aliases accepted by ``SSBUGymEnv``.
REWARD_MODES = ("none", "combat")


def _fighters(snapshot: Any) -> Sequence[dict]:
    fighters = getattr(snapshot, "fighters", None)
    if fighters is None and isinstance(snapshot, dict):
        fighters = snapshot.get("fighters")
    if fighters is None:
        return ()
    return fighters


def _fight_on(snapshot: Any) -> bool:
    if snapshot is None:
        return False
    value = getattr(snapshot, "fight_on", None)
    if value is None and isinstance(snapshot, dict):
        value = snapshot.get("fight_on")
    return bool(value)


def _valid(fighter: dict) -> bool:
    return any(bool(v) for k, v in fighter.items() if k != "raw_flags") or bool(
        fighter.get("raw_flags", 0))


def _number(fighter: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(fighter.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value == value and abs(value) != float("inf") else default


def _stock(fighter: dict) -> int:
    try:
        return int(fighter.get("stock", 0))
    except (TypeError, ValueError):
        return 0


def _rebirth(fighter: dict) -> bool:
    """True while the fighter is on the revival platform."""
    try:
        return int(fighter.get("is_on_rebirth", 0)) == 1
    except (TypeError, ValueError):
        return False


def _edge_danger(fighter: dict, cfg: "RewardConfig") -> float:
    """0..1 danger factor for falling *below* the stage.

    Horizontal edge play is deliberately not penalized (chasing or recovering
    at the side is legitimate).  The penalty ramps in only once the fighter is
    below the stage surface, and maxes out well below it.
    """
    if _rebirth(fighter):
        return 0.0
    try:
        airborne = int(fighter.get("on_ground", 0)) == 0
    except (TypeError, ValueError):
        airborne = False
    if not airborne:
        return 0.0
    y = _number(fighter, "y")
    below = max(0.0, (float(cfg.stage_top) - 10.0) - y)
    return min(1.0, below / 40.0)


def _recovery_distance(fighter: dict, cfg: "RewardConfig") -> float:
    """How far below the stage surface the fighter is (0 on stage)."""
    if _rebirth(fighter):
        return 0.0
    try:
        if int(fighter.get("on_ground", 0)) == 1:
            return 0.0
    except (TypeError, ValueError):
        pass
    y = _number(fighter, "y")
    return max(0.0, (float(cfg.stage_top) - 10.0) - y)


def compute_reward(prev: Any, curr: Any, config: RewardConfig | None = None,
                   ego_slot: int | None = None) -> float:
    """Combat reward for the transition ``prev -> curr``.

    ``prev``/``curr`` are snapshot-like objects.  Missing/invalid fighters or
    snapshots yield 0.0 instead of a spurious KO/win.
    """
    cfg = config or RewardConfig()
    ego = cfg.ego_slot if ego_slot is None else int(ego_slot)
    if ego not in (0, 1):
        raise ValueError(f"ego_slot must be 0 or 1, got {ego}")
    opp = 1 - ego

    if prev is None or curr is None:
        return 0.0
    prev_f = _fighters(prev)
    curr_f = _fighters(curr)
    if len(prev_f) <= max(ego, opp) or len(curr_f) <= max(ego, opp):
        return 0.0

    pf_ego, pf_opp = prev_f[ego], prev_f[opp]
    cf_ego, cf_opp = curr_f[ego], curr_f[opp]
    if not (_valid(pf_ego) and _valid(pf_opp) and _valid(cf_ego) and _valid(cf_opp)):
        return 0.0

    # damage deltas (clamped at 0: a damage reset after a KO is not "healing")
    dealt = max(0.0, _number(cf_opp, "damage") - _number(pf_opp, "damage"))
    taken = max(0.0, _number(cf_ego, "damage") - _number(pf_ego, "damage"))
    reward = cfg.damage_scale * (cfg.damage_dealt * dealt
                                 - cfg.damage_taken * taken)
    danger = _edge_danger(cf_ego, cfg)
    if cfg.offstage_penalty and danger > 0.0:
        reward -= float(cfg.offstage_penalty) * danger
    # Mirror of the ego offstage penalty: the opponent being below the stage is
    # good for the ego, so edgeguarding pays even before the KO.
    opp_danger = _edge_danger(cf_opp, cfg)
    if cfg.opponent_offstage_reward and opp_danger > 0.0:
        reward += float(cfg.opponent_offstage_reward) * opp_danger
    if cfg.recovery_weight and not _rebirth(cf_ego):
        reward += float(cfg.recovery_weight) * (
            _recovery_distance(pf_ego, cfg) - _recovery_distance(cf_ego, cfg))

    # stock / KO deltas
    prev_stock_ego, stock_ego = _stock(pf_ego), _stock(cf_ego)
    prev_stock_opp, stock_opp = _stock(pf_opp), _stock(cf_opp)
    lost_opp = max(0, prev_stock_opp - stock_opp)
    lost_ego = max(0, prev_stock_ego - stock_ego)
    # The live build often reports stock==0 for both slots, so the stock delta
    # cannot see a KO.  A 0->1 rebirth transition is a direct, per-fighter
    # death/respawn signal; count the stock once (max avoids double counting
    # when the stock field is readable as well).
    if not _rebirth(pf_opp) and _rebirth(cf_opp):
        lost_opp = max(lost_opp, 1)
    if not _rebirth(pf_ego) and _rebirth(cf_ego):
        lost_ego = max(lost_ego, 1)
    reward += cfg.ko_bonus * lost_opp
    reward += cfg.stock_loss * lost_ego

    # match result.  A stock hitting zero is a win/loss in the 1-stock rules
    # even if fight_on has not flipped yet; only count it once.
    opp_eliminated = prev_stock_opp > 0 and stock_opp <= 0
    ego_eliminated = prev_stock_ego > 0 and stock_ego <= 0
    ended = _fight_on(prev) and not _fight_on(curr)

    if opp_eliminated and ego_eliminated:
        reward += cfg.draw_reward
    elif opp_eliminated:
        reward += cfg.win_bonus
    elif ego_eliminated:
        reward += cfg.loss_penalty
    elif ended:
        if stock_ego > stock_opp:
            reward += cfg.win_bonus
        elif stock_ego < stock_opp:
            reward += cfg.loss_penalty
        else:
            reward += cfg.draw_reward

    return float(reward)
