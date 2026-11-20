#!/usr/bin/env python3
"""PPO fine-tuning of the SSBU behaviour-cloning policy.

The BC checkpoint's joint-action token head is treated as a categorical policy
over the same vocabulary (including the repeat token).  One token is chosen per
decision, decoded to a pad action, and the live ``SSBUGymEnv`` provides the
damage-differential reward.  A scalar value head is added to the BC GRU/MLP
trunk and initialized from scratch.

    # boot the game and walk the menus
    src/rl/scripts/live_rl.sh --model data/models/bc_dk9_30hz_vocab_egoswap.pt \
        --out data/models/rl_dk9_ppo.pt --iterations 10

    # plumbing smoke test, no game
    .venv/bin/python -m rl.rl.ppo --model ... --mock --iterations 1 \
        --rollout-steps 32 --minibatch 16
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from rl.bc import gym_bc as bc          # noqa: E402
from rl.bc import gym_eval as ge       # noqa: E402


# ---------------------------------------------------------------------------
# config / checkpoint loading
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    iterations: int = 10
    rollout_steps: int = 2048
    max_steps: int = 900
    epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 0.5
    kl_coef: float = 0.0
    #: Per-sample loss multiplier for offstage transitions (1.0 = uniform).
    #: Offstage states are rare in a 3600-step rollout; weighting them keeps the
    #: critic from ignoring the states where deaths actually happen.
    offstage_loss_weight: float = 1.0
    max_grad_norm: float = 0.5
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"

    def check(self) -> None:
        for name in ("iterations", "rollout_steps", "max_steps", "epochs",
                     "minibatch"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 <= float(self.clip) < 1.0:
            raise ValueError("clip must be in [0, 1)")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= float(self.lam) <= 1.0:
            raise ValueError("lam must be in [0, 1]")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be > 0")
        if float(self.entropy) < 0.0:
            raise ValueError("entropy must be >= 0")
        if float(self.value_coef) < 0.0:
            raise ValueError("value_coef must be >= 0")
        if float(self.kl_coef) < 0.0:
            raise ValueError("kl_coef must be >= 0")


class TrainingStats:
    """Small JSON status file the live viewer reads (bottom-left overlay)."""

    def __init__(self, path: str, cpu_level: int, iterations: int):
        self.path = os.path.expanduser(path)
        self.data = {
            "active": True,
            "cpu_level": int(cpu_level),
            "iteration": 0,
            "iterations": int(iterations),
            "episodes": 0,
            "damage_episodes": 0,
            "wins": 0,
            "losses": 0,
            "positive": 0,
            "p1_deaths": 0,
            "p2_deaths": 0,
            "current_episode": 0,
            "current_steps": 0,
            "current_max_steps": 0,
            "current_dealt": 0.0,
            "current_taken": 0.0,
            "current_diff": 0.0,
            "current_return": 0.0,
            "current_p1_deaths": 0,
            "current_p2_deaths": 0,
            "sum_dealt": 0.0,
            "sum_taken": 0.0,
            "sum_diff": 0.0,
            "sum_reward": 0.0,
            "reward_episodes": 0,
            "reward_positive": 0,
            "last_dealt": 0.0,
            "last_taken": 0.0,
            "last_diff": 0.0,
            "last_reward": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_bc": 0.0,
            "updated": time.time(),
        }
        self.last_beat = time.time()
        self.flush()

    def _derived(self) -> None:
        n = max(1, int(self.data["damage_episodes"]))
        eps = max(1, int(self.data["episodes"]))
        self.data["mean_dealt"] = self.data["sum_dealt"] / n
        self.data["mean_taken"] = self.data["sum_taken"] / n
        self.data["mean_diff"] = self.data["sum_diff"] / n
        rn = max(1, int(self.data["reward_episodes"]))
        self.data["mean_reward"] = self.data["sum_reward"] / rn
        self.data["reward_positive_rate"] = self.data["reward_positive"] / eps
        self.data["positive_rate"] = self.data["positive"] / eps
        self.data["win_rate"] = self.data["wins"] / eps
        self.data["updated"] = time.time()

    def flush(self) -> None:
        self._derived()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".",
                        exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp, self.path)
        except (OSError, TypeError, ValueError):
            pass

    def record_step(self, progress: dict | None = None) -> None:
        """Live current-episode stats; throttle file writes to ~2 Hz."""
        if progress:
            self.data.update({
                "current_episode": int(progress.get("episode", 0) or 0),
                "current_steps": int(progress.get("steps", 0) or 0),
                "current_max_steps": int(progress.get("max_steps", 0) or 0),
                "current_dealt": float(progress.get("dealt", 0.0) or 0.0),
                "current_taken": float(progress.get("taken", 0.0) or 0.0),
                "current_diff": float(progress.get("diff", 0.0) or 0.0),
                "current_return": float(progress.get("return", 0.0) or 0.0),
                "current_p1_deaths": int(progress.get("p1_deaths", 0) or 0),
                "current_p2_deaths": int(progress.get("p2_deaths", 0) or 0),
            })
        now = time.time()
        if now - self.last_beat >= 0.5:
            self.last_beat = now
            self.data["updated"] = now
            self.flush()

    def record_episode(self, row: dict) -> None:
        self.data["episodes"] += 1
        self.data["p1_deaths"] += int(row.get("p1_deaths", 0) or 0)
        self.data["p2_deaths"] += int(row.get("p2_deaths", 0) or 0)
        self.data.update({
            "current_episode": 0,
            "current_steps": 0,
            "current_max_steps": 0,
            "current_dealt": 0.0,
            "current_taken": 0.0,
            "current_diff": 0.0,
            "current_return": 0.0,
            "current_p1_deaths": 0,
            "current_p2_deaths": 0,
        })
        ret = row.get("return")
        if ret is not None:
            ret = float(ret)
            self.data["reward_episodes"] += 1
            self.data["sum_reward"] += ret
            self.data["last_reward"] = ret
            if ret > 0.0:
                self.data["reward_positive"] += 1
        p1, p2 = row.get("p1_damage"), row.get("p2_damage")
        if p1 is not None and p2 is not None:
            dealt, taken = float(p2), float(p1)
            diff = dealt - taken
            self.data["damage_episodes"] += 1
            self.data["sum_dealt"] += dealt
            self.data["sum_taken"] += taken
            self.data["sum_diff"] += diff
            self.data["last_dealt"] = dealt
            self.data["last_taken"] = taken
            self.data["last_diff"] = diff
            if diff > 0.0:
                self.data["positive"] += 1
        winner = row.get("winner")
        if winner == 0:
            self.data["wins"] += 1
        elif winner == 1:
            self.data["losses"] += 1
        self.flush()

    def record_iteration(self, iteration: int, stats: dict) -> None:
        self.data["iteration"] = int(iteration)
        for key in ("policy_loss", "value_loss", "entropy", "kl_bc"):
            if key in stats:
                self.data[key] = float(stats[key])
        self.flush()

    def finish(self) -> None:
        self.data["active"] = False
        self.flush()


def load_bc_model(path: str, device: str,
                  dropout: float = 0.0) -> tuple[Any, Any, Any, Any, Any, Any]:
    """BC checkpoint -> model with a fresh value head, vocab, norm stats."""
    ckpt = bc.load_checkpoint(path)
    raw_cfg = {k: v for k, v in ckpt.get("config", {}).items()
               if k in bc.BCConfig.__dataclass_fields__}
    cfg = bc.BCConfig(**raw_cfg)
    if cfg.action_mode != "vocab":
        raise ValueError("PPO fine-tuning currently supports action_mode=vocab "
                         "(the BC token head is the policy)")
    cfg = replace(cfg, value_head=True, dropout=float(dropout))
    model = bc.BCPolicyNet(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    allowed = {"value_head.weight", "value_head.bias"}
    if set(missing) - allowed or unexpected:
        raise ValueError(
            "checkpoint does not match the BC architecture; missing "
            f"{sorted(set(missing) - allowed)}, unexpected {sorted(unexpected)}")
    # Start a *fresh* critic at zero: PPO learns the value scale from the
    # reward instead of trusting random initial outputs.  A checkpoint that
    # already carries a trained critic (an earlier PPO run) must keep it;
    # zeroing it made every chunk restart update the policy against V=0.
    has_critic = not (set(missing) & allowed)
    if model.value_head is not None and not has_critic:
        torch.nn.init.zeros_(model.value_head.weight)
        torch.nn.init.zeros_(model.value_head.bias)
    payload = ckpt.get("vocab")
    if not payload:
        raise ValueError("checkpoint has no action vocabulary")
    vocab = bc.ActionVocab.from_dict(payload)
    mean = torch.as_tensor(ckpt["norm"]["mean"], dtype=torch.float32,
                           device=device)
    std = torch.as_tensor(ckpt["norm"]["std"], dtype=torch.float32,
                          device=device)
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("checkpoint normalization stats are non-finite")
    model.eval()
    return model, cfg, vocab, mean, std, ckpt


# ---------------------------------------------------------------------------
# policy state / action decoding
# ---------------------------------------------------------------------------
class PolicyState:
    """The window history and previous action vector for one live episode."""

    def __init__(self, cfg: bc.BCConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.hist: deque[np.ndarray] = deque(maxlen=self.cfg.window)
        self.prev = np.zeros(self.cfg.prev_dim, np.float32)

    def observe(self, obs) -> None:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape[0] != self.cfg.obs_dim:
            raise ValueError(f"observation has {o.shape[0]} values; model "
                             f"expects {self.cfg.obs_dim}")
        self.hist.append(o)

    def window(self) -> np.ndarray:
        if not self.hist:
            raise RuntimeError("observe() before window()")
        items = list(self.hist)
        while len(items) < self.cfg.window:
            items.insert(0, items[0])
        return np.stack(items).astype(np.float32)

    def set_prev(self, action: dict) -> None:
        self.prev = bc.action_vector(action, self.cfg)


def token_action(token: int, prev: np.ndarray, vocab: bc.ActionVocab,
                 cfg: bc.BCConfig) -> dict:
    """Token -> action dict; the repeat token re-emits ``prev``."""
    if int(token) == bc.REPEAT_TOKEN:
        buttons = (np.asarray(prev[:bc.N_BUTTONS]) >= 0.5).astype(np.float32)
        stick = np.asarray(prev[bc.N_BUTTONS:bc.N_BUTTONS + 2],
                           np.float32).copy()
        off = bc.N_BUTTONS + 2
        cstick = (np.asarray(prev[off:off + 2], np.float32).copy()
                  if cfg.cstick else np.zeros(2, np.float32))
        if cfg.cstick:
            off += 2
        trigger = (float(prev[off]) if cfg.trigger and len(prev) > off
                   else 0.0)
    else:
        vec = vocab.decode(int(token))
        buttons = (vec[:bc.N_BUTTONS] >= 0.5).astype(np.float32)
        stick = vec[bc.N_BUTTONS:bc.N_BUTTONS + 2].copy()
        cstick = vec[bc.N_BUTTONS + 2:bc.N_BUTTONS + 4].copy()
        trigger = float(vec[bc.N_BUTTONS + 4])
    return {"buttons": buttons, "stick": stick, "cstick": cstick,
            "trigger": trigger}


def act_token(model, state: PolicyState, vocab: bc.ActionVocab,
              mean: torch.Tensor, std: torch.Tensor, device: str,
              sample: bool = True) -> tuple[dict, int, float, float]:
    """Sample one token from the current BC policy and update state.prev."""
    w = torch.from_numpy(state.window()).to(device)
    w = ((w - mean) / std).unsqueeze(0)
    p = torch.from_numpy(state.prev).to(device).unsqueeze(0)
    with torch.no_grad():
        out = model(w, p)
        logits = out["action"][0, 0]
        dist = Categorical(logits=logits)
        token = dist.sample() if sample else torch.argmax(logits)
        logp = dist.log_prob(token)
        value = out["value"][0]
    token = int(token.item())
    action = token_action(token, state.prev, vocab, model.cfg)
    state.set_prev(action)
    return action, token, float(logp.item()), float(value.item())


# ---------------------------------------------------------------------------
# rollout / PPO
# ---------------------------------------------------------------------------
class MockCombatEnv(ge.MockEnv):
    """Small no-game env with the same damage-differential reward."""

    def step(self, action):
        before = list(self.damage)
        obs, _reward, terminated, truncated, info = super().step(action)
        reward = 0.01 * ((self.damage[1] - before[1]) -
                         (self.damage[0] - before[0]))
        return obs, reward, terminated, truncated, info


def _damages(info: dict) -> list[float]:
    """[P1_damage, P2_damage] from an env info dict."""
    out = []
    for fighter in (info.get("raw_fighters") or [])[:2]:
        try:
            out.append(float(fighter.get("damage", 0.0) or 0.0))
        except (TypeError, ValueError, AttributeError):
            out.append(0.0)
    if len(out) < 2:
        out = [float(info.get("p1_damage", 0.0) or 0.0),
               float(info.get("p2_damage", 0.0) or 0.0)]
    return out


def _offstage_flag(info: dict, slot: int = 0) -> bool:
    """True when the ego fighter is below the stage, airborne, not respawning.

    Used only to weight the PPO loss toward the rare states where falls happen;
    the reward itself is computed by the environment.
    """
    fighters = info.get("raw_fighters") or []
    if len(fighters) <= int(slot):
        return False
    fighter = fighters[int(slot)]
    try:
        if int(fighter.get("is_on_rebirth", 0) or 0) == 1:
            return False
        if int(fighter.get("on_ground", 0) or 0) == 1:
            return False
        from rl.memory import status_kinds
        if status_kinds.is_ledge(fighter.get("action_state", -1)):
            return False                  # hanging on the ledge is safe
        return float(fighter.get("y", 0.0) or 0.0) < -10.0
    except (TypeError, ValueError, AttributeError):
        return False


def _ego_snapshot(info: dict, slot: int = 0) -> dict:
    """The ego fields the death autopsy needs, from an env info dict."""
    fighters = info.get("raw_fighters") or []
    if len(fighters) <= int(slot):
        return {}
    f = fighters[int(slot)]

    def num(key, default=0.0):
        try:
            return float(f.get(key, default) or default)
        except (TypeError, ValueError):
            return default
    return {"damage": num("damage"), "x": num("x"), "y": num("y"),
            "vx": num("vx"), "vy": num("vy"),
            "state": int(num("action_state", -1)),
            "jumps": int(num("jumps_left")),
            "ground": int(num("on_ground")),
            "hitstun": num("hitstun")}


def classify_death(history: list[dict]) -> dict:
    """Autopsy of one P1 death from the last ego snapshots (oldest first).

    ``history[-1]`` is the frame before the rebirth flag turned on, so its
    damage and position are the ones the fighter died with.  The cause is a
    coarse label, enough to tell whether recovery training can help:

    * ``blast_side/top``: launched out past a side/top blast zone, usually
      at high damage; recovery cannot fix it, damage avoidance can.
    * ``helpless``: died in the helpless fall after Up Special (state 16).
    * ``no_return``: fell from the ledge without a helpless state.
    """
    if not history:
        return {"cause": "unknown"}
    last = history[-1]
    from rl.memory import status_kinds
    helpless = any(h.get("state") in status_kinds.HELPLESS_KINDS
                   for h in history[-90:])
    upb_used = any(h.get("state") == status_kinds.SPECIAL_HI
                   for h in history[-240:])
    ledge_grabbed = any(h.get("state") in status_kinds.LEDGE_KINDS
                        for h in history[-240:])
    jumps_min = min(h.get("jumps", 0) for h in history[-90:])
    below = [h for h in history if h.get("y", 0.0) < -10.0
             and not h.get("ground")]
    # steps spent airborne below the stage before the death frame
    fall_steps = 0
    for h in reversed(history):
        if h.get("y", 0.0) < -10.0 and not h.get("ground"):
            fall_steps += 1
        else:
            break
    hit_recent = any(b["damage"] > a["damage"] + 0.5
                     for a, b in zip(history[-60:], history[-59:]))
    x, y = last.get("x", 0.0), last.get("y", 0.0)
    if y < -10.0:
        cause = "helpless" if helpless else "no_return"
    elif y > 80.0:
        cause = "blast_top"
    elif abs(x) > 150.0:
        cause = "blast_side"
    else:
        cause = "other"
    return {"cause": cause, "damage": round(last.get("damage", 0.0), 1),
            "x": round(x, 1), "y": round(y, 1), "helpless": bool(helpless),
            "jumps_min": int(jumps_min), "fall_steps": int(fall_steps),
            "upb_used": bool(upb_used), "ledge_grabbed": bool(ledge_grabbed),
            "hit_recent": bool(hit_recent), "offstage": bool(below)}


def _rebirth_flags(info: dict) -> list[bool]:
    """[P1_on_rebirth, P2_on_rebirth] from an env info dict."""
    out = []
    for fighter in (info.get("raw_fighters") or [])[:2]:
        try:
            out.append(int(fighter.get("is_on_rebirth", 0) or 0) == 1)
        except (TypeError, ValueError, AttributeError):
            out.append(False)
    while len(out) < 2:
        out.append(False)
    return out


def _episode_row(info: dict, length: int, ret: float,
                 terminated: bool, truncated: bool,
                 peak: dict | None = None,
                 deaths: dict | None = None,
                 totals: dict | None = None,
                 death_rewards: list | None = None,
                 death_infos: list | None = None) -> dict:
    """Summarize one episode.

    The last info frame can be a respawn with damage reset to 0, so the final
    values alone can report a false 0/0.  ``peak`` keeps the largest damage
    seen during the episode, matching gym_eval's reporting.
    """
    r = ge._result_from_info(info)
    p1, p2 = r.get("p1_damage"), r.get("p2_damage")
    winner = r.get("winner")
    if totals:
        # Cumulative damage actually applied during the episode (damage
        # resets after a KO do not erase it).  This is the useful value for
        # training averages; gym_eval still uses peak damage.
        p1 = float(totals.get("taken", p1 or 0.0))
        p2 = float(totals.get("dealt", p2 or 0.0))
    elif peak:
        if p1 in (None, 0.0) and peak.get("p1_damage", 0.0) > 0.0:
            p1 = peak["p1_damage"]
        if p2 in (None, 0.0) and peak.get("p2_damage", 0.0) > 0.0:
            p2 = peak["p2_damage"]
    deaths = deaths or {}
    return {"length": int(length), "return": float(ret),
            "p1_damage": p1, "p2_damage": p2, "winner": winner,
            "p1_deaths": int(deaths.get("p1", 0)),
            "p2_deaths": int(deaths.get("p2", 0)),
            "death_rewards": [float(x) for x in (death_rewards or [])],
            "death_infos": list(death_infos or []),
            "terminated": bool(terminated), "truncated": bool(truncated)}


def prime_state(state: PolicyState, env, obs) -> None:
    """Reset the policy history, replaying a drill setup if present."""
    state.reset()
    for previous in getattr(env, "setup_observations", None) or []:
        try:
            state.observe(ge._extract_obs(previous))
        except Exception:                     # noqa: BLE001
            break
    setup_action = getattr(env, "setup_action", None)
    if setup_action is not None:
        try:
            state.set_prev(setup_action)
        except Exception:                     # noqa: BLE001
            pass
    state.observe(ge._extract_obs(obs))


def collect_rollout(env, model, state: PolicyState, cfg: bc.BCConfig,
                    vocab: bc.ActionVocab, mean, std, device: str,
                    steps: int, seed: int,
                    on_episode=None, on_step=None,
                    on_reset_error=None) -> tuple[dict, list[dict]]:
    """Run the live env, returning flat transition arrays and episode rows."""
    model.eval()
    windows, prevs, actions, logps, values, rewards = [], [], [], [], [], []
    next_windows, next_prevs, terminals, truncateds = [], [], [], []
    offstage_flags: list[bool] = []
    episodes: list[dict] = []

    def do_reset(reset_seed: int):
        try:
            return ge.reset_env(env, reset_seed)
        except Exception as exc:                  # noqa: BLE001
            if on_reset_error is None:
                raise
            return on_reset_error(reset_seed, exc)

    obs, info = do_reset(seed)
    prime_state(state, env, obs)
    ep = 0
    ep_ret = 0.0
    ep_len = 0
    ep_dealt = 0.0
    ep_taken = 0.0
    max_steps = int(getattr(env, "max_steps", 0) or 0)
    peak = {"p1_damage": 0.0, "p2_damage": 0.0}
    prev_rebirth = _rebirth_flags(info)
    prev_damages = _damages(info)
    deaths = {"p1": 0, "p2": 0}
    death_rewards: list[float] = []
    death_infos: list[dict] = []
    ego_hist: deque = deque(maxlen=240)
    ego_slot_id = int(getattr(env, "ego_slot", 0) or 0)
    ego_hist.append(_ego_snapshot(info, ego_slot_id))
    while len(rewards) < steps:
        window = state.window().copy()
        prev = state.prev.copy()
        action, token, logp, value = act_token(
            model, state, vocab, mean, std, device, sample=True)
        env_action = ge.format_action(action, "auto", env)
        next_obs, reward, terminated, truncated, info = ge.step_env(
            env, env_action)
        next_obs = ge._extract_obs(next_obs)
        state.observe(next_obs)
        windows.append(window)
        prevs.append(prev)
        actions.append(token)
        logps.append(logp)
        values.append(value)
        rewards.append(float(reward))
        next_windows.append(state.window().copy())
        next_prevs.append(state.prev.copy())
        terminals.append(bool(terminated))
        truncateds.append(bool(truncated))
        offstage_flags.append(_offstage_flag(
            info, int(getattr(env, "ego_slot", 0) or 0)))
        ep_ret += float(reward)
        ep_len += 1
        step_result = ge._result_from_info(info)
        for key in peak:
            value = step_result.get(key)
            if value is not None:
                peak[key] = max(peak[key], float(value))
        curr_rebirth = _rebirth_flags(info)
        for i, key in enumerate(("p1", "p2")):
            if curr_rebirth[i] and not prev_rebirth[i]:
                deaths[key] += 1
                if key == "p1":
                    death_rewards.append(float(reward))
                    # history still ends at the frame before the rebirth
                    death_infos.append(classify_death(list(ego_hist)))
                else:
                    death_rewards.append(float(reward))
        ego_hist.append(_ego_snapshot(info, ego_slot_id))
        prev_rebirth = curr_rebirth
        curr_damages = _damages(info)
        ep_taken += max(0.0, curr_damages[0] - prev_damages[0])
        ep_dealt += max(0.0, curr_damages[1] - prev_damages[1])
        prev_damages = curr_damages
        if on_step is not None:
            on_step({
                "episode": ep + 1,
                "steps": ep_len,
                "max_steps": max_steps,
                "dealt": ep_dealt,
                "taken": ep_taken,
                "diff": ep_dealt - ep_taken,
                "return": ep_ret,
                "p1_deaths": deaths["p1"],
                "p2_deaths": deaths["p2"],
            })
        if terminated or truncated:
            row = _episode_row(
                info, ep_len, ep_ret, terminated, truncated, peak, deaths,
                totals={"dealt": ep_dealt, "taken": ep_taken},
                death_rewards=death_rewards, death_infos=death_infos)
            episodes.append(row)
            if on_episode is not None:
                on_episode(row)
            ep += 1
            obs, info = do_reset(seed + ep)
            prime_state(state, env, obs)
            ep_ret = 0.0
            ep_len = 0
            ep_dealt = 0.0
            ep_taken = 0.0
            peak = {"p1_damage": 0.0, "p2_damage": 0.0}
            prev_rebirth = _rebirth_flags(info)
            prev_damages = _damages(info)
            deaths = {"p1": 0, "p2": 0}
            death_rewards = []
            death_infos = []
            ego_hist.clear()
            ego_hist.append(_ego_snapshot(info, ego_slot_id))
    batch = {
        "windows": np.stack(windows).astype(np.float32),
        "prevs": np.stack(prevs).astype(np.float32),
        "actions": np.asarray(actions, np.int64),
        "logps": np.asarray(logps, np.float32),
        "values": np.asarray(values, np.float32),
        "rewards": np.asarray(rewards, np.float32),
        "next_windows": np.stack(next_windows).astype(np.float32),
        "next_prevs": np.stack(next_prevs).astype(np.float32),
        "terminals": np.asarray(terminals, np.bool_),
        "truncateds": np.asarray(truncateds, np.bool_),
        "offstage": np.asarray(offstage_flags, np.bool_),
    }
    return batch, episodes


def compute_gae(rewards, values, next_values, terminals, truncateds,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimates.

    A terminal transition does not bootstrap.  A truncated one (the step
    limit) bootstraps from the next state's value, but its GAE chain stops so
    the next episode's returns cannot leak backwards across the reset.
    """
    rewards = np.asarray(rewards, np.float32)
    values = np.asarray(values, np.float32)
    next_values = np.asarray(next_values, np.float32)
    terminals = np.asarray(terminals, bool)
    truncateds = np.asarray(truncateds, bool)
    boundaries = terminals | truncateds
    adv = np.zeros(len(rewards), np.float32)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if terminals[t] else 1.0
        chain = 0.0 if boundaries[t] else 1.0
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        last = delta + gamma * lam * chain * last
        adv[t] = last
    return adv, adv + values


def batch_values(model, windows, prevs, mean, std, device: str,
                 batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            w = torch.from_numpy(windows[start:start + batch_size]).to(device)
            w = (w - mean) / std
            p = torch.from_numpy(prevs[start:start + batch_size]).to(device)
            out.append(model(w, p)["value"].cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def batch_logits(model, windows, prevs, mean, std, device: str,
                 batch_size: int = 128) -> np.ndarray:
    """First-horizon token logits for a rollout, used by the BC KL anchor."""
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            w = torch.from_numpy(windows[start:start + batch_size]).to(device)
            w = (w - mean) / std
            p = torch.from_numpy(prevs[start:start + batch_size]).to(device)
            out.append(model(w, p)["action"][:, 0, :].cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros(
        (0, 0), np.float32)


def ppo_update(model, optimizer, batch: dict, cfg: PPOConfig,
               mean, std, device: str) -> dict:
    """A few PPO epochs over one rollout."""
    model.train()
    n = len(batch["actions"])
    clip = float(cfg.clip)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "kl_bc": 0.0}
    seen = 0
    ref_logits = batch.get("ref_logits")
    kl_coef = float(cfg.kl_coef)
    for _epoch in range(int(cfg.epochs)):
        order = np.random.permutation(n)
        for start in range(0, n, int(cfg.minibatch)):
            idx = order[start:start + int(cfg.minibatch)]
            w = torch.from_numpy(batch["windows"][idx]).to(device)
            w = (w - mean) / std
            p = torch.from_numpy(batch["prevs"][idx]).to(device)
            actions = torch.from_numpy(batch["actions"][idx]).to(device)
            old_logp = torch.from_numpy(batch["logps"][idx]).to(device)
            adv = torch.from_numpy(batch["advantages"][idx]).to(device)
            returns = torch.from_numpy(batch["returns"][idx]).to(device)
            out = model(w, p)
            dist = Categorical(logits=out["action"][:, 0, :])
            logp = dist.log_prob(actions)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
            # Offstage transitions are rare but decisive.  Up-weight them so
            # the critic does not fit only the common on-stage states.
            weights = torch.ones_like(adv)
            if batch.get("offstage") is not None:
                off = torch.from_numpy(
                    np.asarray(batch["offstage"])[idx].astype(np.float32)
                ).to(device)
                weights = 1.0 + (float(cfg.offstage_loss_weight) - 1.0) * off
                weights = weights / weights.mean()
            policy_loss = (-torch.min(surr1, surr2) * weights).mean()
            # Huber, not MSE: stock/death returns are sparse and large (one
            # -5.0 among thousands of near-zero steps).  Squared error on those
            # outliers produced a value loss of ~6 and corrupted the shared
            # trunk, taking the policy with it.
            value_loss = (F.smooth_l1_loss(
                out["value"], returns, reduction="none") * weights).mean()
            entropy = dist.entropy().mean()
            if ref_logits is not None and kl_coef > 0.0:
                ref_dist = Categorical(
                    logits=torch.from_numpy(ref_logits[idx]).to(device))
                kl = torch.distributions.kl_divergence(ref_dist, dist).mean()
            else:
                kl = torch.zeros((), device=device)
            loss = (policy_loss + float(cfg.value_coef) * value_loss
                    - float(cfg.entropy) * entropy + kl_coef * kl)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(cfg.max_grad_norm))
            optimizer.step()
            m = len(idx)
            stats["policy_loss"] += float(policy_loss.detach()) * m
            stats["value_loss"] += float(value_loss.detach()) * m
            stats["entropy"] += float(entropy.detach()) * m
            stats["kl_bc"] += float(kl.detach()) * m
            seen += m
    model.eval()
    return {k: v / max(1, seen) for k, v in stats.items()}


def dump_wedge_diagnostics(tag: str, log) -> str | None:
    """Save what the runtime looked like when a reset was never acknowledged."""
    import json
    import subprocess
    path = f"/tmp/ssbu_wedge_{int(time.time())}_{tag}.txt"
    try:
        parts = []
        try:
            from rl.input.control_block import ControlBlock
            with ControlBlock(create=False) as cb:
                parts.append(f"control: {cb.read()}")
        except Exception as exc:                  # noqa: BLE001
            parts.append(f"control: unreadable ({exc})")
        try:
            from rl.input.gym_reset import read_state
            parts.append(f"state: {read_state()}")
        except Exception as exc:                  # noqa: BLE001
            parts.append(f"state: unreadable ({exc})")
        try:
            out = subprocess.run(
                ["ps", "-eo", "pid,etime,pcpu,rss,stat,wchan:20,comm"],
                capture_output=True, text=True, timeout=5).stdout
            parts.append("\n".join(l for l in out.splitlines()
                                   if "ssbu" in l or "PID" in l))
        except Exception as exc:                  # noqa: BLE001
            parts.append(f"ps failed ({exc})")
        try:
            with open(os.path.expanduser("~/smash.log"), "r",
                      errors="ignore") as fh:
                parts.append("--- smash.log tail\n" + "".join(fh.readlines()[-40:]))
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(x) for x in parts))
        log(f"[ppo] wedge diagnostics -> {path}")
        return path
    except Exception:                             # noqa: BLE001
        return None


def restart_game(env, pad, args, log):
    """Relaunch the game, the reader and the pad inside this process.

    The trainer keeps its model, critic and optimizer; only the live game is
    replaced.  Returns the new pad.  Raises if the fight does not come up.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    live_sh = os.path.abspath(os.path.join(here, "..", "scripts", "live.sh"))
    match = os.environ.get("SSBU_LIVE_MATCH")
    if not match:
        raise RuntimeError("restart_game needs SSBU_LIVE_MATCH (use live_rl.sh)")
    try:
        if pad is not None:
            pad.destroy()
    except Exception:                             # noqa: BLE001
        pass
    log("[ppo] restarting the game (keeping the model and optimizer)")
    subprocess.run(["pkill", "-x", "ssbu_native"], check=False)
    time.sleep(2.0)
    rc = subprocess.run([live_sh, "--no-focus", "--keyboard", "0",
                         "--match", match], capture_output=True, text=True,
                        timeout=240)
    if rc.returncode != 0:
        raise RuntimeError(f"live.sh failed: {rc.stderr[-300:]}")
    from rl.input.gym_input import VirtualPad
    from rl.input.gym_reset import ResetClient
    new_pad = VirtualPad().create()
    base = getattr(env, "base_env", env)
    base._input_adapter = new_pad
    base._input_resolved = True
    # a fresh control block / state file: re-resolve both
    try:
        base.reader.close()
    except Exception:                             # noqa: BLE001
        pass
    base._reset_fn = ResetClient(timeout=float(args.reset_timeout)).reset
    base._reset_resolved = True
    ge.wait_for_pad(args.game_log, args.pad_timeout, log=log)
    game_speed = float(os.environ.get("SSBU_SPEED", "1") or 1.0)
    if not ge.drive_menus(base, new_pad, args.menu_timeout, log=log,
                          speed=game_speed):
        raise RuntimeError("game restarted but the fight did not start")
    return new_pad


def recover_pad(env, pad, args, log) -> Any:
    """Recreate the virtual pad after P1 input stalls, without rebooting the game."""
    from rl.input.gym_input import VirtualPad
    try:
        pad.destroy()
    except Exception:                             # noqa: BLE001
        pass
    new_pad = VirtualPad().create()
    # Training wrappers (e.g. the recovery drill) delegate to a base env.
    base = getattr(env, "base_env", env)
    # Swap the adapter object the env already resolved.
    base._input_adapter = new_pad
    base._input_resolved = True
    ge.wait_for_pad(args.game_log, args.pad_timeout, log=log)
    # A controller disconnect can leave the game on a pause overlay even
    # though fight_on stays 1; a few A presses dismiss it.
    for _ in range(3):
        try:
            new_pad.press("A")
            time.sleep(0.10)
            new_pad.release_all()
        except Exception:                         # noqa: BLE001
            break
        time.sleep(0.15)
    if not ge._fight_on(base):
        game_speed = float(os.environ.get("SSBU_SPEED", "1") or 1.0)
        ge.drive_menus(base, new_pad, args.menu_timeout, log=log,
                       speed=game_speed)
    return new_pad


def iteration_checkpoint_path(path: str, iteration: int) -> str:
    """``rl_dk9_ppo.pt`` + iter 7 -> ``rl_dk9_ppo_iter007.pt``."""
    root, ext = os.path.splitext(path)
    return f"{root}_iter{int(iteration):03d}{ext or '.pt'}"


def save_checkpoint(path: str, model, cfg: bc.BCConfig,
                    vocab: bc.ActionVocab, mean, std, bc_ckpt: dict,
                    iteration: int, stats: dict, args,
                    latest_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    checkpoint = {
        "format": "ssbu_bc_v1",
        "config": cfg.to_dict(),
        "state_dict": {k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
        "norm": {"mean": mean.detach().cpu().tolist(),
                 "std": std.detach().cpu().tolist()},
        "thresholds": bc_ckpt.get("thresholds",
                                  [0.5] * bc.N_BUTTONS),
        "vocab": vocab.to_dict(),
        "expert_stats": bc_ckpt.get("expert_stats"),
        "meta": {
            "rl": True,
            "bc_model": os.path.abspath(args.model),
            "iteration": int(iteration),
            "reward": "0.01 * (damage dealt - damage taken)",
            "gamma": float(args.gamma),
            "lam": float(args.lam),
            "clip": float(args.clip),
            "entropy": float(args.entropy),
            "value_coef": float(args.value_coef),
            "kl_coef": float(getattr(args, "kl_coef", 0.0)),
            "kl_model": getattr(args, "kl_model", None),
            "stock_value": float(getattr(args, "stock_value", 0.0)),
            "offstage_penalty": float(getattr(args, "offstage_penalty", 0.0)),
            "opponent_offstage_reward": float(
                getattr(args, "opponent_offstage_reward", 0.0)),
            "opponent_offstage_potential": float(
                getattr(args, "opponent_offstage_potential", 0.0)),
            "recovery_weight": float(getattr(args, "recovery_weight", 0.0)),
            "recovery_ledge": bool(getattr(args, "recovery_ledge", False)),
            "doomed_penalty": float(getattr(args, "doomed_penalty", 0.0)),
            "offstage_loss_weight": float(
                getattr(args, "offstage_loss_weight", 1.0)),
            "mirror": bool(getattr(args, "mirror", False)),
            "mirror_prob": float(getattr(args, "mirror_prob", 0.0)),
            "recovery_drill": bool(getattr(args, "recovery_drill", False)),
            "drill_setup_steps": int(
                getattr(args, "drill_setup_steps", 0)),
            "drill_prob": float(getattr(args, "drill_prob", 0.0)),
            "recovery_bonus": float(getattr(args, "recovery_bonus", 0.0)),
            "damage_dealt_mult": float(getattr(args, "damage_dealt", 1.0)),
            "damage_taken_mult": float(getattr(args, "damage_taken", 1.0)),
            "last_stats": {k: float(v) for k, v in stats.items()},
            "saved_ns": time.time_ns(),
        },
    }
    targets = [path]
    if latest_path and os.path.abspath(latest_path) != os.path.abspath(path):
        targets.append(latest_path)
    for target in targets:
        os.makedirs(os.path.dirname(os.path.abspath(target)) or ".",
                    exist_ok=True)
        tmp = target + ".tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, target)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PPO fine-tuning for the SSBU BC token policy")
    ap.add_argument("--model", required=True, help="BC checkpoint to warm start")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "models",
                                                  "rl_dk9_ppo.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--rollout-steps", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=900,
                    help="decisions per live episode (30 s at frame_skip 2)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="BC-anchored KL penalty coefficient (0 = off)")
    ap.add_argument("--kl-model", default=None,
                    help="frozen reference checkpoint for KL (default: the "
                         "warm-start model)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--stock-value", type=float, default=1.0,
                    help="reward units for each lost/earned stock "
                         "(1.0 = 100%% damage; 0 disables stock terms)")
    ap.add_argument("--offstage-penalty", type=float, default=0.0,
                    help="small per-decision reward penalty while the ego is "
                         "offstage (e.g. 0.01)")
    ap.add_argument("--opponent-offstage-reward", type=float, default=0.0,
                    help="positive per-decision reward while the opponent is "
                         "below the stage (mirror of --offstage-penalty)")
    ap.add_argument("--opponent-offstage-potential", type=float, default=0.0,
                    help="potential-based edgeguard reward per unit of "
                         "increase in the opponent's below-stage danger")
    ap.add_argument("--recovery-weight", type=float, default=0.0,
                    help="potential-based reward per world-unit of recovery "
                         "toward the stage (e.g. 0.02)")
    ap.add_argument("--recovery-ledge", action="store_true",
                    help="use 2-D distance to the nearest ledge for the "
                         "recovery potential instead of vertical depth")
    ap.add_argument("--doomed-penalty", type=float, default=0.0,
                    help="dense per-decision penalty while below the stage, "
                         "falling, and out of jumps (e.g. 0.05)")
    ap.add_argument("--offstage-loss-weight", type=float, default=1.0,
                    help="per-sample PPO loss multiplier on offstage states "
                         "(1 = uniform; e.g. 3)")
    ap.add_argument("--restart-every", type=int, default=0,
                    help="restart the game every N iterations, keeping the "
                         "optimizer (prevention for the runtime wedge; 0=off)")
    ap.add_argument("--ledge-timeout", type=int, default=45,
                    help="force a getup after this many decisions hanging on "
                         "the ledge (0 = off)")
    ap.add_argument("--drill-fall-delay", type=int, default=4,
                    help="max extra decisions of free fall before the policy "
                         "gets control in the drill (small = earlier handoff)")
    ap.add_argument("--recovery-drill", action="store_true",
                    help="run P1 off the ledge, then hand control to the "
                         "policy mid-fall (training only)")
    ap.add_argument("--drill-setup-steps", type=int, default=180,
                    help="max decisions spent running to the ledge per drill")
    ap.add_argument("--drill-settle-steps", type=int, default=90,
                    help="max decisions to wait for a normal spawn before "
                         "the drill setup")
    ap.add_argument("--recovery-bonus", type=float, default=0.5,
                    help="reward added when the drill returns to the ground")
    ap.add_argument("--drill-prob", type=float, default=1.0,
                    help="fraction of episodes that use the recovery drill "
                         "(1 = drill only, 0 = normal episodes)")
    ap.add_argument("--mirror", action="store_true",
                    help="mirror whole episodes left/right (training only)")
    ap.add_argument("--mirror-prob", type=float, default=0.5,
                    help="probability that an episode is mirrored")
    ap.add_argument("--damage-dealt", type=float, default=1.0,
                    help="multiplier on damage dealt in the reward")
    ap.add_argument("--damage-taken", type=float, default=1.0,
                    help="multiplier on damage taken in the reward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--stats-file", default="/tmp/ssbu_training_stats.json",
                    help="JSON status file for the live viewer overlay")
    ap.add_argument("--cpu-level", type=int, default=0,
                    help="opponent CPU level shown in the overlay")
    ap.add_argument("--max-dead-episodes", type=int, default=0,
                    help="stop when a rollout has more than this many episodes "
                         "with zero damage dealt (default 0: stop immediately)")
    ap.add_argument("--start-menus", action="store_true",
                    help="create the pad and walk boot -> match first")
    ap.add_argument("--game-log", default="~/smash.log")
    ap.add_argument("--pad-timeout", type=float, default=15.0)
    ap.add_argument("--menu-timeout", type=float, default=120.0)
    ap.add_argument("--reset-timeout", type=float, default=20.0,
                    help="seconds to wait for the in-fight rematch ack")
    ap.add_argument("--mock", action="store_true",
                    help="no game; tiny mock env for a plumbing run")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="policy dropout during fine-tuning (default 0)")
    args = ap.parse_args(argv)

    ppo = PPOConfig(
        iterations=args.iterations, rollout_steps=args.rollout_steps,
        max_steps=args.max_steps, epochs=args.epochs,
        minibatch=args.minibatch, lr=args.lr, gamma=args.gamma,
        lam=args.lam, clip=args.clip, entropy=args.entropy,
        value_coef=args.value_coef, kl_coef=args.kl_coef,
        offstage_loss_weight=args.offstage_loss_weight,
        weight_decay=args.weight_decay, seed=args.seed,
        device=args.device)
    ppo.check()
    if float(args.stock_value) < 0.0:
        ap.error("--stock-value must be >= 0")
    if int(args.max_dead_episodes) < 0:
        ap.error("--max-dead-episodes must be >= 0")
    if float(args.offstage_penalty) < 0.0:
        ap.error("--offstage-penalty must be >= 0")
    if float(args.opponent_offstage_reward) < 0.0:
        ap.error("--opponent-offstage-reward must be >= 0")
    if float(args.opponent_offstage_potential) < 0.0:
        ap.error("--opponent-offstage-potential must be >= 0")
    if float(args.recovery_weight) < 0.0:
        ap.error("--recovery-weight must be >= 0")
    if float(args.doomed_penalty) < 0.0:
        ap.error("--doomed-penalty must be >= 0")
    if float(args.offstage_loss_weight) < 1.0:
        ap.error("--offstage-loss-weight must be >= 1")
    if int(args.drill_setup_steps) < 1:
        ap.error("--drill-setup-steps must be >= 1")
    if int(args.drill_settle_steps) < 0:
        ap.error("--drill-settle-steps must be >= 0")
    if float(args.recovery_bonus) < 0.0:
        ap.error("--recovery-bonus must be >= 0")
    if not 0.0 <= float(args.drill_prob) <= 1.0:
        ap.error("--drill-prob must be in [0, 1]")
    if not 0.0 <= float(args.mirror_prob) <= 1.0:
        ap.error("--mirror-prob must be in [0, 1]")
    if float(args.damage_dealt) < 0.0 or float(args.damage_taken) < 0.0:
        ap.error("--damage-dealt/--damage-taken must be >= 0")

    log = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731
    torch.manual_seed(ppo.seed)
    np.random.seed(ppo.seed)
    device = str(ppo.device)
    if device.startswith("cpu"):
        # Leave CPU for the game/reader; four threads is enough for this GRU.
        torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    model, cfg, vocab, mean, std, bc_ckpt = load_bc_model(
        args.model, device, dropout=args.dropout)
    model.to(device)
    ref_model = None
    if ppo.kl_coef > 0.0:
        if args.kl_model:
            ref_model, _ref_cfg, ref_vocab, _rmean, _rstd, _ = \
                load_bc_model(args.kl_model, device, dropout=0.0)
            if (ref_vocab.size != vocab.size or
                    not np.array_equal(ref_vocab.keys, vocab.keys)):
                raise ValueError("--kl-model has a different action vocabulary")
        else:
            ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
    ref_label = args.kl_model or args.model
    log(f"[ppo] warm start {args.model}: vocab {vocab.size} classes, "
        f"window {cfg.window}, horizon {cfg.horizon}, frame_skip "
        f"{cfg.frame_skip}, damage {cfg.damage_mode}, "
        f"kl_bc={ppo.kl_coef:g} ({ref_label}), "
        f"stock_value={args.stock_value:g}")
    stats_writer = TrainingStats(args.stats_file, args.cpu_level,
                                 ppo.iterations)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ppo.lr),
                                  weight_decay=float(ppo.weight_decay))
    pad = None
    env = None
    reset_client = None
    try:
        if args.mock:
            env = MockCombatEnv(max_steps=ppo.max_steps, seed=ppo.seed)
            log("[ppo] mock environment (plumbing only)")
        else:
            if args.start_menus:
                from rl.input.gym_input import VirtualPad
                pad = VirtualPad().create()
                log(f"[ppo] pad up ({pad.name})")
            from rl.env.gym_env import SSBUGymEnv
            from rl.env.gym_reward import RewardConfig
            # Damage differential plus a stock/KO term.  `compute_reward`
            # detects a death from the is_on_rebirth transition when the
            # live stock field is unreadable.
            stock_value = float(args.stock_value)
            reward_cfg = RewardConfig(
                damage_scale=0.01,
                damage_dealt=float(args.damage_dealt),
                damage_taken=float(args.damage_taken),
                ko_bonus=stock_value, stock_loss=-stock_value,
                win_bonus=0.0, loss_penalty=0.0, draw_reward=0.0,
                ego_slot=0, offstage_penalty=float(args.offstage_penalty),
                opponent_offstage_reward=float(
                    args.opponent_offstage_reward),
                opponent_offstage_potential=float(
                    args.opponent_offstage_potential),
                recovery_weight=float(args.recovery_weight),
                recovery_ledge=bool(args.recovery_ledge),
                doomed_penalty=float(args.doomed_penalty))
            reset_fn = None
            try:
                from rl.input.gym_reset import ResetClient
                reset_client = ResetClient(timeout=float(args.reset_timeout))
                reset_fn = reset_client.reset
            except Exception as exc:    # noqa: BLE001
                log(f"[ppo] no long-timeout reset client ({exc}); "
                    "using the environment default")
            env = SSBUGymEnv(
                normalize_obs=cfg.obs_normalize, damage_mode=cfg.damage_mode,
                frame_skip=cfg.frame_skip, max_steps=ppo.max_steps,
                reward_mode="combat", reward_config=reward_cfg,
                stock_termination=False, require_positive_stock=False,
                input_adapter=pad, reset_fn=reset_fn)
            if pad is not None:
                ge.wait_for_pad(args.game_log, args.pad_timeout, log=log)
                game_speed = float(os.environ.get("SSBU_SPEED", "1") or 1.0)
                if not ge.drive_menus(env, pad, args.menu_timeout, log=log,
                                      speed=game_speed):
                    log("[ppo] giving up: no fight")
                    return 2
        if int(args.ledge_timeout) > 0 and not args.mock:
            from rl.env.gym_ledge import LedgeTimeoutEnv
            env = LedgeTimeoutEnv(env, max_hang=int(args.ledge_timeout),
                                  log=log)
            log(f"[ppo] ledge timeout: forced getup after "
                f"{args.ledge_timeout} decisions on the ledge")

        if args.recovery_drill:
            if args.mock:
                log("[ppo] --recovery-drill needs the live game; ignoring "
                    "it for the mock env")
            else:
                from rl.env.gym_drill import RecoveryDrillEnv
                env = RecoveryDrillEnv(
                    env, setup_steps=int(args.drill_setup_steps),
                    settle_reset_steps=int(args.drill_settle_steps),
                    recovery_bonus=float(args.recovery_bonus),
                    drill_prob=float(args.drill_prob),
                    fall_delay_max=int(args.drill_fall_delay), log=log)
                log("[ppo] recovery drill: run P1 off the ledge, then hand "
                    "control to the policy mid-fall")

        if args.mirror:
            from rl.rl.mirror import MirrorWrapper
            env = MirrorWrapper(env, mirror_prob=float(args.mirror_prob),
                                log=log)
            log(f"[ppo] mirror wrapper: {args.mirror_prob:.0%} of episodes "
                "are presented left/right flipped")

        state = PolicyState(cfg)

        def reset_with_recovery(reset_seed: int, exc: Exception):
            """Escalate: retry, recreate the pad, then restart the game."""
            nonlocal pad
            if pad is None:
                raise exc
            dump_wedge_diagnostics("reset", log)
            for attempt in range(1, 4):
                log(f"[ppo] reset failed ({exc}); recovery {attempt}/3 "
                    f"({'pad' if attempt == 1 else 'game restart'})")
                try:
                    if attempt == 1:
                        pad = recover_pad(env, pad, args, log)
                    else:
                        pad = restart_game(env, pad, args, log)
                    return ge.reset_env(env, reset_seed)
                except Exception as retry_exc:    # noqa: BLE001
                    exc = retry_exc
            raise exc

        def episode_log(row: dict) -> None:
            stats_writer.record_episode(row)
            p1 = row.get("p1_damage")
            p2 = row.get("p2_damage")
            diff = ((float(p2) - float(p1))
                    if p1 is not None and p2 is not None else float("nan"))
            log(f"[ppo] episode: dmg_diff {diff:+.1f} "
                f"rew {float(row.get('return', 0.0)):+.3f} "
                f"deaths P1 {row.get('p1_deaths', 0)}/"
                f"P2 {row.get('p2_deaths', 0)} "
                f"death_rew {row.get('death_rewards', [])}")
            for d in row.get("death_infos", []):
                log("[ppo] death: " + " ".join(f"{k}={v}" for k, v in d.items()))

        for iteration in range(1, int(ppo.iterations) + 1):
            if (int(args.restart_every) > 0 and iteration > 1
                    and (iteration - 1) % int(args.restart_every) == 0
                    and pad is not None):
                pad = restart_game(env, pad, args, log)
            t0 = time.monotonic()
            batch, episodes = collect_rollout(
                env, model, state, cfg, vocab, mean, std, device,
                int(ppo.rollout_steps), int(ppo.seed) + 1000 * iteration,
                on_episode=episode_log,
                on_step=stats_writer.record_step,
                on_reset_error=reset_with_recovery)
            # In a recovery drill P1 may never touch P2, so zero damage is
            # expected rather than evidence of a stalled pad.
            dead = [] if args.recovery_drill else [
                e for e in episodes
                if float(e.get("p2_damage", 0.0) or 0.0) <= 0.0]
            if len(dead) > int(args.max_dead_episodes):
                detail = ", ".join(
                    f"ep{e.get('length')} taken={e.get('p1_damage')} "
                    f"dealt={e.get('p2_damage')} deaths P1={e.get('p1_deaths')}"
                    for e in dead[:4])
                log(f"[ppo] dead episode(s): {detail}; recovering the pad "
                    "instead of restarting the game")
                if pad is None:
                    raise RuntimeError(
                        "dead episode with no pad to recover; rerun with "
                        "--start-menus")
                try:
                    pad = recover_pad(env, pad, args, log)
                except Exception as exc:          # noqa: BLE001
                    raise RuntimeError(
                        f"pad recovery failed after dead episodes ({exc}); "
                        "stopping with the last checkpoint") from exc
                log("[ppo] pad recovered; discarding the dead rollout")
                continue
            next_values = batch_values(
                model, batch["next_windows"], batch["next_prevs"],
                mean, std, device)
            adv, returns = compute_gae(
                batch["rewards"], batch["values"], next_values,
                batch["terminals"], batch["truncateds"],
                float(ppo.gamma), float(ppo.lam))
            adv = (adv - float(adv.mean())) / (float(adv.std()) + 1e-8)
            batch["advantages"] = adv.astype(np.float32)
            batch["returns"] = returns.astype(np.float32)
            if ref_model is not None:
                batch["ref_logits"] = batch_logits(
                    ref_model, batch["windows"], batch["prevs"],
                    mean, std, device)
            stats = ppo_update(model, optimizer, batch, ppo, mean, std, device)
            stats_writer.record_iteration(iteration, stats)
            if episodes:
                dealt = [e["p2_damage"] for e in episodes
                         if e["p2_damage"] is not None]
                taken = [e["p1_damage"] for e in episodes
                         if e["p1_damage"] is not None]
                rews = [e.get("return") for e in episodes
                        if e.get("return") is not None]
                p1d = sum(int(e.get("p1_deaths", 0) or 0) for e in episodes)
                p2d = sum(int(e.get("p2_deaths", 0) or 0) for e in episodes)
                log(f"[ppo] iter {iteration}/{ppo.iterations}: "
                    f"steps {len(batch['actions'])} eps {len(episodes)} "
                    f"dealt {np.mean(dealt) if dealt else float('nan'):.1f} "
                    f"taken {np.mean(taken) if taken else float('nan'):.1f} "
                    f"rew {np.mean(rews) if rews else float('nan'):+.2f} "
                    f"deaths P1 {p1d}/P2 {p2d} "
                    f"policy {stats['policy_loss']:.4f} "
                    f"value {stats['value_loss']:.4f} "
                    f"entropy {stats['entropy']:.3f} "
                    f"kl_bc {stats['kl_bc']:.4f} "
                    f"({time.monotonic() - t0:.0f}s)")
            else:
                log(f"[ppo] iter {iteration}/{ppo.iterations}: "
                    f"steps {len(batch['actions'])} partial episode "
                    f"policy {stats['policy_loss']:.4f} "
                    f"value {stats['value_loss']:.4f} "
                    f"kl_bc {stats['kl_bc']:.4f} "
                    f"({time.monotonic() - t0:.0f}s)")
            if args.save_every > 0 and iteration % args.save_every == 0:
                path = iteration_checkpoint_path(args.out, iteration)
                save_checkpoint(path, model, cfg, vocab, mean, std,
                                bc_ckpt, iteration, stats, args,
                                latest_path=args.out)
                log(f"[ppo] saved {path} (latest {args.out})")
        final_stats = stats if "stats" in locals() else {}
        final_path = iteration_checkpoint_path(args.out, int(ppo.iterations))
        save_checkpoint(final_path, model, cfg, vocab, mean, std, bc_ckpt,
                        int(ppo.iterations), final_stats, args,
                        latest_path=args.out)
        log(f"[ppo] final checkpoint {final_path} (latest {args.out})")
        return 0
    finally:
        stats_writer.finish()
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if reset_client is not None:
            try:
                reset_client.close()
            except Exception:                     # noqa: BLE001
                pass
        if pad is not None:
            pad.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
