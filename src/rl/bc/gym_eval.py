#!/usr/bin/env python3
"""Run a behaviour-cloning checkpoint in the SSBU gym (or a mock env).

    .venv/bin/python tools/mem/gym_eval.py --model data/E/bc.pt --episodes 5
    .venv/bin/python tools/mem/gym_eval.py --model data/E/bc.pt --mock --episodes 3

The real environment is imported lazily (``from gym_env import SSBUGymEnv``),
so this script and its unit tests work without the game or Agent C's files.
If the import fails and ``--mock`` was not given, a clear message is printed
and the mock is used only when ``--mock`` is explicit; otherwise the script
exits with code 2 (a closed-loop acceptance run must not silently evaluate a
mock).

Reports wins/losses, damage, episode lengths and the policy action distribution
compared with the expert distribution (from ``--data`` or the stats stored in
the checkpoint).

Environment interface assumed (documented in FINDINGS, Agent E):
* ``SSBUGymEnv()`` constructible with no required arguments;
* Gymnasium reset/step: ``reset() -> (obs, info)`` (a bare ``obs`` is accepted),
  ``step(action) -> (obs, reward, terminated, truncated, info)`` (4-tuples are
  accepted);
* ``obs`` is the 56-per-fighter CONTRACTS.md policy observation (or a dict with it
  under ``obs``/``observation``/``state``);
* actions are either the 16-float vector ``[buttons(12), stick(2), cstick(2)]``
  (what ``gym_action.decode_action`` accepts) or a dict with
  ``buttons``/``stick``/``cstick`` (and an ignored ``trigger``).
  ``--action-format auto`` picks by ``env.action_space``.
* the checkpoint config is restored (window, horizon, frame_skip,
  obs_normalize, trigger, cstick, dropout, shield_from_status); the env is
  built with the checkpoint's ``frame_skip``/``normalize_obs``.  ``--cstick``
  only checks that the C-stick head is present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from rl.bc import gym_bc as bc  # noqa: E402

RESULT_KEYS = ("winner", "p1_winner", "who_won", "result")
WIN_VALUES = {"p1", "player1", "player_1", "self", 0, "0"}
LOSS_VALUES = {"p2", "player2", "player_2", "opponent", 1, "1"}


# ---------------------------------------------------------------------------
# mock environment
# ---------------------------------------------------------------------------
class MockEnv:
    """A tiny deterministic-ish stand-in for closed-loop smoke tests.

    It has the Gymnasium reset/step shape, the CONTRACTS.md observation and a
    17-float action space, but no game: pressing A raises the opponent's
    damage, the opponent occasionally damages us, and the episode ends after
    ``max_steps`` or when one side reaches 140%.
    """

    def __init__(self, max_steps: int = 120, seed: int = 0):
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.damage = [0.0, 0.0]
        self.stocks = [1, 1]
        self.obs = np.zeros(bc.OBS_DIM, np.float32)
        self.action_space = _Box((17,), -1.0, 1.0)
        self.observation_space = _Box((bc.OBS_DIM,), -np.inf, np.inf)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.damage = [0.0, 0.0]
        self.stocks = [1, 1]
        self.obs = self._make_obs()
        return self.obs.copy(), {"mock": True}

    def _make_obs(self) -> np.ndarray:
        obs = self.obs.copy()
        # Both fighter blocks: position wanders, damage is visible.
        obs[0] = float(np.clip(obs[0] + self.rng.normal(0, 0.02), -1.5, 1.5))
        obs[1] = float(np.clip(obs[1] + self.rng.normal(0, 0.005), -1, 1))
        obs[2] = float(np.clip(obs[2] + self.rng.normal(0, 0.05), -1, 1))
        obs[4] = self.damage[0] / 100.0
        obs[5] = 1.0 if self.rng.random() > 0.5 else -1.0
        obs[6] = 1.0
        obs[7] = 1.0
        obs[8] = 1.0
        obs[46] = (self.rng.normal(0, 0.5)) / 100.0
        obs[49] = 1.0
        base = bc.OBS_PER_FIGHTER
        obs[base + 0] = float(np.clip(obs[base + 0] + self.rng.normal(0, 0.02),
                                      -1.5, 1.5))
        obs[base + 4] = self.damage[1] / 100.0
        obs[base + 5] = -obs[5]
        obs[base + 6] = 1.0
        obs[base + 7] = 1.0
        obs[base + 46] = -obs[46]
        obs[base + 49] = 1.0
        return obs

    def step(self, action):
        buttons = _action_buttons(action)
        a_pressed = bool(buttons[bc.BUTTON_INDEX["A"]] > 0.5)
        b_pressed = bool(buttons[bc.BUTTON_INDEX["B"]] > 0.5)
        if a_pressed:
            self.damage[1] += float(self.rng.uniform(3.0, 9.0))
        if b_pressed:
            self.damage[1] += float(self.rng.uniform(1.0, 4.0))
        if self.rng.random() < 0.25:
            self.damage[0] += float(self.rng.uniform(0.0, 5.0))
        self.t += 1
        self.obs = self._make_obs()
        terminated = False
        winner = None
        if self.damage[1] >= 140.0 and self.damage[0] < 140.0:
            self.stocks[1] = 0
            winner = 0
            terminated = True
        elif self.damage[0] >= 140.0:
            self.stocks[0] = 0
            winner = 1
            terminated = True
        truncated = self.t >= self.max_steps
        info = {
            "mock": True,
            "winner": winner,
            "p1_stock": self.stocks[0],
            "p2_stock": self.stocks[1],
            "p1_damage": self.damage[0],
            "p2_damage": self.damage[1],
            "p1_char": "mario",
            "p2_char": "donkey_kong",
            "stage": "final_destination",
            "action": action,
        }
        return self.obs.copy(), 0.0, terminated, truncated, info


class _Box:
    def __init__(self, shape, low, high):
        self.shape = tuple(shape)
        self.low = low
        self.high = high

    def contains(self, x):
        arr = np.asarray(x)
        return arr.shape == self.shape


def _action_buttons(action) -> np.ndarray:
    if isinstance(action, dict):
        return np.asarray(action.get("buttons", np.zeros(bc.N_BUTTONS)),
                          np.float32).reshape(-1)
    arr = np.asarray(action, np.float32).reshape(-1)
    return arr[:bc.N_BUTTONS]


def _action_dict(action) -> dict:
    if isinstance(action, dict):
        return action
    arr = np.asarray(action, np.float32).reshape(-1)
    trigger = float(arr[16]) if arr.size >= 17 else 0.0
    return {"buttons": arr[0:12], "stick": arr[12:14],
            "cstick": arr[14:16], "trigger": trigger}


# ---------------------------------------------------------------------------
# real environment adapter
# ---------------------------------------------------------------------------
def make_env(cfg: bc.BCConfig, episodes: int, max_steps: int, seed: int,
             mock: bool, log=print, input_adapter=None):
    if not mock:
        try:
            from rl.env.gym_env import SSBUGymEnv  # type: ignore
        except Exception as exc:
            log(f"[eval] cannot import gym_env.SSBUGymEnv ({exc}); "
                "pass --mock for a smoke test")
            raise SystemExit(2)
        # Match the checkpoint's config exactly: observation scaling and the
        # decision interval (reader frames per step).  Fall back for older env
        # constructors that lack some of these keywords.
        kwargs = {"normalize_obs": cfg.obs_normalize,
                  "frame_skip": cfg.frame_skip, "max_steps": max_steps}
        if input_adapter is not None:
            kwargs["input_adapter"] = input_adapter
        try:
            env = SSBUGymEnv(**kwargs)
        except TypeError:
            try:
                env = SSBUGymEnv(normalize_obs=cfg.obs_normalize,
                                 frame_skip=cfg.frame_skip)
            except TypeError:
                try:
                    env = SSBUGymEnv(normalize_obs=cfg.obs_normalize)
                except TypeError as exc:
                    log(f"[eval] SSBUGymEnv() needs arguments ({exc}); "
                        "the adapter assumes a no-argument constructor")
                    raise SystemExit(2)
        log(f"[eval] env config: frame_skip={cfg.frame_skip} "
            f"normalize_obs={cfg.obs_normalize} max_steps={max_steps}")
        return env, False
    return MockEnv(max_steps=max_steps, seed=seed), True


#: Boot -> fight with a pad, one step per screen.  Every screen of the Smash
#: menu advances on A; the only exception is the character select, where the
#: hand has to travel from the player panel up into the roster (a stick flick)
#: before A picks; A on the "READY TO FIGHT" banner starts the match.  This is
#: what a person does with a controller, only from code.
MENU_MACRO: tuple[tuple[str, float, float], ...] = (
    ("A", 0.10, 0.9),          # "PRESS ANY BUTTON" -> main menu
    ("A", 0.10, 0.9),          # Smash tile -> Smash submenu
    ("A", 0.10, 0.9),          # regular Smash -> rules
    ("A", 0.10, 0.9),          # rules -> stage select
    ("A", 0.10, 1.6),          # stage -> loading, then character select
    ("A", 0.10, 1.6),          # character select: open the roster
    ("up", 0.50, 0.4),         # hand up into the roster
    ("right", 0.30, 0.4),      # onto Donkey Kong
    ("A", 0.10, 0.9),          # pick -> "READY TO FIGHT"
    ("A", 0.10, 2.0),          # start the match
)


def _state_ready(env) -> bool:
    """True when the reader has published at least one state frame."""
    reader = getattr(env, "reader", None)
    if reader is None:
        return True
    try:
        reader.read_latest(timeout=1.0)
        return True
    except Exception:
        return False


def _fight_on(env) -> bool:
    reader = getattr(env, "reader", None)
    if reader is None:
        return False
    try:
        return bool(reader.read_latest(timeout=1.0).fight_on)
    except Exception:
        return False


def drive_menus(env, pad, timeout: float = 120.0, log=print) -> bool:
    """Walk boot -> match with ``pad``; True once the fight is on.

    The pad must be the one the policy will use: it joins player 1, and a human
    slot left without its controller is dropped back to the title by the game.
    """
    deadline = time.monotonic() + max(60.0, timeout)
    while not _state_ready(env):                 # the game may still be loading
        if time.monotonic() > deadline:
            log("[eval] menu walk: the state block never appeared")
            return False
        time.sleep(0.25)
    log("[eval] menu walk: state block is live, pressing A through the menus")
    t0 = time.monotonic()
    cycle = 0
    while time.monotonic() - t0 < timeout:
        cycle += 1
        # The macro is repeated, not just its A presses: boot time varies, so a
        # first pass can finish before the character select is up, and the
        # roster needs the stick flick before A does anything useful.
        for keys, hold, wait in MENU_MACRO:
            if _fight_on(env):
                log(f"[eval] menu walk: the fight is on after {time.monotonic() - t0:.0f}s")
                return True
            _pad_step(pad, keys, hold)
            time.sleep(wait)
        log(f"[eval] menu walk: pass {cycle} did not reach the fight, retrying")
    log(f"[eval] menu walk: no fight after {timeout:.0f}s")
    return _fight_on(env)


def _pad_step(pad, keys: str, hold: float) -> None:
    stick = {"up": (0.0, 1.0), "down": (0.0, -1.0),
             "left": (-1.0, 0.0), "right": (1.0, 0.0)}.get(keys)
    if stick is not None:
        pad.set_stick(*stick)
        time.sleep(hold)
        pad.release_all()
        return
    for key in keys.split("+"):
        pad.press(key)
    time.sleep(hold)
    pad.release_all()


def reset_env(env, seed: int):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
    return _split_reset(result)


def _split_reset(result):
    if isinstance(result, tuple):
        obs = result[0]
        info = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
        return obs, info
    return result, {}


def _extract_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        for key in ("obs", "observation", "state", "policy"):
            if key in obs:
                obs = obs[key]
                break
        else:
            raise KeyError(f"cannot find observations in dict keys {list(obs)}")
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 2:
        arr = arr.reshape(-1)
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1])[-1]
    return arr.reshape(-1)


def step_env(env, action):
    result = env.step(action)
    if not isinstance(result, tuple):
        raise TypeError("env.step() must return a tuple")
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        terminated, truncated = bool(done), False
    else:
        raise TypeError(f"env.step() returned {len(result)} values")
    return obs, float(reward), bool(terminated), bool(truncated), info or {}


def format_action(action: dict, fmt: str, env) -> object:
    if fmt == "dict":
        return bc.action_to_env(action, "dict")
    if fmt == "array":
        return bc.action_to_env(action, "array")
    space = getattr(env, "action_space", None)
    shape = getattr(space, "shape", None)
    if space is not None and not isinstance(space, _Box) and shape is None:
        # Gymnasium spaces.Dict has no plain ``shape`` but has ``spaces``.
        if hasattr(space, "spaces"):
            return bc.action_to_env(action, "dict")
    if isinstance(space, _Box) or (shape is not None and
                                   tuple(shape) in ((16,), (17,))):
        return bc.action_to_env(action, "array")
    if hasattr(space, "spaces"):
        return bc.action_to_env(action, "dict")
    return bc.action_to_env(action, "dict")


def _pair(info: dict, *keys):
    """First list/tuple value among ``keys`` with at least two numbers."""
    for key in keys:
        value = info.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                continue
    return None, None


def _result_from_info(info: dict) -> dict:
    """Extract winner/stocks/damage from a real or mock env info dict."""
    winner = None
    for key in RESULT_KEYS:
        if key in info:
            value = info[key]
            if value in WIN_VALUES:
                winner = 0
            elif value in LOSS_VALUES:
                winner = 1
            break
    p1s, p2s = _pair(info, "stocks")
    if p1s is None:
        p1s = _num(info, "p1_stock", "stock_p1")
        p2s = _num(info, "p2_stock", "stock_p2")
    p1d, p2d = _pair(info, "damages", "damage")
    if p1d is None:
        p1d = _num(info, "p1_damage", "damage_p1", "self_damage")
        p2d = _num(info, "p2_damage", "damage_p2", "opp_damage")
    if winner is None and p1s is not None and p2s is not None:
        if p1s > p2s:
            winner = 0
        elif p2s > p1s:
            winner = 1
    if winner is None and p1d is not None and p2d is not None:
        if p2d >= 140.0 > p1d:
            winner = 0
        elif p1d >= 140.0 > p2d:
            winner = 1
    return {"winner": winner, "p1_stock": p1s, "p2_stock": p2s,
            "p1_damage": p1d, "p2_damage": p2d}


def _num(info: dict, *keys, default=None):
    for key in keys:
        if key in info and isinstance(info[key], (int, float)):
            return float(info[key])
    return default


def run_episodes(env, ensemble: bc.TemporalEnsemble, episodes: int,
                 max_steps: int, seed: int, action_format: str = "auto",
                 mock: bool = False, log=print) -> dict:
    results = []
    policy_buttons = []
    policy_stick = []
    policy_cstick = []
    policy_trigger = []
    for ep in range(episodes):
        obs, info = reset_env(env, seed + ep)
        ensemble.reset()
        done = False
        length = 0
        last_info = dict(info)
        peak_damage = {"p1_damage": 0.0, "p2_damage": 0.0}
        while not done and length < max_steps:
            action = ensemble.act(_extract_obs(obs))
            env_action = format_action(action, action_format, env)
            obs, _reward, terminated, truncated, info = step_env(env, env_action)
            last_info = dict(info)
            step_result = _result_from_info(info)
            for key in peak_damage:
                value = step_result.get(key)
                if value is not None:
                    peak_damage[key] = max(peak_damage[key], float(value))
            policy_buttons.append(np.asarray(action["buttons"], np.float32))
            policy_stick.append(np.asarray(action["stick"], np.float32))
            policy_cstick.append(np.asarray(action.get("cstick", np.zeros(2)),
                                            np.float32))
            policy_trigger.append(float(action.get("trigger", 0.0)))
            length += 1
            done = terminated or truncated
        result = _result_from_info(last_info)
        for key, peak in peak_damage.items():
            if result.get(key) in (None, 0.0) and peak > 0.0:
                result[key] = peak
        result.update({
            "episode": ep,
            "length": length,
            "mock": bool(last_info.get("mock", mock)),
        })
        results.append(result)
        log(f"[eval] episode {ep + 1}/{episodes}: winner={result['winner']} "
            f"len={length} p1_dmg={results[-1]['p1_damage']} "
            f"p2_dmg={results[-1]['p2_damage']}")

    buttons = np.stack(policy_buttons) if policy_buttons else \
        np.zeros((0, bc.N_BUTTONS), np.float32)
    stick = np.stack(policy_stick) if policy_stick else \
        np.zeros((0, 2), np.float32)
    cstick = np.stack(policy_cstick) if policy_cstick else \
        np.zeros((0, 2), np.float32)
    trigger = np.asarray(policy_trigger, np.float32)
    wins = sum(1 for r in results if r["winner"] == 0)
    losses = sum(1 for r in results if r["winner"] == 1)
    unknown = len(results) - wins - losses
    def _mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    policy_stats = {
        "button_rate": {name: float(buttons[:, i].mean())
                        for i, name in enumerate(bc.BUTTONS)}
        if len(buttons) else {name: 0.0 for name in bc.BUTTONS},
        "stick_mean": [float(stick[:, 0].mean()), float(stick[:, 1].mean())]
                      if len(stick) else [0.0, 0.0],
        "stick_abs_mean": [float(np.abs(stick[:, 0]).mean()),
                           float(np.abs(stick[:, 1]).mean())]
                          if len(stick) else [0.0, 0.0],
        # how often the stick is left neutral: a policy that stands still
        # shows up here long before the damage numbers say anything
        "stick_neutral_rate": float(np.mean(
            np.hypot(stick[:, 0], stick[:, 1]) < 0.05)) if len(stick) else 0.0,
        "cstick_mean": [float(cstick[:, 0].mean()), float(cstick[:, 1].mean())]
                       if len(cstick) else [0.0, 0.0],
        "trigger_rate": float(trigger.mean()) if len(trigger) else 0.0,
    }
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "unknown": unknown,
        "win_rate": wins / episodes if episodes else 0.0,
        "mean_episode_length": _mean("length"),
        "mean_p1_damage": _mean("p1_damage"),
        "mean_p2_damage": _mean("p2_damage"),
        "results": results,
        "policy_stats": policy_stats,
        "mock": bool(mock),
    }


def compare_stats(policy_stats: dict, expert_stats: dict) -> dict:
    divergence = {}
    for name in bc.BUTTONS:
        p = float(policy_stats["button_rate"].get(name, 0.0))
        e = float(expert_stats["button_rate"].get(name, 0.0))
        divergence[name] = abs(p - e)
    stick_diff = [abs(a - b) for a, b in
                  zip(policy_stats["stick_mean"], expert_stats["stick_mean"])]
    return {
        "button_rate_divergence": divergence,
        "mean_button_rate_divergence": float(np.mean(list(divergence.values())))
                                       if divergence else 0.0,
        "stick_mean_abs_diff": stick_diff,
        "trigger_rate_diff": abs(policy_stats["trigger_rate"] -
                                 expert_stats.get("trigger_rate", 0.0)),
    }


def expert_stats_from_data(data_dir: str, cfg: bc.BCConfig) -> dict:
    episodes = bc.load_episodes(data_dir, player=cfg.player,
                                normalize_obs=cfg.obs_normalize,
                                shield_from_status=cfg.shield_from_status)
    episodes = [bc.resample_episode(ep, cfg.frame_skip) for ep in episodes]
    samples = bc.samples_for(episodes, cfg, mirror=False)
    return bc.dataset_action_stats(samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint from gym_bc.py")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--mock", action="store_true",
                    help="use the built-in mock environment (no game)")
    ap.add_argument("--data", default=None,
                    help="expert recordings for the distribution comparison")
    ap.add_argument("--max-steps", type=int, default=1800,
                    help="decision steps per episode (12 Hz; env default 1800)")
    ap.add_argument("--start-menus", action="store_true",
                    help="walk boot -> match with the policy pad first (the pad "
                         "joins player 1 and drives the menus; needs the game "
                         "started with SSBU_KEYBOARD_PLAYER=0)")
    ap.add_argument("--sample", action="store_true",
                    help="draw actions from the predicted distribution instead "
                         "of the argmax: keeps a drifted BC policy moving")
    ap.add_argument("--button-threshold", type=float, default=None,
                    help="override every button threshold (the checkpoint's "
                         "per-button thresholds are tuned for offline F1 and "
                         "are silent in states the expert never visited)")
    ap.add_argument("--menu-timeout", type=float, default=120.0,
                    help="seconds to keep retrying the menu walk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--execute", type=int, default=2,
                    help="actions executed per plan (temporal ensemble)")
    ap.add_argument("--no-blend", action="store_true")
    ap.add_argument("--cstick", action="store_true",
                    help="run models trained with --cstick (the C-stick head "
                         "is restored from the checkpoint; this flag only "
                         "checks it is present)")
    ap.add_argument("--action-format", choices=("auto", "array", "dict"),
                    default="auto")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.episodes < 1:
        ap.error("--episodes must be >= 1")

    log = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731
    policy = bc.load_policy(args.model, device=args.device,
                            sample=bool(args.sample))
    cfg = policy.cfg
    if args.button_threshold is not None:
        policy.thresholds = np.full(bc.N_BUTTONS, float(args.button_threshold),
                                    np.float32)
        log(f"[eval] button thresholds forced to {args.button_threshold:.2f}")
    log(f"[eval] checkpoint config: window={cfg.window} horizon={cfg.horizon} "
        f"frame_skip={cfg.frame_skip} obs_normalize={cfg.obs_normalize} "
        f"trigger={cfg.trigger} cstick={cfg.cstick} dropout={cfg.dropout} "
        f"shield_from_status={cfg.shield_from_status}")
    if args.cstick and not cfg.cstick:
        log("[eval] warning: --cstick given but this checkpoint has no "
            "C-stick head; using the checkpoint config")
    ensemble = bc.TemporalEnsemble(policy, execute=args.execute,
                                   blend=0.0 if args.no_blend else 0.5)
    pad = None
    if args.start_menus and not args.mock:
        from rl.input.gym_input import VirtualPad
        pad = VirtualPad().create()
        log(f"[eval] pad up ({pad.name}); walking the menus to the match")
    try:
        env, is_mock = make_env(policy.cfg, args.episodes, args.max_steps,
                                args.seed, args.mock, log=log,
                                input_adapter=pad)
    except BaseException:
        if pad is not None:
            pad.destroy()
        raise
    try:
        if pad is not None and not drive_menus(env, pad, args.menu_timeout, log=log):
            log("[eval] giving up: the game is not in a fight (see the menu walk)")
            return 2
        result = run_episodes(env, ensemble, args.episodes, args.max_steps,
                              args.seed, args.action_format, is_mock, log=log)
    except Exception as exc:
        log(f"[eval] episode run failed: {type(exc).__name__}: {exc}")
        return 2
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        if pad is not None:
            pad.destroy()

    if args.data:
        expert = expert_stats_from_data(args.data, policy.cfg)
        source = args.data
    else:
        expert = policy.checkpoint.get("expert_stats")
        source = "checkpoint"
    if expert is None:
        expert = {"button_rate": {n: 0.0 for n in bc.BUTTONS},
                  "stick_mean": [0.0, 0.0], "trigger_rate": 0.0}
        source = "none"
    result["expert_stats"] = expert
    result["expert_stats_source"] = source
    result["comparison"] = compare_stats(result["policy_stats"], expert)
    result["checkpoint"] = os.path.abspath(args.model)
    result["config"] = {
        "window": cfg.window, "horizon": cfg.horizon,
        "frame_skip": cfg.frame_skip, "obs_normalize": cfg.obs_normalize,
        "trigger": cfg.trigger, "cstick": cfg.cstick,
        "dropout": cfg.dropout, "player": cfg.player,
        "hidden": cfg.hidden, "bins": cfg.bins,
        "shield_from_status": cfg.shield_from_status,
    }

    if args.json:
        print(json.dumps(bc.jsonable(result), indent=2))
    else:
        print(f"[eval] model {os.path.abspath(args.model)}"
              f"{' (MOCK)' if is_mock else ''}")
        print(f"[eval] episodes={result['episodes']} wins={result['wins']} "
              f"losses={result['losses']} unknown={result['unknown']} "
              f"win_rate={result['win_rate']:.3f}")
        print(f"[eval] mean length={result['mean_episode_length']} "
              f"p1_damage={result['mean_p1_damage']} "
              f"p2_damage={result['mean_p2_damage']}")
        print(f"[eval] button rate divergence vs {source}: "
              f"{result['comparison']['mean_button_rate_divergence']:.4f}")
        print("[eval] policy button rate: " + ", ".join(
            f"{n}={v:.3f}" for n, v in result["policy_stats"]["button_rate"].items()))
        print(f"[eval] policy stick: neutral {result['policy_stats']['stick_neutral_rate']:.3f}"
              f" | |x| {result['policy_stats']['stick_abs_mean'][0]:.2f}"
              f" | |y| {result['policy_stats']['stick_abs_mean'][1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
