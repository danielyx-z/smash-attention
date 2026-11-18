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
            "sum_dealt": 0.0,
            "sum_taken": 0.0,
            "sum_diff": 0.0,
            "last_dealt": 0.0,
            "last_taken": 0.0,
            "last_diff": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_bc": 0.0,
            "updated": time.time(),
        }
        self.flush()

    def _derived(self) -> None:
        n = max(1, int(self.data["damage_episodes"]))
        eps = max(1, int(self.data["episodes"]))
        self.data["mean_dealt"] = self.data["sum_dealt"] / n
        self.data["mean_taken"] = self.data["sum_taken"] / n
        self.data["mean_diff"] = self.data["sum_diff"] / n
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

    def record_episode(self, row: dict) -> None:
        self.data["episodes"] += 1
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
    if model.value_head is not None:
        # Start the critic at zero: PPO learns the value scale from the
        # damage reward instead of trusting random initial outputs.
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


def _episode_row(info: dict, length: int, ret: float,
                 terminated: bool, truncated: bool,
                 peak: dict | None = None) -> dict:
    """Summarize one episode.

    The last info frame can be a respawn with damage reset to 0, so the final
    values alone can report a false 0/0.  ``peak`` keeps the largest damage
    seen during the episode, matching gym_eval's reporting.
    """
    r = ge._result_from_info(info)
    p1, p2 = r.get("p1_damage"), r.get("p2_damage")
    winner = r.get("winner")
    if peak:
        if p1 in (None, 0.0) and peak.get("p1_damage", 0.0) > 0.0:
            p1 = peak["p1_damage"]
        if p2 in (None, 0.0) and peak.get("p2_damage", 0.0) > 0.0:
            p2 = peak["p2_damage"]
    return {"length": int(length), "return": float(ret),
            "p1_damage": p1, "p2_damage": p2, "winner": winner,
            "terminated": bool(terminated), "truncated": bool(truncated)}


def collect_rollout(env, model, state: PolicyState, cfg: bc.BCConfig,
                    vocab: bc.ActionVocab, mean, std, device: str,
                    steps: int, seed: int,
                    on_episode=None) -> tuple[dict, list[dict]]:
    """Run the live env, returning flat transition arrays and episode rows."""
    model.eval()
    windows, prevs, actions, logps, values, rewards = [], [], [], [], [], []
    next_windows, next_prevs, terminals, truncateds = [], [], [], []
    episodes: list[dict] = []
    obs, info = ge.reset_env(env, seed)
    state.reset()
    state.observe(ge._extract_obs(obs))
    ep = 0
    ep_ret = 0.0
    ep_len = 0
    peak = {"p1_damage": 0.0, "p2_damage": 0.0}
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
        ep_ret += float(reward)
        ep_len += 1
        step_result = ge._result_from_info(info)
        for key in peak:
            value = step_result.get(key)
            if value is not None:
                peak[key] = max(peak[key], float(value))
        if terminated or truncated:
            row = _episode_row(info, ep_len, ep_ret,
                               terminated, truncated, peak)
            episodes.append(row)
            if on_episode is not None:
                on_episode(row)
            ep += 1
            obs, info = ge.reset_env(env, seed + ep)
            state.reset()
            state.observe(ge._extract_obs(obs))
            ep_ret = 0.0
            ep_len = 0
            peak = {"p1_damage": 0.0, "p2_damage": 0.0}
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
                 batch_size: int = 512) -> np.ndarray:
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
                 batch_size: int = 512) -> np.ndarray:
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
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(out["value"], returns)
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
            "stock_value": float(getattr(args, "stock_value", 0.0)),
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
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--stock-value", type=float, default=1.0,
                    help="reward units for each lost/earned stock "
                         "(1.0 = 100%% damage; 0 disables stock terms)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--stats-file", default="/tmp/ssbu_training_stats.json",
                    help="JSON status file for the live viewer overlay")
    ap.add_argument("--cpu-level", type=int, default=0,
                    help="opponent CPU level shown in the overlay")
    ap.add_argument("--start-menus", action="store_true",
                    help="create the pad and walk boot -> match first")
    ap.add_argument("--game-log", default="~/smash.log")
    ap.add_argument("--pad-timeout", type=float, default=15.0)
    ap.add_argument("--menu-timeout", type=float, default=120.0)
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
        weight_decay=args.weight_decay, seed=args.seed,
        device=args.device)
    ppo.check()
    if float(args.stock_value) < 0.0:
        ap.error("--stock-value must be >= 0")

    log = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731
    torch.manual_seed(ppo.seed)
    np.random.seed(ppo.seed)
    device = str(ppo.device)
    model, cfg, vocab, mean, std, bc_ckpt = load_bc_model(
        args.model, device, dropout=args.dropout)
    model.to(device)
    ref_model = None
    if ppo.kl_coef > 0.0:
        ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
    log(f"[ppo] warm start {args.model}: vocab {vocab.size} classes, "
        f"window {cfg.window}, horizon {cfg.horizon}, frame_skip "
        f"{cfg.frame_skip}, damage {cfg.damage_mode}, "
        f"kl_bc={ppo.kl_coef:g}, stock_value={args.stock_value:g}")
    stats_writer = TrainingStats(args.stats_file, args.cpu_level,
                                 ppo.iterations)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(ppo.lr),
                                  weight_decay=float(ppo.weight_decay))
    pad = None
    env = None
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
                damage_scale=0.01, damage_dealt=1.0, damage_taken=1.0,
                ko_bonus=stock_value, stock_loss=-stock_value,
                win_bonus=0.0, loss_penalty=0.0, draw_reward=0.0,
                ego_slot=0)
            env = SSBUGymEnv(
                normalize_obs=cfg.obs_normalize, damage_mode=cfg.damage_mode,
                frame_skip=cfg.frame_skip, max_steps=ppo.max_steps,
                reward_mode="combat", reward_config=reward_cfg,
                stock_termination=False, require_positive_stock=False,
                input_adapter=pad)
            if pad is not None:
                ge.wait_for_pad(args.game_log, args.pad_timeout, log=log)
                if not ge.drive_menus(env, pad, args.menu_timeout, log=log):
                    log("[ppo] giving up: no fight")
                    return 2
        state = PolicyState(cfg)
        for iteration in range(1, int(ppo.iterations) + 1):
            t0 = time.monotonic()
            batch, episodes = collect_rollout(
                env, model, state, cfg, vocab, mean, std, device,
                int(ppo.rollout_steps), int(ppo.seed) + 1000 * iteration,
                on_episode=stats_writer.record_episode)
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
                log(f"[ppo] iter {iteration}/{ppo.iterations}: "
                    f"steps {len(batch['actions'])} eps {len(episodes)} "
                    f"dealt {np.mean(dealt) if dealt else float('nan'):.1f} "
                    f"taken {np.mean(taken) if taken else float('nan'):.1f} "
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
        if pad is not None:
            pad.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
