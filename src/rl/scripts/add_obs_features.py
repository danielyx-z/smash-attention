#!/usr/bin/env python3
"""Add memory features to a trained model's input without retraining.

The observation gets k extra values (``rl.env.gym_memory``).  The GRU's input
weight matrix is widened by k **zero** columns, the normalisation gets
mean 0 / std 1 for them and ``config.obs_dim`` / ``config.extra_obs`` are
updated.  The widened model computes exactly what the original did for any
value of the new inputs, so nothing is lost; training then learns to use them.
(The optimizer state is not part of a checkpoint, so nothing to widen there.)

    python -m rl.scripts.add_obs_features --model data/models/rl_v13_lock.pt \
        --out data/models/rl_v13_upb.pt --features ego_upb_used
"""
from __future__ import annotations

import argparse
import copy
import sys

import torch

from rl.env.gym_memory import EXTRA_OBS_NAMES


def widen_checkpoint(ck: dict, features: list[str]) -> dict:
    unknown = [f for f in features if f not in EXTRA_OBS_NAMES]
    if unknown:
        raise ValueError(f"unknown features {unknown}; known {EXTRA_OBS_NAMES}")
    cfg = dict(ck["config"])
    have = list(cfg.get("extra_obs") or [])
    new = [f for f in features if f not in have]
    if not new:
        raise ValueError(f"model already has {features}")
    if int(cfg.get("gru_layers", 1)) != 1:
        raise ValueError("only single-layer GRU encoders are supported")
    k = len(new)
    out = copy.deepcopy(ck)
    sd = out["state_dict"]
    w = sd["encoder.weight_ih_l0"]                      # (3H, obs_dim)
    if int(w.shape[1]) != int(cfg["obs_dim"]):
        raise ValueError(f"encoder input {w.shape[1]} != obs_dim "
                         f"{cfg['obs_dim']}")
    sd["encoder.weight_ih_l0"] = torch.cat(
        [w, torch.zeros(w.shape[0], k, dtype=w.dtype)], dim=1)
    norm = out["norm"]
    norm["mean"] = list(norm["mean"]) + [0.0] * k
    norm["std"] = list(norm["std"]) + [1.0] * k
    cfg["obs_dim"] = int(cfg["obs_dim"]) + k
    cfg["extra_obs"] = have + new
    out["config"] = cfg
    meta = dict(out.get("meta") or {})
    meta["widened_from_obs_dim"] = int(ck["config"]["obs_dim"])
    meta["extra_obs"] = cfg["extra_obs"]
    out["meta"] = meta
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", nargs="+", default=["ego_upb_used"])
    args = ap.parse_args(argv)
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    out = widen_checkpoint(ck, args.features)
    torch.save(out, args.out)
    print(f"{args.model} (obs_dim {ck['config']['obs_dim']}) -> {args.out} "
          f"(obs_dim {out['config']['obs_dim']}, extra {out['config']['extra_obs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
