#!/usr/bin/env python3
"""Export a vocab-mode BC/PPO checkpoint to a torch-free ``weights.npz``.

The ssbu-gym sample submission runs the policy with numpy only.  This writes
the weights it needs plus a ``golden.npz`` (random windows and the logits the
torch model gives for them) so the numpy port can be checked without torch.

    .venv/bin/python -m rl.scripts.export_numpy_policy \
        --checkpoint data/models/rl_v21_launch.pt --out-dir /path/to/submission
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from rl.bc import gym_bc as bc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--golden", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    ck = bc.load_checkpoint(a.checkpoint)
    cfg = bc.BCConfig(**{k: v for k, v in ck["config"].items()
                         if k in bc.BCConfig.__dataclass_fields__})
    if cfg.action_mode != "vocab" or cfg.gru_layers != 1:
        raise SystemExit("only vocab-mode, single-layer GRU checkpoints")
    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in ck["state_dict"].items()}
    need = ["encoder.weight_ih_l0", "encoder.weight_hh_l0", "encoder.bias_ih_l0",
            "encoder.bias_hh_l0", "trunk.0.weight", "trunk.0.bias",
            "trunk.3.weight", "trunk.3.bias", "action_head.weight",
            "action_head.bias"]
    out = {k.replace(".", "__"): sd[k] for k in need}
    out["norm_mean"] = np.asarray(ck["norm"]["mean"], np.float32)
    out["norm_std"] = np.asarray(ck["norm"]["std"], np.float32)
    out["vocab_keys"] = np.asarray(ck["vocab"]["keys"], np.int64)
    out["config"] = np.asarray([cfg.obs_dim, cfg.window, cfg.horizon, cfg.hidden,
                                cfg.vocab_size, cfg.prev_dim], np.int64)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, "weights.npz"), **out)

    # golden: logits of the first horizon step for random windows
    rng = np.random.default_rng(a.seed)
    net = bc.BCPolicyNet(cfg)
    net.load_state_dict(ck["state_dict"], strict=False)
    net.eval()
    mean, std = out["norm_mean"], out["norm_std"]
    raw = (mean + std * rng.normal(size=(a.golden, cfg.window, cfg.obs_dim))
           ).astype(np.float32)
    prev = (rng.random((a.golden, cfg.prev_dim)) < 0.2).astype(np.float32)
    with torch.no_grad():
        logits = net(torch.from_numpy((raw - mean) / std),
                     torch.from_numpy(prev))["action"][:, 0].numpy()
    np.savez_compressed(os.path.join(a.out_dir, "golden.npz"),
                        windows=raw, prev=prev, logits=logits)
    print(f"wrote {a.out_dir}/weights.npz and golden.npz "
          f"({cfg.obs_dim}-dim obs, window {cfg.window}, vocab {cfg.vocab_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
