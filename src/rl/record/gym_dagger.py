#!/usr/bin/env python3
"""DAgger collection: the policy plays, the CPU is handed player 1 periodically.

    .venv/bin/python -m rl.record.gym_dagger \
        --model data/models/bc_dk9_30hz_cwbc.pt --out data/il/dagger1 \
        --episodes 20 --episode-frames 5400 \
        --handover-every 120 --handover-frames 40

Why it works without touching the AI: the runtime can hand player 1 to the CPU
for N presented frames (control block command ``handover``).  The CPU then
writes player 1's control module, which the state block publishes as that
fighter's buttons/stick — that is the expert's action *in the state the policy
produced*.  So the label needs no extra plumbing, and the segments also pull the
trajectory back towards expert play.

Every recorded frame carries ``expert_mask`` (1 while the CPU was driving), so
the trainer can up-weight exactly the correction frames.

The game must be in a match with player 1 human and the pad joined (the same
setup as ``live_eval.sh``: SSBU_KEYBOARD_PLAYER=0, SSBU_INPUT_NO_FOCUS=1,
SSBU_RL_MODE=1).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import numpy as np

from rl.bc import gym_bc as bc
from rl.env.gym_env import RawSnapshotReader
from rl.input import control_block as cb
from rl.input.gym_input import VirtualPad
from rl.record import gym_record as rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint the policy plays from")
    ap.add_argument("--out", required=True, help="recording directory")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-frames", type=int, default=5400)
    ap.add_argument("--handover-every", type=int, default=120,
                    help="decisions between takeovers (0 disables)")
    ap.add_argument("--handover-frames", type=int, default=40,
                    help="presented frames the CPU drives player 1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sample", action="store_true", default=True,
                    help="stochastic decoding (default on: an argmax policy "
                         "stands still, and the states then carry no information)")
    ap.add_argument("--deterministic", dest="sample", action="store_false")
    ap.add_argument("--reset-every", type=float, default=3.0,
                    help="seconds between reset requests while no episode runs")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    log = lambda *a: print(*a, flush=True)  # noqa: E731
    policy = bc.load_policy(args.model, device=args.device, sample=args.sample)
    skip = max(1, int(policy.cfg.frame_skip))
    log(f"[dagger] policy {os.path.basename(args.model)} frame_skip {skip} "
        f"sample {args.sample} | handover every {args.handover_every} decisions "
        f"for {args.handover_frames} frames")

    reader = RawSnapshotReader()
    core = rec.RecorderCore(args.out, episodes=args.episodes, frame_skip=skip,
                            meta={"source": "dagger", "model": args.model,
                                  "handover_every": args.handover_every,
                                  "handover_frames": args.handover_frames},
                            reset=True, episode_frames=args.episode_frames,
                            log=log)
    if core.finished:
        log(f"[dagger] {len(core.done)} episodes already in {args.out}")
        return 0

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    ctl = cb.ControlBlock()
    ctl.open()
    pad = VirtualPad().create()
    stats = {"decisions": 0, "handovers": 0, "expert_frames": 0, "frames": 0}
    last_seq = -1
    next_reset = time.monotonic()
    try:
        while not stop["flag"] and not core.finished:
            frame = reader.read()
            if frame is None or frame["seq"] == last_seq:
                if (core.ep is None and frame is not None
                        and frame["fight_on"] == 0
                        and time.monotonic() >= next_reset):
                    next_reset = time.monotonic() + args.reset_every
                    try:
                        ctl.request_reset()
                    except Exception as exc:      # noqa: BLE001 - keep collecting
                        log(f"[dagger] reset request failed: {exc}")
                time.sleep(0.001)
                continue
            last_seq = frame["seq"]
            before = len(core.ep) if core.ep is not None else -1
            event = core.feed(frame)
            stats["frames"] += 1

            # the CPU drives player 1 while flags bit 4 is set (runtime echo)
            snap = ctl.read()
            expert = 1 if (snap.flags & 0x10) else 0
            if expert:
                stats["expert_frames"] += 1
            if core.ep is not None and len(core.ep) > before:
                mask = core.ep.extra.setdefault("expert_mask", [])
                mask.extend([expert] * (len(core.ep) - max(before, 0)))

            if core.ep is None or not frame["fight_on"]:
                continue

            # policy decision every `skip` frames
            if stats["frames"] % skip == 0 and not expert:
                s = reader.read_latest(timeout=0.5)
                obs = rec_obs(s, policy.cfg.obs_normalize)
                action = policy.act(obs)
                pad.apply(bc.action_to_env(action, "dict"))
                stats["decisions"] += 1
                if (args.handover_every > 0
                        and stats["decisions"] % args.handover_every == 0):
                    snap = ctl.send_command(cb.CMD_HANDOVER,
                                            args.handover_frames)
                    stats["handovers"] += 1
                    log(f"[dagger] handover {stats['handovers']} at decision "
                        f"{stats['decisions']} for {args.handover_frames} frames "
                        f"(flags {snap.flags:#x})")

            if event == "split":
                log(f"[dagger] episode cut at {args.episode_frames} frames; "
                    f"requesting an in-fight rematch")
                try:
                    ctl.request_reset()
                except Exception as exc:          # noqa: BLE001
                    log(f"[dagger] reset failed: {exc}")
    finally:
        try:
            pad.destroy()
        finally:
            ctl.close()
    log(f"[dagger] frames {stats['frames']} decisions {stats['decisions']} "
        f"handovers {stats['handovers']} expert frames {stats['expert_frames']} "
        f"episodes {len(core.done)}")
    if args.json:
        import json
        print(json.dumps({"frames": stats["frames"],
                          "decisions": stats["decisions"],
                          "handovers": stats["handovers"],
                          "expert_frames": stats["expert_frames"],
                          "episodes": sorted(core.done)}))
    return 0


def rec_obs(snapshot, normalize_obs: bool) -> np.ndarray:
    """The policy observation for this snapshot (same builder as the env)."""
    from rl.env import gym_obs
    return gym_obs.PolicyObservation(normalize=normalize_obs).build(snapshot)


if __name__ == "__main__":
    sys.exit(main())
