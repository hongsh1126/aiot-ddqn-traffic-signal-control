"""
Final experiment script for the AIoT-DDQN paper.

Runs the full benchmark suite:
  1. Single-intersection comparison: 3 demand levels x 6 controllers
  2. 2x2 grid network: Medium demand x 6 controllers
  3. V2X channel ablation: 4 channel conditions on AIoT-DDQN
  4. Reward-weight ablation: 5 weight settings
  5. Sensing-modality ablation: 4 sensing configurations
All RL controllers receive an equal training budget. Numbers reported are
post-training evaluation on a held-out seed.
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_sim import (Intersection, GridNetwork, V2XChannel,
                          N_LANES_PER_INTERSECTION)
from controllers import (FixedTimeController, ActuatedController,
                          MaxPressureController, DQNController,
                          DDQNController, IntelliLightLite,
                          CommunicationAwareWrapper, BaseController)

DT = 0.5
WARMUP = 200
EVAL = 800
TRAIN = 1800
DEMANDS = {"Low": 200, "Medium": 300, "High": 400}


def run_single(controller, demand, duration, seed):
    intr = Intersection(iid=0)
    rng = np.random.default_rng(seed)
    rate_dt = (demand / 3600.0) * DT
    t = 0.0
    while t < duration:
        for lane in range(N_LANES_PER_INTERSECTION):
            if rng.random() < rate_dt:
                intr.inject(lane, t)
        controller.act(intr, DT)
        intr.step(DT, t)
        t += DT
    completed = [v for v in intr.completed_vehicles
                  if v.completion_time > WARMUP]
    avg_w = float(np.mean([v.waited_total for v in completed])) if completed else 0.0
    thru = len(completed) * 3600.0 / max(duration - WARMUP, 1) if completed else 0.0
    queue = sum(sum(1 for v in lane if v.v < 1.0) for lane in intr.lanes)
    em_per_veh = intr.cum_emission_g / max(len(intr.completed_vehicles), 1)
    return {"avg_wait_s": avg_w, "queue_veh": float(queue),
            "throughput_vph": thru, "emission_g_per_veh": em_per_veh,
            "n_completed": len(completed)}


def run_grid(factory, rows, cols, demand, duration, seed):
    net = GridNetwork(rows=rows, cols=cols, seed=seed)
    ctrls = [factory() for _ in net.intersections]
    t = 0.0
    while t < duration:
        for intr, c in zip(net.intersections, ctrls):
            c.act(intr, DT)
        net.step(DT, demand)
        t += DT
    all_eval = []
    for intr in net.intersections:
        all_eval.extend([v for v in intr.completed_vehicles
                         if v.completion_time > WARMUP])
    avg_w = float(np.mean([v.waited_total for v in all_eval])) if all_eval else 0.0
    thru = len(all_eval) * 3600.0 / max(duration - WARMUP, 1) if all_eval else 0.0
    queue = sum(sum(sum(1 for v in lane if v.v < 1.0)
                    for lane in intr.lanes)
                for intr in net.intersections)
    em_total = sum(intr.cum_emission_g for intr in net.intersections)
    em_per_veh = em_total / max(sum(len(i.completed_vehicles)
                                     for i in net.intersections), 1)
    return {"avg_wait_s": avg_w, "queue_veh": float(queue),
            "throughput_vph": thru, "emission_g_per_veh": em_per_veh,
            "n_completed": len(all_eval)}


def make(name, seed=2026):
    if name == "Fixed-Time":   return FixedTimeController()
    if name == "Actuated":     return ActuatedController()
    if name == "MaxPressure":  return MaxPressureController()
    if name == "DQN":          return DQNController(seed=seed)
    if name == "IntelliLight": return IntelliLightLite(seed=seed + 100)
    if name == "AIoT-DDQN":    return DDQNController(seed=seed + 200)


def train_rl(name, demand, seed=2026):
    ctrl = make(name, seed=seed)
    if name in ("DQN", "IntelliLight", "AIoT-DDQN"):
        run_single(ctrl, demand, TRAIN, seed=seed + 1)
        ctrl.eps_start = ctrl.eps_end = 0.05
        ctrl.steps = ctrl.eps_decay_steps
    return ctrl


def main():
    SEED = 2026
    duration = WARMUP + EVAL
    results = {"single": {}, "grid": {}, "v2x": {},
               "state_ablation": {}, "reward_ablation": {}}

    print("\n=== 1. Single 4-arm intersection ===")
    for d_name, d_val in DEMANDS.items():
        print(f"\n[demand={d_name} ({d_val} vph/lane)]")
        results["single"][d_name] = {}
        for cname in ["Fixed-Time", "Actuated", "MaxPressure",
                      "DQN", "IntelliLight", "AIoT-DDQN"]:
            t0 = time.time()
            ctrl = train_rl(cname, d_val, seed=SEED)
            m = run_single(ctrl, d_val, duration, seed=SEED + 100)
            results["single"][d_name][cname] = m
            print(f"  {cname:14s} wait={m['avg_wait_s']:6.2f}s "
                  f"queue={m['queue_veh']:5.1f} thru={m['throughput_vph']:6.0f}vph "
                  f"emit={m['emission_g_per_veh']:5.2f}g/veh n={m['n_completed']:4d} "
                  f"({time.time()-t0:.1f}s)")

    print("\n=== 2. 2x2 grid network (Medium demand) ===")
    for cname in ["Fixed-Time", "Actuated", "MaxPressure",
                  "DQN", "IntelliLight", "AIoT-DDQN"]:
        t0 = time.time()
        if cname in ("DQN", "IntelliLight", "AIoT-DDQN"):
            shared = train_rl(cname, DEMANDS["Medium"], seed=SEED)
            def factory(c=cname, s=shared):
                ctrl = make(c, seed=SEED + 999)
                ctrl.q.W1 = s.q.W1.copy(); ctrl.q.b1 = s.q.b1.copy()
                ctrl.q.W2 = s.q.W2.copy(); ctrl.q.b2 = s.q.b2.copy()
                ctrl.q.W3 = s.q.W3.copy(); ctrl.q.b3 = s.q.b3.copy()
                ctrl.q_targ.copy_from(ctrl.q, 1.0)
                ctrl.eps_start = ctrl.eps_end = 0.05
                ctrl.steps = ctrl.eps_decay_steps
                return ctrl
        else:
            factory = lambda c=cname: make(c)
        m = run_grid(factory, 2, 2, DEMANDS["Medium"], duration, seed=SEED + 200)
        results["grid"][cname] = m
        print(f"  {cname:14s} wait={m['avg_wait_s']:6.2f}s "
              f"queue={m['queue_veh']:5.1f} thru={m['throughput_vph']:6.0f}vph "
              f"emit={m['emission_g_per_veh']:5.2f}g/veh n={m['n_completed']:5d} "
              f"({time.time()-t0:.1f}s)")

    print("\n=== 3. V2X channel-condition sensitivity (AIoT-DDQN, Medium) ===")
    conditions = [
        ("Ideal",        0.0,    0.0,   0.0),
        ("Low load",     8.0,    0.02,  0.10),
        ("Medium load",  25.0,   0.10,  0.30),
        ("High load",    100.0,  0.30,  0.60),
    ]
    for cname, delay, loss, load in conditions:
        ch = V2XChannel(nominal_delay_ms=delay, base_loss_rate=loss,
                         rng=np.random.default_rng(SEED + 500))
        ddqn = train_rl("AIoT-DDQN", DEMANDS["Medium"], seed=SEED)
        wrapped = CommunicationAwareWrapper(ddqn, ch, channel_load=load)
        t0 = time.time()
        m = run_single(wrapped, DEMANDS["Medium"], duration, seed=SEED + 600)
        results["v2x"][cname] = {"delay_ms": delay, "loss": loss, **m}
        print(f"  {cname:13s} delay={delay:5.1f}ms loss={loss:.2f}  "
              f"wait={m['avg_wait_s']:6.2f}s thru={m['throughput_vph']:6.0f}vph "
              f"({time.time()-t0:.1f}s)")

    print("\n=== 4. Reward-weight ablation ===")
    settings = [
        ("Wait only",        (1.0, 0.0, 0.0, 0.0)),
        ("Wait+Queue",       (0.5, 0.5, 0.0, 0.0)),
        ("Default",          (0.4, 0.3, 0.2, 0.1)),
        ("Wait+Throughput",  (0.5, 0.0, 0.5, 0.0)),
        ("Emission-heavy",   (0.3, 0.2, 0.2, 0.3)),
    ]
    for label, (a, b, g, dlt) in settings:
        class CustomReward(DDQNController):
            def reward(self, intr, prev_metrics):
                s = intr.state_vector()
                queue = s[4:8]; total_q = float(queue.sum())
                w_max_norm = s[8]
                cur = intr.metrics(window_s=30.0, time_now=0.0)
                recent_thr = cur["throughput_vph"] / 3600.0
                em_rate = cur["emission_g_total"] / max(intr.next_vid, 1)
                r = (- a * w_max_norm - b * total_q / 30.0
                     + g * recent_thr - dlt * em_rate / 100.0)
                return float(np.clip(r, -5.0, 5.0))
        ctrl = CustomReward(seed=SEED + 800)
        run_single(ctrl, DEMANDS["Medium"], TRAIN, seed=SEED + 800)
        ctrl.eps_start = ctrl.eps_end = 0.05; ctrl.steps = ctrl.eps_decay_steps
        m = run_single(ctrl, DEMANDS["Medium"], duration, seed=SEED + 801)
        results["reward_ablation"][label] = {"weights": (a, b, g, dlt), **m}
        print(f"  {label:18s} α,β,γ,δ=({a:.1f},{b:.1f},{g:.1f},{dlt:.1f}) "
              f"wait={m['avg_wait_s']:6.2f}s thru={m['throughput_vph']:6.0f}vph")

    print("\n=== 5. Sensing-modality ablation ===")
    masks = {
        "RSU only (loop)":   np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
        "Loop + queue":      np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=np.float32),
        "Full smart pole":   np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        "No-phase-context":  np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.float32),
    }
    for cname, mask in masks.items():
        class MaskedDDQN(DDQNController):
            def __init__(self, **kw):
                super().__init__(**kw); self.mask = mask
            def act(self, intr, dt):
                orig = intr.state_vector
                def masked(): return orig() * self.mask
                intr.state_vector = masked
                super().act(intr, dt)
                intr.state_vector = orig
        ctrl = MaskedDDQN(seed=SEED + 700)
        run_single(ctrl, DEMANDS["Medium"], TRAIN, seed=SEED + 700)
        ctrl.eps_start = ctrl.eps_end = 0.05; ctrl.steps = ctrl.eps_decay_steps
        m = run_single(ctrl, DEMANDS["Medium"], duration, seed=SEED + 701)
        results["state_ablation"][cname] = m
        print(f"  {cname:24s} wait={m['avg_wait_s']:6.2f}s "
              f"thru={m['throughput_vph']:6.0f}vph")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "results_aiot.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nDone. results_aiot.json saved.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.1f}s")
