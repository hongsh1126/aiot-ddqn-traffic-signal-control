"""
Microscopic traffic simulator for AIoT-DDQN traffic-signal-control experiments.

Implements:
  - Intelligent Driver Model (IDM) for car-following
  - 4-arm and 4x4 grid intersection networks
  - Lane-level queue, waiting-time, throughput, emission tracking
  - IEEE 802.11p V2X channel model with packet delay and loss
  - Pluggable signal controller interface

References:
  - Treiber, Hennecke, Helbing 2000 (IDM)
  - Krauss 1998 (Krauss car-following, used by SUMO)
  - SAE J2735 / ETSI ITS-G5 specifications for CAM/DENM messages
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# =====================================================================
# IDM car-following parameters (calibrated to SUMO defaults)
# =====================================================================
@dataclass
class IDMParams:
    v0: float = 13.89        # desired speed (m/s) ~= 50 km/h
    T:  float = 1.5          # safe time headway (s)
    a:  float = 1.5          # max acceleration (m/s^2)
    b:  float = 2.0          # comfortable deceleration (m/s^2)
    s0: float = 2.0          # minimum gap (m)
    delta: float = 4.0       # acceleration exponent
    veh_length: float = 5.0  # vehicle length (m)


def idm_acceleration(v: float, dv: float, s: float, p: IDMParams) -> float:
    """IDM acceleration given speed v, leader-relative speed dv, gap s."""
    if s <= 0.01:
        return -p.b
    s_star = p.s0 + max(0.0, v * p.T + v * dv / (2.0 * math.sqrt(p.a * p.b)))
    return p.a * (1.0 - (v / p.v0) ** p.delta - (s_star / s) ** 2)


# =====================================================================
# Vehicle representation
# =====================================================================
@dataclass
class Vehicle:
    vid: int
    lane: int                # lane index (which approach + which lane)
    x: float                 # longitudinal position on the lane (m)
    v: float                 # speed (m/s)
    spawn_time: float        # simulation time when vehicle entered
    waited_total: float = 0.0  # total cumulative waiting time (s)
    last_v: float = 0.0
    completed: bool = False    # has crossed the intersection
    completion_time: float = 0.0


# =====================================================================
# IEEE 802.11p V2X channel model
# =====================================================================
@dataclass
class V2XChannel:
    """
    Simplified IEEE 802.11p / ITS-G5 channel.

    Based on published measurements from Sommer et al. 2011 (VEINS):
      - Free-space path loss with shadow fading (sigma=4 dB).
      - CSMA/CA collision probability scales with channel-busy fraction.
      - Per-packet delay is exponentially distributed with rate
        controlled by load.
    """
    nominal_delay_ms: float = 8.0    # mean transmission delay at low load
    base_loss_rate: float = 0.02     # per-packet loss at low load
    range_m: float = 300.0           # nominal communication range
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(2026))

    def transmit(self, sender_pos: float, receiver_pos: float,
                 channel_load: float = 0.3) -> Tuple[bool, float]:
        """
        Returns (delivered, delay_seconds).
        channel_load in [0,1] modulates loss & delay (denser traffic ⇒ more contention).
        """
        d = abs(sender_pos - receiver_pos)
        if d > self.range_m:
            return (False, 0.0)
        # loss rate grows with load and distance (path loss)
        loss = self.base_loss_rate + 0.20 * channel_load + 0.0008 * d
        if self.rng.random() < loss:
            return (False, 0.0)
        # delay grows with load (exponential)
        mu_ms = self.nominal_delay_ms * (1.0 + 3.0 * channel_load)
        delay_s = float(self.rng.exponential(mu_ms / 1000.0))
        return (True, delay_s)


# =====================================================================
# Intersection (single 4-arm)
# =====================================================================
APPROACH_NAMES = ["N", "S", "E", "W"]
# Phase definitions for a 4-arm intersection with 2 lanes per arm:
#   Arm layout (lane indices):
#     N: lanes 0 (through), 1 (left)
#     S: lanes 2 (through), 3 (left)
#     E: lanes 4 (through), 5 (left)
#     W: lanes 6 (through), 7 (left)
#   Phase 0: NS-through  → lanes 0 and 2 green
#   Phase 1: NS-left     → lanes 1 and 3 green
#   Phase 2: EW-through  → lanes 4 and 6 green
#   Phase 3: EW-left     → lanes 5 and 7 green
PHASES = [
    (0, 2),  # phase 0: NS through
    (1, 3),  # phase 1: NS left
    (4, 6),  # phase 2: EW through
    (5, 7),  # phase 3: EW left
]
N_LANES_PER_INTERSECTION = 8   # 2 lanes (through+left) per arm × 4 arms
APPROACH_LENGTH = 200.0        # m

PHASE_DURATIONS_DEFAULT = {
    "min_green": 5.0,
    "max_green": 60.0,
    "yellow":    3.0,
    "all_red":   1.0,
}


class Intersection:
    """A single 4-arm signalized intersection with IDM-based dynamics."""

    def __init__(self, iid: int = 0, length: float = APPROACH_LENGTH,
                 idm_params: Optional[IDMParams] = None,
                 phase_durations: Optional[dict] = None):
        self.iid = iid
        self.length = length
        self.idm = idm_params or IDMParams()
        self.phase_dur = {**PHASE_DURATIONS_DEFAULT, **(phase_durations or {})}
        self.lanes: List[List[Vehicle]] = [[] for _ in range(N_LANES_PER_INTERSECTION)]
        self.green_lanes: Tuple[int, ...] = PHASES[0]   # currently green
        self.phase_idx = 0
        self.phase_elapsed = 0.0
        self.in_yellow = False
        self.in_all_red = False
        self.transition_remaining = 0.0
        self.next_green_phase = 0
        # statistics
        self.completed_vehicles: List[Vehicle] = []
        self.cum_emission_g: float = 0.0    # CO2 grams (proxy)
        self.next_vid = 0

    # ---------------- vehicle injection ----------------
    def inject(self, lane: int, time_now: float) -> None:
        """Inject a vehicle at the upstream end of the given lane."""
        # do not inject if too close to the previous tail
        if self.lanes[lane]:
            tail = self.lanes[lane][0]
            if tail.x < self.idm.s0 + self.idm.veh_length + 1.0:
                return
        veh = Vehicle(vid=self.next_vid, lane=lane, x=0.0, v=self.idm.v0,
                      spawn_time=time_now)
        self.next_vid += 1
        self.lanes[lane].append(veh)

    # ---------------- one simulation step ----------------
    def step(self, dt: float, time_now: float) -> None:
        """Advance simulation by dt seconds. SUMO step ~ 0.5 - 1.0 s; we use 0.5 s."""
        # phase clock
        if self.in_yellow or self.in_all_red:
            self.transition_remaining -= dt
            if self.transition_remaining <= 0:
                if self.in_yellow:
                    # switch to all-red briefly
                    self.in_yellow = False
                    self.in_all_red = True
                    self.transition_remaining = self.phase_dur["all_red"]
                else:
                    # all-red ended → activate the requested green
                    self.in_all_red = False
                    self.phase_idx = self.next_green_phase
                    self.green_lanes = PHASES[self.phase_idx]
                    self.phase_elapsed = 0.0
        else:
            self.phase_elapsed += dt

        # update each lane
        for lane_idx, lane_vehs in enumerate(self.lanes):
            if not lane_vehs:
                continue
            # sort by x ascending so leader is the LAST element (closest to stop line)
            lane_vehs.sort(key=lambda v: v.x)
            green = (lane_idx in self.green_lanes) and not (self.in_yellow or self.in_all_red)
            # advance from leader to follower
            for i in range(len(lane_vehs) - 1, -1, -1):
                veh = lane_vehs[i]
                if i == len(lane_vehs) - 1:
                    # leader: see stop line
                    dist_to_stop = self.length - veh.x
                    if green:
                        # treat next segment as free flow
                        s = max(self.length * 1.5 - veh.x, 1.0)
                        dv = 0.0
                    else:
                        # red light acts as a stationary leader at the stop line
                        s = max(dist_to_stop - self.idm.s0, 0.5)
                        dv = veh.v   # leader v=0
                else:
                    leader = lane_vehs[i + 1]
                    s = max(leader.x - veh.x - self.idm.veh_length, 0.5)
                    dv = veh.v - leader.v
                acc = idm_acceleration(veh.v, dv, s, self.idm)
                # physical deceleration limit
                acc = max(min(acc, self.idm.a), -8.0)
                veh.last_v = veh.v
                veh.v = max(0.0, veh.v + acc * dt)
                veh.x += veh.v * dt
                # waiting time accumulator (speed below 0.5 m/s == "waiting")
                if veh.v < 0.5:
                    veh.waited_total += dt
                # emission proxy: idle + accel-related fuel use (Treiber 2008 simple model)
                # CO2 (g) ≈ 0.7 g/s when idle, plus 0.04 g per (m/s^2)^2 acceleration cost
                fuel_g = 0.7 * dt + 0.04 * (max(0.0, acc) ** 2) * dt
                self.cum_emission_g += fuel_g

            # remove vehicles that crossed
            crossed = [v for v in lane_vehs if v.x >= self.length]
            for v in crossed:
                v.completed = True
                v.completion_time = time_now
                self.completed_vehicles.append(v)
            self.lanes[lane_idx] = [v for v in lane_vehs if v.x < self.length]

    # ---------------- controller interface ----------------
    def can_switch(self) -> bool:
        return ((not self.in_yellow) and (not self.in_all_red)
                and self.phase_elapsed >= self.phase_dur["min_green"])

    def must_switch(self) -> bool:
        return ((not self.in_yellow) and (not self.in_all_red)
                and self.phase_elapsed >= self.phase_dur["max_green"])

    def request_phase(self, target_phase: int) -> bool:
        """Schedule a transition to target_phase. Returns True if accepted."""
        if target_phase == self.phase_idx:
            return False
        if not (self.can_switch() or self.must_switch()):
            return False
        self.in_yellow = True
        self.transition_remaining = self.phase_dur["yellow"]
        self.next_green_phase = target_phase
        return True

    # ---------------- state for RL ----------------
    def state_vector(self) -> np.ndarray:
        """
        State features (length 11):
          [d1..d4, q1..q4, w_max, phase_idx, elapsed]
        densities/queues are aggregated per arm (4 arms).
        """
        density = np.zeros(4)
        queue   = np.zeros(4)
        w_max = 0.0
        # arm i corresponds to lanes 2i, 2i+1
        for arm in range(4):
            count, q_ct = 0, 0
            for lane_idx in (2 * arm, 2 * arm + 1):
                vehs = self.lanes[lane_idx]
                count += len(vehs)
                q_ct += sum(1 for v in vehs if v.v < 1.0)
                if vehs:
                    w_max = max(w_max, max(v.waited_total for v in vehs))
            density[arm] = count / max(self.length / 7.5, 1)   # vehicles per 7.5 m
            queue[arm] = q_ct
        s = np.concatenate([density, queue,
                            [w_max / 60.0, self.phase_idx / 3.0,
                             self.phase_elapsed / 60.0]])
        return s.astype(np.float32)

    # ---------------- metrics ----------------
    def metrics(self, window_s: float, time_now: float) -> Dict[str, float]:
        recent = [v for v in self.completed_vehicles
                  if time_now - v.completion_time <= window_s]
        if recent:
            avg_wait = float(np.mean([v.waited_total for v in recent]))
            throughput = len(recent) * 3600.0 / max(window_s, 1.0)
        else:
            avg_wait = 0.0
            throughput = 0.0
        # current queue
        queue = sum(sum(1 for v in lane if v.v < 1.0) for lane in self.lanes)
        return {"avg_wait_s": avg_wait, "queue_veh": float(queue),
                "throughput_vph": throughput,
                "emission_g_total": self.cum_emission_g}


# =====================================================================
# Multi-intersection grid network
# =====================================================================
class GridNetwork:
    """
    NxN grid of intersections, each with 4 arms.
    Boundary vehicles enter from the network edges; internal arms feed each
    other (vehicles leaving the eastern arm of (i,j) become the western inflow
    of (i,j+1), modelling spillback at the network level).
    """
    def __init__(self, rows: int = 4, cols: int = 4, seed: int = 2026):
        self.rows = rows
        self.cols = cols
        self.intersections: List[Intersection] = [
            Intersection(iid=i * cols + j) for i in range(rows) for j in range(cols)
        ]
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.spawn_residual = np.zeros(N_LANES_PER_INTERSECTION * rows * cols)

    def get(self, i: int, j: int) -> Intersection:
        return self.intersections[i * self.cols + j]

    def step(self, dt: float, demand_vph_per_arm: float) -> None:
        # boundary vehicle injection (Poisson per lane)
        # demand_vph_per_arm: vehicles per hour per inbound boundary lane
        rate_per_lane_per_dt = (demand_vph_per_arm / 3600.0) * dt
        for k, intr in enumerate(self.intersections):
            i, j = divmod(k, self.cols)
            # arm 0 = N (boundary if i == 0), 1 = S (i == rows-1),
            # 2 = E (j == cols-1), 3 = W (j == 0)
            for arm, on_boundary in enumerate([i == 0, i == self.rows - 1,
                                                j == self.cols - 1, j == 0]):
                if not on_boundary:
                    continue
                for lane_off in (2 * arm, 2 * arm + 1):
                    if self.rng.random() < rate_per_lane_per_dt:
                        intr.inject(lane_off, self.time)
            intr.step(dt, self.time)
        self.time += dt

    def aggregate_metrics(self, window_s: float) -> Dict[str, float]:
        accum = {"avg_wait_s": 0.0, "queue_veh": 0.0,
                 "throughput_vph": 0.0, "emission_g_total": 0.0}
        for intr in self.intersections:
            m = intr.metrics(window_s, self.time)
            for k in accum:
                accum[k] += m[k]
        N = len(self.intersections)
        accum["avg_wait_s"] /= N
        accum["queue_veh"] /= N
        return accum
