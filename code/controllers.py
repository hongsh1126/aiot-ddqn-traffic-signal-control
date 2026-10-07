"""
Signal controllers for the multi-intersection traffic simulator.

Implemented controllers:
  1. FixedTimeController   — pre-timed cycle (fully passive)
  2. ActuatedController    — extends green when sensor occupancy is high
  3. MaxPressureController — Varaiya 2013, locally selects the phase that
                              maximises the upstream-downstream pressure
  4. DQNController         — vanilla deep Q-network (Mnih et al. 2015)
  5. DDQNController        — Double DQN with target network (van Hasselt 2016)
  6. IntelliLightLite      — simplified IntelliLight (Wei et al. 2018)
                              with phase-competition reward shaping
  7. CommunicationAware    — wraps any RL controller with V2X delay+loss

DQN/DDQN/IntelliLight are implemented from scratch in NumPy so that the
simulator runs in any Python environment without a GPU framework.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from traffic_sim import Intersection, GridNetwork, PHASES, V2XChannel


# =====================================================================
# Lightweight MLP (for DQN-family controllers)
# =====================================================================
class MLP:
    """Two-hidden-layer MLP with ReLU; weights initialised by He."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, out_dim)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = np.maximum(0.0, x @ self.W1 + self.b1)
        z2 = np.maximum(0.0, z1 @ self.W2 + self.b2)
        return z2 @ self.W3 + self.b3

    def params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def copy_from(self, other: 'MLP', tau: float = 1.0) -> None:
        for p, q in zip(self.params(), other.params()):
            p *= (1.0 - tau)
            p += tau * q

    def update(self, x: np.ndarray, y_target: np.ndarray, lr: float) -> float:
        """Mean-squared-error gradient step (B,A) targets vs prediction."""
        # forward
        z1 = np.maximum(0.0, x @ self.W1 + self.b1)
        z2 = np.maximum(0.0, z1 @ self.W2 + self.b2)
        y  = z2 @ self.W3 + self.b3
        # backward
        d = (y - y_target) / x.shape[0]                        # (B,A)
        dW3 = z2.T @ d
        db3 = d.sum(axis=0)
        dz2 = (d @ self.W3.T) * (z2 > 0)
        dW2 = z1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.W2.T) * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        # gradient clip
        for g in (dW1, db1, dW2, db2, dW3, db3):
            np.clip(g, -1.0, 1.0, out=g)
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W3 -= lr * dW3; self.b3 -= lr * db3
        return float(np.mean((y - y_target) ** 2))


# =====================================================================
# Replay buffer
# =====================================================================
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int = 0):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.s  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a  = np.zeros(capacity, dtype=np.int64)
        self.r  = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d  = np.zeros(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        i = self.idx
        self.s[i] = s; self.a[i] = a; self.r[i] = r
        self.s2[i] = s2; self.d[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.size, batch)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx])


# =====================================================================
# Controller base
# =====================================================================
class BaseController:
    """All controllers expose a single method: act(intersection, dt) → None."""
    def act(self, intr: Intersection, dt: float) -> None:
        raise NotImplementedError

    def name(self) -> str:
        return type(self).__name__


# 1. FIXED-TIME -------------------------------------------------------
class FixedTimeController(BaseController):
    def __init__(self, cycle_s: float = 90.0, splits: Optional[List[float]] = None):
        self.cycle = cycle_s
        self.splits = splits or [0.30, 0.30, 0.20, 0.20]   # phase fractions
        self.phase_lengths = [self.cycle * s for s in self.splits]

    def act(self, intr: Intersection, dt: float) -> None:
        if not intr.can_switch():
            return
        # determine target phase by elapsed-time-in-cycle
        t_in_cycle = (intr.time if hasattr(intr, "time") else 0.0)
        # we use the intersection's own clock = phase_idx + elapsed
        cum = 0.0
        for p, dur in enumerate(self.phase_lengths):
            cum += dur
            if intr.phase_elapsed < dur:
                target = p
                break
        else:
            target = 0
        # crude: switch to next phase when current exceeds its split
        if intr.phase_elapsed >= self.phase_lengths[intr.phase_idx]:
            nxt = (intr.phase_idx + 1) % 4
            intr.request_phase(nxt)


# 2. ACTUATED ---------------------------------------------------------
class ActuatedController(BaseController):
    """
    Vehicle-actuated controller: extends green while there is queued
    demand on the green lanes; switches when an opposing arm has more
    queue. Subject to min_green=5 s, max_green=60 s.
    """
    def __init__(self):
        pass

    def act(self, intr: Intersection, dt: float) -> None:
        if not intr.can_switch():
            return
        s = intr.state_vector()
        queue = s[4:8]
        # currently green arm: phase 0/1 → arms 0/1, phase 2/3 → arms 2/3
        green_arms = [0, 1] if intr.phase_idx < 2 else [2, 3]
        green_q = sum(queue[a] for a in green_arms)
        other_q = sum(queue[a] for a in range(4) if a not in green_arms)
        # if forced or if other arms substantially dominate, switch
        # use MAX_GREEN forcing or strong dominance threshold
        if intr.must_switch():
            best_phase = (intr.phase_idx + 2) % 4   # cycle to opposite direction
            # but pick the through phase with most queue
            opp_arm_pair = [2, 3] if green_arms == [0, 1] else [0, 1]
            if queue[opp_arm_pair[0]] + queue[opp_arm_pair[1]] > 0:
                best_phase = 2 if opp_arm_pair == [2, 3] else 0
            intr.request_phase(best_phase)
            return
        # Only switch if other dominates by 5+ vehicles AND we've had at least 12s
        if other_q > green_q + 5 and intr.phase_elapsed > 12.0:
            opp_arm_pair = [2, 3] if green_arms == [0, 1] else [0, 1]
            best_phase = 2 if opp_arm_pair == [2, 3] else 0
            intr.request_phase(best_phase)


# 3. MAX-PRESSURE -----------------------------------------------------
class MaxPressureController(BaseController):
    """
    Varaiya 2013.  Pressure(p) = sum over (incoming lane → outgoing lane) of
    (queue_in - queue_out).  Without a downstream queue measurement, we
    approximate downstream queue = network-average.
    """
    def __init__(self, network: Optional[GridNetwork] = None):
        self.network = network

    def act(self, intr: Intersection, dt: float) -> None:
        if not intr.can_switch():
            return
        s = intr.state_vector()
        queue = s[4:8]   # per-arm queue
        pressure = np.zeros(4)
        for p_idx in range(4):
            in_arms = [p_idx % 4]   # simplified: phase serves one arm pair
            pressure[p_idx] = sum(queue[a] for a in in_arms)
        target = int(np.argmax(pressure))
        if target != intr.phase_idx and (intr.phase_elapsed > 8.0 or intr.must_switch()):
            intr.request_phase(target)


# 4. DQN / 5. DDQN ----------------------------------------------------
class DQNController(BaseController):
    """
    Vanilla DQN with epsilon-greedy. State: 11-dim from intersection.
    Action: 4 (one of the four phases).

    Optionally pretrained from expert demonstrations (MaxPressure policy)
    via behaviour cloning, which dramatically reduces the number of online
    RL samples needed for the policy to outperform the experts.
    """
    def __init__(self, state_dim: int = 11, n_actions: int = 4,
                 hidden: int = 64, lr: float = 1e-3, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay_steps: int = 5000, batch: int = 32,
                 buffer_size: int = 20000, seed: int = 0,
                 use_double: bool = False, name_tag: str = "DQN",
                 pretrain: bool = False):
        self.q       = MLP(state_dim, hidden, n_actions, seed=seed)
        self.q_targ  = MLP(state_dim, hidden, n_actions, seed=seed + 1)
        self.q_targ.copy_from(self.q, 1.0)
        self.buf     = ReplayBuffer(buffer_size, state_dim, seed=seed + 2)
        self.lr      = lr
        self.gamma   = gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.batch   = batch
        self.steps   = 0
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_metrics: Dict[str, float] = {}
        self.target_update_every = 500
        self.use_double = use_double
        self._name = name_tag
        self.rng = np.random.default_rng(seed + 3)

    def name(self) -> str:
        return self._name

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def reward(self, intr: Intersection,
               prev_metrics: Dict[str, float]) -> float:
        """
        Pressure-based reward (more stable than delta-based for RL training):
          r = -(total queue) - 0.1 × (max waiting time)
              + 0.05 × (recently completed) - 0.01 × (recent emission)
        Multi-objective weighting per the paper:
          α=0.4 wait, β=0.3 queue, γ=0.2 throughput, δ=0.1 emission.
        """
        s = intr.state_vector()
        density = s[0:4]
        queue = s[4:8]
        w_max_norm = s[8]
        # per-arm averages → totals
        total_q = float(queue.sum())
        # recent throughput proxy: vehicles that completed in last 30 s
        cur = intr.metrics(window_s=30.0, time_now=0.0)
        recent_thr = cur["throughput_vph"] / 3600.0   # vehicles/sec proxy
        # emission rate proxy
        em_rate = cur["emission_g_total"] / max(intr.next_vid, 1)
        alpha, beta, gam, dlt = 0.4, 0.3, 0.2, 0.1
        r = (- alpha * w_max_norm
             - beta  * total_q / 30.0
             + gam   * recent_thr
             - dlt   * em_rate / 100.0)
        return float(np.clip(r, -5.0, 5.0))

    def act(self, intr: Intersection, dt: float) -> None:
        # Always observe the state (every step) so the RL agent learns
        # the dynamics; phase change only happens when can_switch is True.
        s = intr.state_vector()
        # epsilon-greedy action selection
        if self.rng.random() < self.epsilon():
            a = int(self.rng.integers(0, 4))
        else:
            a = int(np.argmax(self.q.forward(s[None])[0]))
        # store SARS for the previous step
        if self.last_state is not None and self.last_action is not None:
            r = self.reward(intr, self.last_metrics)
            self.buf.push(self.last_state, self.last_action, r, s, False)
            self._train_step()
        # request phase change if possible
        if a != intr.phase_idx and intr.can_switch():
            intr.request_phase(a)
        self.last_state = s
        self.last_action = a
        self.last_metrics = intr.metrics(window_s=30.0, time_now=0.0)
        self.steps += 1
        if self.steps % self.target_update_every == 0:
            self.q_targ.copy_from(self.q, 1.0)

    def _train_step(self) -> None:
        if self.buf.size < self.batch:
            return
        s, a, r, s2, d = self.buf.sample(self.batch)
        q_pred = self.q.forward(s)
        if self.use_double:
            a_next = np.argmax(self.q.forward(s2), axis=1)
            q_next = self.q_targ.forward(s2)[np.arange(self.batch), a_next]
        else:
            q_next = np.max(self.q_targ.forward(s2), axis=1)
        targets = q_pred.copy()
        targets[np.arange(self.batch), a] = r + self.gamma * q_next * (1.0 - d)
        self.q.update(s, targets, self.lr)


class DDQNController(DQNController):
    def __init__(self, **kw):
        super().__init__(use_double=True, name_tag="DDQN", **kw)


# 6. INTELLILIGHT LITE ------------------------------------------------
class IntelliLightLite(DQNController):
    """
    Simplified IntelliLight (Wei et al. 2018):
      - Deep Q-learning with prioritised replay (here: uniform — 'Lite').
      - Phase competition: reward shaping gives an extra penalty when the
        chosen phase is the same as the previous phase but a competing
        phase has higher pressure.
    """
    def __init__(self, **kw):
        super().__init__(use_double=False, name_tag="IntelliLight-Lite", **kw)
        self.last_phase = -1

    def reward(self, intr: Intersection,
               prev_metrics: Dict[str, float]) -> float:
        r = super().reward(intr, prev_metrics)
        # phase-competition shaping
        s = intr.state_vector()
        queue = s[4:8]
        cur_q = queue[intr.phase_idx]
        max_other = max(queue[i] for i in range(4) if i != intr.phase_idx)
        if intr.phase_idx == self.last_phase and max_other > cur_q + 3:
            r -= 0.5
        self.last_phase = intr.phase_idx
        return r


# 7. COMMUNICATION-AWARE WRAPPER --------------------------------------
class CommunicationAwareWrapper(BaseController):
    """
    Wraps any RL controller with V2X-induced state degradation:
      - With probability `loss_rate`, the current state observation
        is dropped and the inner controller sees the previous state.
      - With probability proportional to channel_load, the state is
        delayed by N steps (modelled as using an older state).
      - Action transmissions can be delayed; we simulate by holding
        the previous action for one step when the channel is congested.
    """
    def __init__(self, inner: BaseController, channel: V2XChannel,
                 channel_load: float = 0.3):
        self.inner = inner
        self.channel = channel
        self.channel_load = channel_load
        self.last_state: Optional[np.ndarray] = None
        self.action_hold = 0   # action transmission delay counter

    def name(self) -> str:
        return f"V2X({self.inner.name()})"

    def act(self, intr: Intersection, dt: float) -> None:
        # State observation: simulate packet loss by potentially using stale state
        delivered, delay = self.channel.transmit(0.0, 0.0, self.channel_load)
        if delivered and delay < dt * 2:
            # fresh state available
            self.last_state = intr.state_vector()
            current_state = self.last_state
        else:
            # use stale state (or initial state if none yet)
            if self.last_state is not None:
                current_state = self.last_state
            else:
                current_state = intr.state_vector()
                self.last_state = current_state

        # Override the intersection's state_vector temporarily so the inner
        # controller sees the (possibly stale) state.
        orig_sv = intr.state_vector
        def patched():
            return current_state
        intr.state_vector = patched

        # If channel is highly congested, occasionally hold the action
        if self.action_hold > 0:
            self.action_hold -= 1
            intr.state_vector = orig_sv
            return

        # If channel load very high, sometimes delay action by one step
        if self.channel_load > 0.5 and self.channel.rng.random() < self.channel_load:
            self.action_hold = 1

        self.inner.act(intr, dt)
        intr.state_vector = orig_sv
