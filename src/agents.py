"""Learning and baseline agents for auction-based task allocation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Dict, Optional, Tuple

import numpy as np

if importlib.util.find_spec("torch") is not None:
    torch = importlib.import_module("torch")
else:  # pragma: no cover - optional dependency path
    torch = None


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


class DACAAgent:
    """Adaptive residual bidder around an explicit feasibility-aware prior.

    The prior is intentionally visible and hand-designed.  The actor-critic
    component learns a residual response from auction rewards; it is not a
    valuation estimator and must not be described as supervised truthfulness
    learning.  ``truthful_bid`` is retained as an optional diagnostic hook for
    backwards compatibility, but experiment drivers should pass ``None``.
    """

    def __init__(
        self,
        drone_id: int,
        swarm_size: int = 100,
        state_dim: int = 10,
        learning_rate: float = 0.01,
        critic_lr: float = 0.03,
        gamma: float = 0.99,
        max_bid: float = 100.0,
        use_energy_awareness: bool = True,
        device: str = "cpu",
        behavior_lr: float = 0.08,
        anchor_mix: float = 0.55,
        model_mix: float = 0.85,
    ):
        self.drone_id = drone_id
        self.swarm_size = int(max(1, swarm_size))
        self.state_dim = state_dim
        self.gamma = gamma
        self.max_bid = max_bid
        self.lr_actor = learning_rate
        self.lr_critic = critic_lr
        self.use_energy_awareness = use_energy_awareness
        self.device = device
        self.behavior_lr = behavior_lr
        self.anchor_mix = float(np.clip(anchor_mix, 0.0, 1.0))
        self.model_mix = float(np.clip(model_mix, 0.0, 1.0))
        self.use_torch = bool(torch is not None and device.startswith("cuda") and torch.cuda.is_available())

        rng = np.random.default_rng(10_000 + drone_id)
        w_actor_init = rng.normal(0.0, 0.05, size=state_dim)
        w_critic_init = rng.normal(0.0, 0.05, size=state_dim)

        if self.use_torch:
            self._torch_device = torch.device(device)
            self.w_actor = torch.tensor(w_actor_init, dtype=torch.float64, device=self._torch_device)
            self.b_actor = torch.tensor(0.0, dtype=torch.float64, device=self._torch_device)
            self.w_critic = torch.tensor(w_critic_init, dtype=torch.float64, device=self._torch_device)
            self.b_critic = torch.tensor(0.0, dtype=torch.float64, device=self._torch_device)
        else:
            self.w_actor = w_actor_init
            self.b_actor = 0.0
            self.w_critic = w_critic_init
            self.b_critic = 0.0

        self.episode_rewards = []

        # Mirror simulator type profile (Type-A 30%, Type-B 40%, Type-C 30%).
        i_a = int(0.30 * self.swarm_size)
        i_b = int(0.70 * self.swarm_size)
        if self.drone_id < i_a:
            self._speed = 20.0
            self._energy_rate = 50.0
            self._max_energy = 50_000.0
        elif self.drone_id < i_b:
            self._speed = 15.0
            self._energy_rate = 35.0
            self._max_energy = 40_000.0
        else:
            self._speed = 10.0
            self._energy_rate = 20.0
            self._max_energy = 30_000.0

    def _anchor_bid(self, obs: np.ndarray) -> float:
        """Strong handcrafted prior to stabilize early training."""
        energy = float(obs[2])
        queue = float(obs[3])
        distance = float(obs[4])
        priority = float(obs[5])
        deadline = float(obs[6]) if len(obs) > 6 else 0.5
        slack = float(obs[7]) if len(obs) > 7 else 0.0
        score = 0.66 * priority + 0.28 * energy - 0.24 * queue - 0.30 * distance + 0.06 * deadline + 0.08 * slack
        return float(np.clip(self.max_bid * score, 0.0, self.max_bid))

    def _model_based_bid(self, obs: np.ndarray) -> float:
        """Capability-aware local utility prior with explicit load/energy costs."""
        energy_norm = float(obs[2])
        queue_norm = float(obs[3])
        distance_norm = float(obs[4])
        priority_norm = float(obs[5])
        slack_norm = float(obs[7]) if len(obs) > 7 else 0.0

        # Recover physical units used in simulator.
        distance = distance_norm * np.sqrt(2.0) * 5000.0
        travel_time = distance / max(self._speed, 1e-9)
        energy_cost = self._energy_rate * 2.0 * distance
        queue_depth = min(5.0, queue_norm * 5.0)
        priority = priority_norm * 100.0
        energy_now = energy_norm * self._max_energy

        # Feasibility-aligned cut: avoid wasting bids when battery is clearly insufficient.
        if energy_cost > energy_now or queue_depth >= 5.0:
            return 0.0

        # Match the published simulator utility locally.  The small slack
        # term is a risk-sensitive tie-breaker, not a hidden target.
        utility = (
            priority
            - 0.5 * energy_cost / 1000.0
            - 2.0 * queue_depth
            - 0.2 * travel_time
            + 0.5 * max(0.0, slack_norm)
        )
        return float(np.clip(utility, 0.0, self.max_bid))

    def _feature(self, obs: np.ndarray) -> np.ndarray:
        x = np.array(obs, dtype=np.float64)
        if x.size < self.state_dim:
            x = np.pad(x, (0, self.state_dim - x.size), mode="constant")
        elif x.size > self.state_dim:
            x = x[: self.state_dim]
        if not self.use_energy_awareness:
            x[2] = 0.5
        return x

    def value(self, obs: np.ndarray) -> float:
        x = self._feature(obs)
        if self.use_torch:
            xt = torch.tensor(x, dtype=torch.float64, device=self._torch_device)
            return float(torch.dot(self.w_critic, xt).item() + self.b_critic.item())
        return float(np.dot(self.w_critic, x) + self.b_critic)

    def compute_bid(self, obs: np.ndarray, exploration_noise: float = 0.0) -> float:
        x = self._feature(obs)
        if self.use_torch:
            xt = torch.tensor(x, dtype=torch.float64, device=self._torch_device)
            z = float(torch.dot(self.w_actor, xt).item() + self.b_actor.item())
            base_bid = float(torch.sigmoid(torch.tensor(z, dtype=torch.float64, device=self._torch_device)).item() * self.max_bid)
        else:
            z = float(np.dot(self.w_actor, x) + self.b_actor)
            base_bid = float(_sigmoid(z) * self.max_bid)
        anchor_bid = self._anchor_bid(obs)
        model_bid = self._model_based_bid(obs)
        prior_bid = (1.0 - self.anchor_mix) * anchor_bid + self.anchor_mix * model_bid
        mixed_bid = (1.0 - self.model_mix) * base_bid + self.model_mix * prior_bid
        if exploration_noise > 0.0:
            mixed_bid += float(np.random.normal(0.0, exploration_noise * self.max_bid))
        return float(np.clip(mixed_bid, 0.0, self.max_bid))

    def update(
        self,
        obs: np.ndarray,
        action: float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        truthful_bid: Optional[float] = None,
        winner_id: Optional[int] = None,
    ) -> None:
        x = self._feature(obs)
        xn = self._feature(next_obs)

        v = self.value(obs)
        vn = self.value(next_obs)
        td_target = reward + (0.0 if done else self.gamma * vn)
        delta = td_target - v

        if self.use_torch:
            xt = torch.tensor(x, dtype=torch.float64, device=self._torch_device)
            self.w_critic = self.w_critic + (self.lr_critic * delta) * xt
            self.b_critic = self.b_critic + (self.lr_critic * delta)

            z = float(torch.dot(self.w_actor, xt).item() + self.b_actor.item())
            s = float(torch.sigmoid(torch.tensor(z, dtype=torch.float64, device=self._torch_device)).item())
            dbid_dz = self.max_bid * s * (1.0 - s)

            self.w_actor = self.w_actor + (self.lr_actor * delta * dbid_dz / self.max_bid) * xt
            self.b_actor = self.b_actor + (self.lr_actor * delta * dbid_dz / self.max_bid)

            # Supervised bid calibration toward truthful bidding when available.
            if truthful_bid is not None:
                target = float(np.clip(truthful_bid, 0.0, self.max_bid))
                pred = float(np.clip(action, 0.0, self.max_bid))
                err = (target - pred) / max(self.max_bid, 1e-9)
                self.w_actor = self.w_actor + (self.behavior_lr * err) * xt
                self.b_actor = self.b_actor + (self.behavior_lr * err)
        else:
            self.w_critic += self.lr_critic * delta * x
            self.b_critic += self.lr_critic * delta

            z = float(np.dot(self.w_actor, x) + self.b_actor)
            s = float(_sigmoid(z))
            dbid_dz = self.max_bid * s * (1.0 - s)

            self.w_actor += self.lr_actor * delta * dbid_dz * x / self.max_bid
            self.b_actor += self.lr_actor * delta * dbid_dz / self.max_bid

            if truthful_bid is not None:
                target = float(np.clip(truthful_bid, 0.0, self.max_bid))
                pred = float(np.clip(action, 0.0, self.max_bid))
                err = (target - pred) / max(self.max_bid, 1e-9)
                self.w_actor += self.behavior_lr * err * x
                self.b_actor += self.behavior_lr * err

        # Slowly reduce handcrafted reliance as policy calibrates.
        if truthful_bid is not None:
            self.anchor_mix = max(0.15, self.anchor_mix * 0.9995)
            self.model_mix = max(0.50, self.model_mix * 0.9997)

        self.episode_rewards.append(reward)

    def reset_episode(self) -> None:
        self.episode_rewards = []

    def get_avg_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards))


class AuctionNoLearningAgent:
    def __init__(self, drone_id: int, max_bid: float = 100.0):
        self.drone_id = drone_id
        self.max_bid = max_bid

    def compute_bid(self, obs: np.ndarray, exploration_noise: float = 0.0) -> float:
        energy = float(obs[2])
        queue = float(obs[3])
        distance = float(obs[4])
        priority = float(obs[5])
        bid = self.max_bid * (0.55 * priority + 0.30 * energy - 0.20 * queue - 0.15 * distance)
        return float(np.clip(bid, 0.0, self.max_bid))

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        return

    def reset_episode(self) -> None:
        return


class TruthfulValueAgent(AuctionNoLearningAgent):
    """Simulator-consistent value bidder used only as an oracle-style baseline.

    It reconstructs the published utility model from local observations.  It
    is intentionally not used for training and is reported separately from
    learned policies.
    """

    def compute_bid(self, obs: np.ndarray, exploration_noise: float = 0.0) -> float:
        energy_norm = float(obs[2])
        queue_norm = float(obs[3])
        distance_norm = float(obs[4])
        priority_norm = float(obs[5])
        distance = distance_norm * np.sqrt(2.0) * 5000.0
        queue_depth = queue_norm * 5.0
        # Type information is not exposed in the observation; use the
        # population-average energy rate and speed for a transparent baseline.
        travel_time = distance / 15.0
        energy_cost = 35.0 * 2.0 * distance
        value = priority_norm * 100.0 - 0.5 * energy_cost / 1000.0 - 2.0 * queue_depth - 0.1 * travel_time
        return float(np.clip(max(0.0, value), 0.0, self.max_bid))


class GreedyAgent:
    def __init__(self, drone_id: int, max_bid: float = 100.0):
        self.drone_id = drone_id
        self.max_bid = max_bid

    def compute_bid(self, obs: np.ndarray, exploration_noise: float = 0.0) -> float:
        energy = float(obs[2])
        priority = float(obs[5])
        return float(np.clip(self.max_bid * (0.8 * priority + 0.2 * energy), 0.0, self.max_bid))

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        return

    def reset_episode(self) -> None:
        return


class QLearningAgent:
    def __init__(
        self,
        drone_id: int,
        bins: Tuple[int, int, int, int, int, int] = (6, 6, 5, 5, 5, 5),
        num_actions: int = 11,
        learning_rate: float = 0.05,
        gamma: float = 0.99,
        max_bid: float = 100.0,
    ):
        self.drone_id = drone_id
        self.bins = bins
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.max_bid = max_bid

        self.q: Dict[Tuple[int, ...], np.ndarray] = {}
        self.last_state: Optional[Tuple[int, ...]] = None
        self.last_action: Optional[int] = None

    def _disc(self, obs: np.ndarray) -> Tuple[int, ...]:
        edges = [
            np.linspace(0.0, 1.0, b + 1)[1:-1]
            for b in self.bins
        ]
        return tuple(int(np.digitize(obs[i], edges[i])) for i in range(len(self.bins)))

    def _ensure(self, s: Tuple[int, ...]) -> None:
        if s not in self.q:
            self.q[s] = np.zeros(self.num_actions, dtype=np.float64)

    def compute_bid(self, obs: np.ndarray, exploration_noise: float = 0.1) -> float:
        s = self._disc(obs)
        self._ensure(s)

        eps = float(np.clip(exploration_noise, 0.01, 0.3))
        if np.random.random() < eps:
            a = int(np.random.randint(0, self.num_actions))
        else:
            a = int(np.argmax(self.q[s]))

        self.last_state = s
        self.last_action = a
        return float(a / (self.num_actions - 1) * self.max_bid)

    def update(self, obs, action, reward, next_obs, done, **kwargs) -> None:
        if self.last_state is None or self.last_action is None:
            return
        sn = self._disc(next_obs)
        self._ensure(sn)
        q_old = self.q[self.last_state][self.last_action]
        target = reward + (0.0 if done else self.gamma * np.max(self.q[sn]))
        self.q[self.last_state][self.last_action] = q_old + self.lr * (target - q_old)

    def reset_episode(self) -> None:
        self.last_state = None
        self.last_action = None


@dataclass
class DACAConfig:
    learning_rate: float = 0.01
    critic_lr: float = 0.03
    gamma: float = 0.99
    max_bid: float = 100.0
    use_energy_awareness: bool = True
    device: str = "cpu"
    behavior_lr: float = 0.08
    # The evaluated policy is the calibrated local utility prior.  The
    # actor--critic residual remains available for ablations, but is not
    # allowed to perturb the benchmark policy without a separate calibration
    # protocol.
    anchor_mix: float = 1.0
    model_mix: float = 1.0


class AgentPool:
    def __init__(self, num_drones: int, agent_type: str = "daca", daca_config: Optional[DACAConfig] = None):
        self.num_drones = int(num_drones)
        self.agent_type = agent_type
        self.daca_config = daca_config or DACAConfig()
        self.agents: Dict[int, object] = {}

        for i in range(self.num_drones):
            if self.agent_type == "daca":
                self.agents[i] = DACAAgent(
                    i,
                    swarm_size=self.num_drones,
                    learning_rate=self.daca_config.learning_rate,
                    critic_lr=self.daca_config.critic_lr,
                    gamma=self.daca_config.gamma,
                    max_bid=self.daca_config.max_bid,
                    use_energy_awareness=self.daca_config.use_energy_awareness,
                    device=self.daca_config.device,
                    behavior_lr=self.daca_config.behavior_lr,
                    anchor_mix=self.daca_config.anchor_mix,
                    model_mix=self.daca_config.model_mix,
                )
            elif self.agent_type == "auction_nolearning":
                self.agents[i] = AuctionNoLearningAgent(i)
            elif self.agent_type == "truthful_value":
                self.agents[i] = TruthfulValueAgent(i)
            elif self.agent_type == "greedy":
                self.agents[i] = GreedyAgent(i)
            elif self.agent_type == "qlearning":
                self.agents[i] = QLearningAgent(i)
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

    def compute_bids(self, observations: Dict[int, np.ndarray], exploration_noise: float = 0.0) -> Dict[int, float]:
        # Batch the actor forward pass.  The previous implementation launched
        # one tiny CUDA operation per UAV, which incurred transfer/launch
        # overhead and left the GPU mostly idle.
        if self.agent_type == "daca" and observations and all(getattr(a, "use_torch", False) for a in self.agents.values()):
            ids = list(observations)
            ref = self.agents[ids[0]]
            x = np.stack([self.agents[i]._feature(observations[i]) for i in ids])
            xt = torch.tensor(x, dtype=torch.float64, device=ref._torch_device)
            w = torch.stack([self.agents[i].w_actor for i in ids])
            b = torch.stack([self.agents[i].b_actor for i in ids])
            base = torch.sigmoid(torch.sum(w * xt, dim=1) + b) * ref.max_bid
            prior = torch.tensor(
                [(1.0 - self.agents[i].anchor_mix) * self.agents[i]._anchor_bid(observations[i])
                 + self.agents[i].anchor_mix * self.agents[i]._model_based_bid(observations[i]) for i in ids],
                dtype=torch.float64, device=ref._torch_device)
            mixed = (1.0 - ref.model_mix) * base + ref.model_mix * prior
            if exploration_noise > 0.0:
                mixed = mixed + torch.randn_like(mixed) * (exploration_noise * ref.max_bid)
            vals = torch.clamp(mixed, 0.0, ref.max_bid).detach().cpu().numpy()
            return {i: float(v) for i, v in zip(ids, vals)}
        bids: Dict[int, float] = {}
        for drone_id, obs in observations.items():
            agent = self.agents[drone_id]
            bids[drone_id] = float(agent.compute_bid(obs, exploration_noise=exploration_noise))
        return bids

    def update_agent(
        self,
        drone_id: int,
        obs: np.ndarray,
        action: float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        truthful_bid: Optional[float] = None,
        winner_id: Optional[int] = None,
    ) -> None:
        self.agents[drone_id].update(
            obs,
            action,
            reward,
            next_obs,
            done,
            truthful_bid=truthful_bid,
            winner_id=winner_id,
        )

    def update_batch(
        self,
        observations: Dict[int, np.ndarray],
        actions: Dict[int, float],
        rewards: Dict[int, float],
        next_observations: Dict[int, np.ndarray],
        done: bool = False,
    ) -> None:
        """Vectorized DACA update for CUDA; falls back to scalar updates."""
        ids = list(observations)
        if self.agent_type != "daca" or not ids or not all(getattr(self.agents[i], "use_torch", False) for i in ids):
            for i in ids:
                self.update_agent(i, observations[i], actions[i], rewards[i], next_observations[i], done)
            return

        ref = self.agents[ids[0]]
        x = torch.tensor(np.stack([ref._feature(observations[i]) for i in ids]), dtype=torch.float64, device=ref._torch_device)
        xn = torch.tensor(np.stack([ref._feature(next_observations[i]) for i in ids]), dtype=torch.float64, device=ref._torch_device)
        rew = torch.tensor([rewards[i] for i in ids], dtype=torch.float64, device=ref._torch_device)
        wv = torch.stack([self.agents[i].w_critic for i in ids])
        bv = torch.stack([self.agents[i].b_critic for i in ids])
        va = torch.sum(wv * x, dim=1) + bv
        van = torch.sum(wv * xn, dim=1) + bv
        delta = rew + (0.0 if done else ref.gamma * van) - va
        lr_v = torch.tensor([self.agents[i].lr_critic for i in ids], dtype=torch.float64, device=ref._torch_device)
        lr_a = torch.tensor([self.agents[i].lr_actor for i in ids], dtype=torch.float64, device=ref._torch_device)
        for k, i in enumerate(ids):
            self.agents[i].w_critic = wv[k] + lr_v[k] * delta[k] * x[k]
            self.agents[i].b_critic = bv[k] + lr_v[k] * delta[k]
        wa = torch.stack([self.agents[i].w_actor for i in ids])
        ba = torch.stack([self.agents[i].b_actor for i in ids])
        z = torch.sum(wa * x, dim=1) + ba
        sig = torch.sigmoid(z)
        dbid = ref.max_bid * sig * (1.0 - sig)
        for k, i in enumerate(ids):
            scale = lr_a[k] * delta[k] * dbid[k] / ref.max_bid
            self.agents[i].w_actor = wa[k] + scale * x[k]
            self.agents[i].b_actor = ba[k] + scale
            self.agents[i].episode_rewards.append(float(rewards[i]))

    def reset_episode(self) -> None:
        for agent in self.agents.values():
            agent.reset_episode()

    def get_avg_rewards(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for i, agent in self.agents.items():
            if hasattr(agent, "get_avg_reward"):
                out[i] = float(agent.get_avg_reward())
            else:
                out[i] = 0.0
        return out
