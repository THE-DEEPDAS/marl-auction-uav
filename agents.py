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
    """Lightweight actor-critic with deterministic bid policy."""

    def __init__(
        self,
        drone_id: int,
        state_dim: int = 6,
        learning_rate: float = 0.01,
        critic_lr: float = 0.03,
        gamma: float = 0.99,
        max_bid: float = 100.0,
        use_energy_awareness: bool = True,
        device: str = "cpu",
    ):
        self.drone_id = drone_id
        self.state_dim = state_dim
        self.gamma = gamma
        self.max_bid = max_bid
        self.lr_actor = learning_rate
        self.lr_critic = critic_lr
        self.use_energy_awareness = use_energy_awareness
        self.device = device
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

    def _feature(self, obs: np.ndarray) -> np.ndarray:
        x = np.array(obs, dtype=np.float64)
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
        if exploration_noise > 0.0:
            base_bid += float(np.random.normal(0.0, exploration_noise))
        return float(np.clip(base_bid, 0.0, self.max_bid))

    def update(
        self,
        obs: np.ndarray,
        action: float,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
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
        else:
            self.w_critic += self.lr_critic * delta * x
            self.b_critic += self.lr_critic * delta

            z = float(np.dot(self.w_actor, x) + self.b_actor)
            s = float(_sigmoid(z))
            dbid_dz = self.max_bid * s * (1.0 - s)

            self.w_actor += self.lr_actor * delta * dbid_dz * x / self.max_bid
            self.b_actor += self.lr_actor * delta * dbid_dz / self.max_bid

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

    def update(self, obs, action, reward, next_obs, done) -> None:
        return

    def reset_episode(self) -> None:
        return


class GreedyAgent:
    def __init__(self, drone_id: int, max_bid: float = 100.0):
        self.drone_id = drone_id
        self.max_bid = max_bid

    def compute_bid(self, obs: np.ndarray, exploration_noise: float = 0.0) -> float:
        energy = float(obs[2])
        priority = float(obs[5])
        return float(np.clip(self.max_bid * (0.8 * priority + 0.2 * energy), 0.0, self.max_bid))

    def update(self, obs, action, reward, next_obs, done) -> None:
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

    def update(self, obs, action, reward, next_obs, done) -> None:
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
                    learning_rate=self.daca_config.learning_rate,
                    critic_lr=self.daca_config.critic_lr,
                    gamma=self.daca_config.gamma,
                    max_bid=self.daca_config.max_bid,
                    use_energy_awareness=self.daca_config.use_energy_awareness,
                    device=self.daca_config.device,
                )
            elif self.agent_type == "auction_nolearning":
                self.agents[i] = AuctionNoLearningAgent(i)
            elif self.agent_type == "greedy":
                self.agents[i] = GreedyAgent(i)
            elif self.agent_type == "qlearning":
                self.agents[i] = QLearningAgent(i)
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")

    def compute_bids(self, observations: Dict[int, np.ndarray], exploration_noise: float = 0.0) -> Dict[int, float]:
        bids: Dict[int, float] = {}
        for drone_id, obs in observations.items():
            agent = self.agents[drone_id]
            bids[drone_id] = float(agent.compute_bid(obs, exploration_noise=exploration_noise))
        return bids

    def update_agent(self, drone_id: int, obs: np.ndarray, action: float, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.agents[drone_id].update(obs, action, reward, next_obs, done)

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
