"""Core UAV swarm simulator and VCG auction environment."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


@dataclass(frozen=True)
class DroneType:
    type_id: int
    cruise_speed: float
    energy_rate: float
    max_energy: float
    processing_time: float


@dataclass
class Task:
    task_id: int
    location: np.ndarray
    deadline: float
    priority: float
    arrival_time: float
    processing_time: float
    completed: bool = False
    assigned_drone: Optional[int] = None


class Drone:
    def __init__(self, drone_id: int, dtype: DroneType, init_pos: Tuple[float, float]):
        self.drone_id = drone_id
        self.dtype = dtype
        self.init_pos = np.array(init_pos, dtype=np.float64)
        self.position = self.init_pos.copy()
        self.energy = float(dtype.max_energy)
        self.pending_completion_times: List[float] = []

        self.total_tasks_assigned = 0
        self.total_tasks_completed = 0
        self.total_energy_consumed = 0.0

    def reset(self) -> None:
        self.position = self.init_pos.copy()
        self.energy = float(self.dtype.max_energy)
        self.pending_completion_times = []
        self.total_tasks_assigned = 0
        self.total_tasks_completed = 0
        self.total_energy_consumed = 0.0

    def advance_time(self, now: float) -> None:
        if not self.pending_completion_times:
            return
        remaining = []
        for t_done in self.pending_completion_times:
            if t_done <= now:
                self.total_tasks_completed += 1
            else:
                remaining.append(t_done)
        self.pending_completion_times = remaining

    @property
    def queue_depth(self) -> int:
        return len(self.pending_completion_times)

    def state_vector(self, area_size: float) -> np.ndarray:
        return np.array(
            [
                self.position[0] / area_size,
                self.position[1] / area_size,
                self.energy / max(self.dtype.max_energy, 1e-9),
                min(self.queue_depth / 5.0, 1.0),
            ],
            dtype=np.float64,
        )


class SwarmSimulator:
    """Task-stream simulator with VCG auctions and heterogeneous drones."""

    def __init__(
        self,
        num_drones: int = 50,
        deployment_area_size: float = 5000.0,
        task_arrival_rate: float = 1.0,
        duration: float = 1000.0,
        seed: int = 42,
    ):
        self.num_drones = int(num_drones)
        self.area_size = float(deployment_area_size)
        self.task_arrival_rate = float(task_arrival_rate)
        self.duration = float(duration)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.alpha = 0.5
        self.beta = 2.0
        self.gamma = 0.1

        self.drone_types = [
            DroneType(1, 20.0, 50.0, 50_000.0, 20.0),
            DroneType(2, 15.0, 35.0, 40_000.0, 20.0),
            DroneType(3, 10.0, 20.0, 30_000.0, 20.0),
        ]

        self.drones: List[Drone] = []
        self._build_drones()

        self.current_time = 0.0
        self.total_tasks_arrived = 0
        self.completed_tasks = 0
        self.missed_tasks = 0
        self.total_welfare = 0.0
        self.total_priority_arrived = 0.0

        self.tasks: Dict[int, Task] = {}
        self._task_events: List[Tuple[float, int]] = []
        self._next_task_id = 0
        self._generate_task_events()

    def _build_drones(self) -> None:
        distribution = [0.30, 0.40, 0.30]
        counts = [int(self.num_drones * p) for p in distribution]
        counts[-1] = self.num_drones - sum(counts[:-1])

        drone_id = 0
        for idx, count in enumerate(counts):
            dtype = self.drone_types[idx]
            for _ in range(count):
                x = self.rng.uniform(0.0, self.area_size)
                y = self.rng.uniform(0.0, self.area_size)
                self.drones.append(Drone(drone_id, dtype, (x, y)))
                drone_id += 1

    def _generate_task_events(self) -> None:
        self._task_events = []
        self._next_task_id = 0
        t = self.rng.exponential(1.0 / max(self.task_arrival_rate, 1e-9))
        while t < self.duration:
            tid = self._next_task_id
            self._next_task_id += 1
            heapq.heappush(self._task_events, (float(t), tid))
            t += self.rng.exponential(1.0 / max(self.task_arrival_rate, 1e-9))

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        for d in self.drones:
            d.reset()

        self.current_time = 0.0
        self.total_tasks_arrived = 0
        self.completed_tasks = 0
        self.missed_tasks = 0
        self.total_welfare = 0.0
        self.total_priority_arrived = 0.0
        self.tasks = {}
        self._generate_task_events()

    def has_next_task(self) -> bool:
        return len(self._task_events) > 0

    def next_task(self) -> Optional[Task]:
        if not self._task_events:
            return None

        arrival_time, task_id = heapq.heappop(self._task_events)
        self.current_time = float(arrival_time)

        for drone in self.drones:
            drone.advance_time(self.current_time)

        task = Task(
            task_id=task_id,
            location=np.array(
                [
                    self.rng.uniform(0.0, self.area_size),
                    self.rng.uniform(0.0, self.area_size),
                ],
                dtype=np.float64,
            ),
            deadline=float(self.rng.uniform(60.0, 300.0)),
            priority=float(self.rng.uniform(10.0, 100.0)),
            arrival_time=float(arrival_time),
            processing_time=20.0,
        )
        self.tasks[task.task_id] = task
        self.total_tasks_arrived += 1
        self.total_priority_arrived += task.priority
        return task

    def _distance(self, drone: Drone, task: Task) -> float:
        return _euclidean(drone.position, task.location)

    def compute_feasibility(self, drone: Drone, task: Task, now: Optional[float] = None) -> bool:
        now = self.current_time if now is None else now
        distance = self._distance(drone, task)
        travel_time = distance / max(drone.dtype.cruise_speed, 1e-9)

        earliest_start = now
        if drone.pending_completion_times:
            earliest_start = max(earliest_start, max(drone.pending_completion_times))
        completion_time = earliest_start + travel_time + task.processing_time
        deadline_time = task.arrival_time + task.deadline

        energy_cost = drone.dtype.energy_rate * 2.0 * distance

        return (
            completion_time <= deadline_time
            and energy_cost <= drone.energy
            and drone.queue_depth < 5
        )

    def compute_valuation(self, drone: Drone, task: Task, now: Optional[float] = None) -> float:
        if not self.compute_feasibility(drone, task, now):
            return -1e9
        distance = self._distance(drone, task)
        travel_time = distance / max(drone.dtype.cruise_speed, 1e-9)
        energy_cost = drone.dtype.energy_rate * 2.0 * distance
        return float(
            task.priority
            - self.alpha * energy_cost / 1000.0
            - self.beta * drone.queue_depth
            - self.gamma * travel_time
        )

    def get_observation(self, drone: Drone, task: Task) -> np.ndarray:
        distance = self._distance(drone, task)
        distance_norm = min(distance / (np.sqrt(2.0) * self.area_size), 1.0)
        priority_norm = task.priority / 100.0
        return np.array(
            [
                *drone.state_vector(self.area_size).tolist(),
                distance_norm,
                priority_norm,
            ],
            dtype=np.float64,
        )

    def run_auction(self, task: Task, bids: Dict[int, float]) -> Dict[str, object]:
        feasible_bids: List[Tuple[float, int]] = []
        truthful_vals: Dict[int, float] = {}

        for drone in self.drones:
            val = self.compute_valuation(drone, task)
            truthful_vals[drone.drone_id] = max(0.0, val)
            b = float(max(0.0, bids.get(drone.drone_id, 0.0)))
            if self.compute_feasibility(drone, task):
                feasible_bids.append((b, drone.drone_id))

        rewards = {d.drone_id: 0.0 for d in self.drones}
        winner_id: Optional[int] = None
        winner_bid = 0.0
        payment = 0.0
        lyapunov = 0.0

        for drone_id, val in truthful_vals.items():
            lyapunov += (max(0.0, bids.get(drone_id, 0.0)) - val) ** 2
        lyapunov /= max(len(truthful_vals), 1)

        if feasible_bids:
            feasible_bids.sort(key=lambda x: x[0], reverse=True)
            winner_bid, winner_id = feasible_bids[0]
            payment = feasible_bids[1][0] if len(feasible_bids) > 1 else 0.0

            winner = self.drones[winner_id]
            winner_val = self.compute_valuation(winner, task)
            utility = winner_val - payment
            rewards[winner_id] = float(utility)

            distance = self._distance(winner, task)
            travel_time = distance / max(winner.dtype.cruise_speed, 1e-9)
            energy_cost = winner.dtype.energy_rate * 2.0 * distance

            winner.energy = max(0.0, winner.energy - energy_cost)
            winner.total_energy_consumed += energy_cost
            winner.position = task.location.copy()

            finish_time = self.current_time + travel_time + task.processing_time
            winner.pending_completion_times.append(finish_time)
            winner.total_tasks_assigned += 1

            task.completed = True
            task.assigned_drone = winner_id
            self.completed_tasks += 1
            self.total_welfare += task.priority
        else:
            self.missed_tasks += 1

        return {
            "task_id": task.task_id,
            "time": self.current_time,
            "winner_id": winner_id,
            "winner_bid": winner_bid,
            "payment": payment,
            "rewards": rewards,
            "truthful_bids": truthful_vals,
            "lyapunov_distance": float(lyapunov),
            "completed_tasks": self.completed_tasks,
            "missed_tasks": self.missed_tasks,
        }

    def get_task_acceptance_rate(self) -> float:
        total = self.completed_tasks + self.missed_tasks
        return self.completed_tasks / total if total > 0 else 0.0

    def get_avg_energy_consumption(self) -> float:
        if self.completed_tasks == 0:
            return 0.0
        total_energy = sum(d.total_energy_consumed for d in self.drones)
        return total_energy / self.completed_tasks / 1000.0

    def get_fairness_index(self) -> float:
        loads = np.array([d.total_tasks_assigned for d in self.drones], dtype=np.float64)
        if loads.size == 0 or np.sum(loads) == 0.0:
            return 0.0
        mean = np.mean(loads)
        return float(np.sum(np.abs(loads - mean)) / (2.0 * loads.size * mean + 1e-12))

    def get_normalized_welfare(self) -> float:
        if self.total_priority_arrived <= 1e-12:
            return 0.0
        return self.total_welfare / self.total_priority_arrived

    def get_stats_summary(self) -> Dict[str, float]:
        return {
            "task_acceptance_rate": self.get_task_acceptance_rate(),
            "avg_energy_consumption": self.get_avg_energy_consumption(),
            "normalized_welfare": self.get_normalized_welfare(),
            "fairness_index": self.get_fairness_index(),
            "completed_tasks": float(self.completed_tasks),
            "missed_tasks": float(self.missed_tasks),
            "arrived_tasks": float(self.total_tasks_arrived),
            "total_time": float(self.current_time),
        }
