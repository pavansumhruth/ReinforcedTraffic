# ---------------------------------------------------------------------------
# RL traffic‑signal framework – refactored for apples‑to‑apples comparison
# ---------------------------------------------------------------------------
# This file contains **both** the gymnasium environment (`SumoTrafficEnv`) and
# the train/eval script (`rl_run.py`) so you can drop a single file beside the
# SUMO assets and run it with:
#     python rl_refactor.py  --algo PPO  --timesteps 500_000
# ---------------------------------------------------------------------------
# Notes on the changes vs your original implementation
#   • Episode runs **until the network is empty** (minExpectedNumber == 0)
#   • Reward = –mean waiting time per vehicle – 0.1 × teleports this step
#     → normalises by traffic volume & discourages teleports
#   • Observation unchanged (quick upgrade: add downstream occupancies later)
#   • Action == phase index applied **immediately**; you can add yellow
#     transition logic if desired.
#   • Training + evaluation now write the same summary metrics as baseline_run
# ---------------------------------------------------------------------------

import argparse
import json
import os
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pandas as pd
import traci
import sumolib
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

###########################################################################
# 1.   Environment                                                          #
###########################################################################


class SumoTrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        sumo_config: str,
        net_file: str = "simulation_files/Filtered.net.xml",
        max_steps: int = 20_000,
        step_length: int = 1,
    ):
        super().__init__()
        self.sumo_config = sumo_config
        self.net_file = net_file
        self.max_steps = max_steps
        self.step_length = step_length  # simulation steps per env.step()

        self.tls_id = None
        self.step_count = 0
        self._prev_wait_total = 0.0

        # ---- network description --------------------------------------
        net = sumolib.net.readNet(net_file)
        self.lanes = [lane.getID() for edge in net.getEdges() for lane in edge.getLanes() if not lane.getID().startswith(":")]

        # action_space will be resized in reset() once we know num phases
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(len(self.lanes) * 2 + 1,), dtype=np.float32)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _state(self):
        veh_cnt = [traci.lane.getLastStepVehicleNumber(l) for l in self.lanes]
        halt_cnt = [traci.lane.getLastStepHaltingNumber(l) for l in self.lanes]
        phase = traci.trafficlight.getPhase(self.tls_id) if self.tls_id else 0
        return np.asarray(veh_cnt + halt_cnt + [phase], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()
        traci.start(["sumo-gui", "-c", self.sumo_config, "--no-warnings", "--no-step-log"])

        self.step_count = 0
        self._prev_wait_total = 0.0

        # select first TLS (single‑junction networks) – extendable later
        tls_ids = traci.trafficlight.getIDList()
        self.tls_id = tls_ids[0] if tls_ids else None
        if self.tls_id:
            n_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0].phases)
            self.action_space = gym.spaces.Discrete(n_phases)
            traci.trafficlight.setPhase(self.tls_id, 0)

        return self._state(), {}

    def step(self, action):
        # Apply chosen phase immediately
        if self.tls_id is not None:
            phase = int(action) % self.action_space.n
            traci.trafficlight.setPhase(self.tls_id, phase)

        # advance simulation for step_length seconds
        for _ in range(self.step_length):
            traci.simulationStep()
            self.step_count += 1

        # ----- compute reward -----------------------------------------
        veh_ids = traci.vehicle.getIDList()
        wait_total = sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)
        teleports = len(traci.simulation.getStartingTeleportIDList())

        mean_wait = wait_total / max(len(veh_ids), 1)
        reward = -mean_wait - 0.1 * teleports
        # bonus for improvement
        reward += (self._prev_wait_total - wait_total) * 0.001
        self._prev_wait_total = wait_total

        # ----- termination criteria -----------------------------------
        empty = traci.simulation.getMinExpectedNumber() == 0
        time_exceeded = self.step_count >= self.max_steps
        terminated = empty or time_exceeded
        truncated = False

        info = {
            "mean_wait": mean_wait,
            "teleports": teleports,
            "simulation_step": self.step_count,
        }
        return self._state(), reward, terminated, truncated, info

    def close(self):
        if traci.isLoaded():
            traci.close()

###########################################################################
# 2.   Training + evaluation                                               #
###########################################################################

def train_and_eval(algo: str, total_timesteps: int = 500_000):
    algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    Agent = algo_map[algo]

    # --- vectorised env for better sampling ---------------------------
    vec_env = make_vec_env(
        lambda: Monitor(
            SumoTrafficEnv("simulation_files/simulation.sumocfg"),
            filename=f"results/{algo}_monitor.csv",
        ),
        n_envs=1,
    )

    model = Agent("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"logs/{algo}")
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/{algo}_traffic")
    vec_env.close()

    # --- evaluation & metric parity with baseline --------------------
    eval_env = SumoTrafficEnv("simulation_files/simulation.sumocfg")
    obs, _ = eval_env.reset()

    depart_step = {}
    travel_times, wait_times = [], defaultdict(float)
    teleports = 0

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)

        # per‑vehicle stats
        for vid in traci.vehicle.getIDList():
            wait_times[vid] += traci.vehicle.getWaitingTime(vid)
            depart_step.setdefault(vid, eval_env.step_count)
        for vid in traci.simulation.getArrivedIDList():
            travel_times.append(eval_env.step_count - depart_step[vid])

        teleports += info["teleports"]
        done = terminated or truncated

    eval_env.close()

    # --- summarise ----------------------------------------------------
    summary = {
        "agent": algo,
        "avg_waiting_time": (sum(wait_times.values()) / len(wait_times)) if wait_times else 0,
        "avg_travel_time": (sum(travel_times) / len(travel_times)) if travel_times else 0,
        "throughput": len(travel_times),
        "teleport_count": teleports,
        "total_reward": 0,  # can compute cumulative reward if needed
    }

    with open(f"results/{algo}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=4)
    print(json.dumps(summary, indent=2))

###########################################################################
# 3.   CLI entry‑point                                                     #
###########################################################################

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["PPO", "A2C", "DQN", "ALL"], default="ALL",
                        help="RL algorithm to train/evaluate or ALL for a sweep")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training steps per algorithm")
    args = parser.parse_args()

    algos = [args.algo] if args.algo != "ALL" else ["PPO", "A2C", "DQN"]
    for algo in algos:
        print(f"=== Training & evaluating {algo} ===")
        train_and_eval(algo, args.timesteps)
