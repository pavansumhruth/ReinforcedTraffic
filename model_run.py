# -------------------------------------------------------------
# Run a trained RL traffic‑light model (PPO / A2C / DQN) inside
# the SUMO simulation and watch it in real‑time (optional).
# -------------------------------------------------------------
# Usage examples
#   python run_trained_model.py --model models/PPO_traffic.zip --gui
#   python run_trained_model.py --model models/A2C_traffic.zip --no‑gui
# -------------------------------------------------------------

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import traci
from stable_baselines3 import PPO, A2C, DQN

# we reuse the environment class defined in rl_refactor.py
from rl_refactor import SumoTrafficEnv

ALGO_MAP = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


def evaluate(model_path: str, use_gui: bool = False):
    algo_name = Path(model_path).stem.split("_")[0]  # PPO_traffic → PPO
    if algo_name not in ALGO_MAP:
        raise ValueError(f"Cannot infer algorithm from filename {model_path}")

    # ensure SUMO runs with or without GUI as requested
    sumo_binary = "sumo-gui" if use_gui else "sumo"

    # patch traci binary for this run
    os.environ["SUMO_BINARY"] = sumo_binary

    # environment (same settings used for training evaluation)
    env = SumoTrafficEnv("simulation_files/simulation.sumocfg")

    # load model
    ModelCls = ALGO_MAP[algo_name]
    model = ModelCls.load(model_path)

    obs, _ = env.reset()
    depart_step = {}
    travel_times, wait_times = [], defaultdict(float)
    teleports = 0

    print("\nRunning live simulation … press Ctrl‑C to abort\n")

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        # collect metrics similar to baseline
        for vid in traci.vehicle.getIDList():
            wait_times[vid] += traci.vehicle.getWaitingTime(vid)
            depart_step.setdefault(vid, env.step_count)
        for vid in traci.simulation.getArrivedIDList():
            travel_times.append(env.step_count - depart_step[vid])

        teleports += info["teleports"]
        done = terminated or truncated

    env.close()

    summary = {
        "agent": algo_name + "‑live",
        "avg_waiting_time": (sum(wait_times.values()) / len(wait_times)) if wait_times else 0,
        "avg_travel_time": (sum(travel_times) / len(travel_times)) if travel_times else 0,
        "throughput": len(travel_times),
        "teleport_count": teleports,
    }

    print("\n=== Run summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # save alongside results folder
    Path("results").mkdir(exist_ok=True)
    with open(f"results/{algo_name}_live_summary.json", "w") as fh:
        json.dump(summary, fh, indent=4)

    print("Summary saved to", fh.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to *.zip file saved by rl_refactor.py")
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument("--gui", action="store_true", help="Run SUMO with GUI (visualise)")
    gui_group.add_argument("--no-gui", action="store_true", help="Run headless (default)")
    args = parser.parse_args()

    evaluate(args.model, use_gui=args.gui)
