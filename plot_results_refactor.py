import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")

###############################################################################
# 1. Collect *_summary.json produced by baseline_run.py and rl_refactor.py
###############################################################################

summaries = []
for file in RESULTS_DIR.glob("*_summary.json"):
    with open(file) as fh:
        data = json.load(fh)
        summaries.append(data)

if not summaries:
    raise SystemExit("No summary files found in ./results – run baseline and RL first!")

df = pd.DataFrame(summaries)
# order: baseline first, then RL algorithms alphabetically
baseline_mask = df["agent"].str.contains("FixedTime", case=False)
rl_mask = ~baseline_mask

ordered = pd.concat([
    df.loc[baseline_mask],
    df.loc[rl_mask].sort_values("agent")
])

###############################################################################
# 2. Plot helpers
###############################################################################

def _bar(metric: str, ylabel: str, title: str, fname: str):
    plt.figure()
    plt.bar(ordered["agent"], ordered[metric])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname)
    plt.close()

_bar("avg_waiting_time", "Average Waiting Time (s)", "Average Waiting Time", "compare_avg_waiting.png")
_bar("teleport_count", "Teleports", "Teleportation Events", "compare_teleports.png")
_bar("throughput", "Vehicles Processed", "Throughput", "compare_throughput.png")
_bar("avg_travel_time", "Average Travel Time (s)", "Average Travel Time", "compare_avg_travel.png")

###############################################################################
###############################################################################
# 3. Learning curves – rely on monitor CSV
###############################################################################
plt.figure(figsize=(8, 5))
for algo in ["PPO", "A2C", "DQN"]:
    csv_path = RESULTS_DIR / f"{algo}_monitor.csv"
    if not csv_path.exists():
        continue
    # Stable‑baselines3 monitor: first row is a JSON comment, second row header
    try:
        data = pd.read_csv(csv_path, skiprows=1)
    except Exception as e:
        print(f"Could not read {csv_path}: {e}")
        continue

    if "r" not in data.columns or "l" not in data.columns:
        print(f"Monitor file for {algo} lacks 'r' or 'l' columns – skipped.")
        continue

    rolling = data["r"].rolling(10).mean()
    plt.plot(data["l"], rolling, label=algo)

plt.xlabel("Episode length (steps)")
plt.ylabel("Reward (smoothed, window=10)")
plt.title("Learning curves – RL algorithms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "compare_learning_curves.png")
plt.close()

print("Plots saved in", RESULTS_DIR.resolve())
