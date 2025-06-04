import os
import json
from collections import defaultdict

import pandas as pd
import traci

# Constants --------------------------------------------------------------
SUMO_BINARY = "sumo-gui"  # change to "sumo-gui" for visual runs
CONFIG_FILE = "simulation_files/simulation.sumocfg"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _setup_tls_controllers(tls_ids):
    """Build a per-traffic-light controller that simply replays the fixed-time
    program already embedded in the network (as created by netconvert)."""
    controllers = {}
    for tls_id in tls_ids:
        try:
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            durations = [max(1, p.duration or 1) for p in program.phases]
            controllers[tls_id] = {"durations": durations, "idx": 0, "timer": 0}
        except traci.exceptions.TraCIException:
            # ignore lights that cannot be queried (should not happen)
            continue
    return controllers


def _update_controllers(controllers):
    for tls_id, ctrl in controllers.items():
        ctrl["timer"] += 1
        if ctrl["timer"] >= ctrl["durations"][ctrl["idx"]]:
            ctrl["idx"] = (ctrl["idx"] + 1) % len(ctrl["durations"])
            ctrl["timer"] = 0
            traci.trafficlight.setPhase(tls_id, ctrl["idx"])


# ----------------------------------------------------------------------
# Baseline runner
# ----------------------------------------------------------------------

def run_fixed_time_baseline():
    try:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--no-warnings", "--no-step-log"])
    except Exception as e:
        print(f"Error starting SUMO: {e}")
        return {"error": str(e)}

    tls_ids = traci.trafficlight.getIDList()
    controllers = _setup_tls_controllers(tls_ids)

    vehicle_depart = {}
    travel_times = []
    total_wait_time = defaultdict(float)
    teleport_count = 0
    step = 0

    # Run until the network is empty
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # update every traffic light once per simulation step
        _update_controllers(controllers)

        # accumulate waiting times & departure times
        for vid in traci.vehicle.getIDList():
            total_wait_time[vid] += traci.vehicle.getWaitingTime(vid)
            vehicle_depart.setdefault(vid, step)

        # vehicles that finished this step
        for vid in traci.simulation.getArrivedIDList():
            travel_times.append(step - vehicle_depart.get(vid, step))

        teleport_count += len(traci.simulation.getStartingTeleportIDList())

    traci.close()

    # Aggregate metrics -------------------------------------------------
    avg_wait = sum(total_wait_time.values()) / len(total_wait_time) if total_wait_time else 0
    avg_travel = sum(travel_times) / len(travel_times) if travel_times else 0
    throughput = len(travel_times)

    summary = {
        "agent": "FixedTime-NativeProgram",
        "avg_waiting_time": avg_wait,
        "avg_travel_time": avg_travel,
        "throughput": throughput,
        "teleport_count": teleport_count,
        "total_reward": 0,
    }

    # Persist results ---------------------------------------------------
    with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=4)

    pd.DataFrame(
        {"vehicle_id": list(total_wait_time.keys()), "wait_time": list(total_wait_time.values())}
    ).to_csv(os.path.join(RESULTS_DIR, "baseline_waiting_times.csv"), index=False)

    return summary


if __name__ == "__main__":
    print(json.dumps(run_fixed_time_baseline(), indent=2))