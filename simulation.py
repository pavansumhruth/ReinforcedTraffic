import os
import sumolib
from sumolib import net
import random

# === SETTINGS ===
NETWORK_FILE = "simulation_files/Filtered.net.xml"
ROUTE_FILE = "routes.rou.xml"
CONFIG_FILE = "config.sumocfg"
SIMULATION_STEPS = 5000
VEHICLE_RATE = 300  # spawn a vehicle every X seconds
OUTPUT_DIR = "simulation_files"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Step 1: Generate simple random routes ===
def generate_routes(network_file, route_file):
    net = sumolib.net.readNet(network_file)
    edges = list(net.getEdges())
    edge_ids = [edge.getID() for edge in edges if not edge.getID().startswith(":")]  # skip internal edges

    with open(os.path.join(OUTPUT_DIR, route_file), "w") as routes:
        print("""<routes>""", file=routes)

        for i in range(SIMULATION_STEPS // VEHICLE_RATE):
            from_edge = random.choice(edge_ids)
            to_edge = random.choice(edge_ids)
            while to_edge == from_edge:
                to_edge = random.choice(edge_ids)

            print(f"""    <vehicle id="veh{i}" type="car" depart="{i * VEHICLE_RATE}">
        <route edges="{from_edge} {to_edge}"/>
    </vehicle>""", file=routes)

        print("</routes>", file=routes)

# === Step 2: Generate config file ===
def generate_config(net_file, route_file, config_file):
    with open(os.path.join(OUTPUT_DIR, config_file), "w") as cfg:
        print(f"""<configuration>
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{SIMULATION_STEPS}"/>
    </time>
</configuration>""", file=cfg)

# === Run the automation ===
generate_routes(NETWORK_FILE, ROUTE_FILE)
generate_config(NETWORK_FILE, ROUTE_FILE, CONFIG_FILE)

print("✅ Route and config files generated!")
