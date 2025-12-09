import yaml
import json
import matplotlib.pyplot as plt
from collections import defaultdict


def estimate_power(temp, mode, fan):
    """
    Estimate AC power consumption using a simple proportional load model:
    
    Total Power = Base Load + (ΔT × Mode Multiplier × Temperature Factor) + Fan Power
    """

    # -----------------------------
    # PARAMETERS (simplified model)
    # -----------------------------
    T_room = 24  # assumed room baseline for ΔT calculation
    base_load = 300        # constant AC base power (W)
    temperature_factor = 70  # watts per degree ΔT

    mode_multipliers = {
        "cool": 1.2,
        "heat": 1.6
    }

    fan_power = {
        "low": 30,
        "medium": 60,
        "high": 90
    }

    # -----------------------------
    # CALCULATE ΔT (depends on mode)
    # -----------------------------
    if mode == "cool":
        delta_t = max(0, T_room - temp)
    elif mode == "heat":
        delta_t = max(0, temp - T_room)
    else:
        raise ValueError("Mode must be 'cool' or 'heat'")

    # -----------------------------
    # LOAD CALCULATION
    # -----------------------------
    mode_mult = mode_multipliers[mode]
    fan_load = fan_power[fan]

    thermal_load = delta_t * mode_mult * temperature_factor

    power = base_load + thermal_load + fan_load
    return round(power, 2)


def extract_ac_records(path):
    ac_pairs = []

    with open(path, "r") as f:
        data = list(yaml.safe_load_all(f))

    for record in data:
        try:
            # Extract power consumption level
            level = record["input"]["PowerConsumption"]["level"]

            # Extract input AC
            input_ac = record["input"]["AC"]

            # Extract langgraph output for AC
            lg_output_text = record["output"]["langgraph"]["output"]
            lg_output = json.loads(lg_output_text)
            output_ac = lg_output.get("AC", {})

            ac_pairs.append({
                "input_ac": input_ac,
                "output_ac": output_ac,
                "level": level
            })

        except Exception as e:
            print("Skipping record due to error:", e)
            continue

    return ac_pairs


if __name__ == "__main__":
    result = extract_ac_records("evaluation_results.yaml")
    power_improvement = defaultdict(list)

    for entry in result:
        level = entry["level"]
        input_ac = entry["input_ac"]
        output_ac = entry["output_ac"]

        input_power_usage = 0
        if input_ac["on"]:
            input_power_usage = estimate_power(input_ac["temperature"], input_ac["mode"], input_ac["fan"])

        output_power_usage = 0
        if input_ac["on"]:
            output_power_usage = estimate_power(output_ac["temperature"], output_ac["mode"], output_ac["fan"])

        print("Input AC:", entry["input_ac"], "Power", input_power_usage)
        print("Output AC:", entry["output_ac"], "Power", output_power_usage)

        delta = 0
        if input_power_usage != 0:
            delta = 100 * ((input_power_usage - output_power_usage) / input_power_usage)
            power_improvement[level].append(delta)

        print("Power Improvement, %:", f"{delta:.2f}")
        print("---")

    for power_level in dict(power_improvement).keys():
        avg_improvement = sum(power_improvement[power_level])/len(power_improvement[power_level])
        print(f"Avg Improvement for {power_level} Level, %:", f"{avg_improvement:.2f}")

    # -------------------------
    # PLOT RESULTS
    # -------------------------
    plt.figure(figsize=(10, 6))

    colors = {"Normal": "green", "Moderate": "blue", "Critical": "red"}
    for level, deltas in power_improvement.items():
        avg = sum(deltas)/len(deltas)
        plt.plot(range(1, len(deltas)+1), deltas, marker='o', linestyle='-', color=colors.get(level, "black"),
                 label=f"{level} Level Improvement (Avg: {avg:.2f}%)")

    plt.title("AC Power Improvement per Scenario by Power Consumption Level")
    plt.xlabel("Scenario Index")
    plt.ylabel("Power Improvement (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
