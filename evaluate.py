import os
import sys
import yaml
import random
import asyncio
import copy
import time
from filelock import FileLock
from collections import defaultdict

from clients.preferences_rl import PreferenceEnv, PreferenceRLAgent
from clients.weather_client import Weather
from clients.power_client import PowerConsumption, POWER_THRESHOLDS
from clients.constants import CONFIG_PATH
from clients.agno_team import run as agno_runner
from clients.smart_crew import run as crew_runner
from clients.lang_graph import run as langgraph_runner
from clients.smart_device_base import SmartInput
from clients.utils import load_config


class SmartHouseEvaluator:
    def __init__(self):
        self.results_file = "evaluation/evaluation_results.yaml"

    @staticmethod
    def make_model_record():
        return {
            "output": "",
            "response-time": 0.0,
            "personalization-score": None,
            "contextual-score": None,
            "interpretability-score": None
        }

    def register_response(self, state, outputs):
        input_state = copy.deepcopy(state)
        input_state["CurrentlyHome"] = ",".join(input_state["TempPreferencesDistributions"].keys())
        input_state.pop("BrightnessPreferencesDistributions")
        input_state.pop("TempPreferencesDistributions")

        result = {
            "input": input_state,
            "output": outputs
        }

        lock = FileLock(self.results_file + ".lock")

        with lock:
            with open(self.results_file, "a") as file:
                yaml.safe_dump(result, file, explicit_start=True, sort_keys=False)

    async def run(self, state):
        print(f"Collected State:\n{state}")

        outputs = defaultdict(self.make_model_record)

        # --- CrewAI (sync) ---
        start = time.perf_counter()
        crewai_response = crew_runner(state)
        crewai_time = time.perf_counter() - start
        outputs["crewai"]["output"] = crewai_response
        outputs["crewai"]["response-time"] = float(f"{crewai_time:.2f}")

        # --- Agno (async) ---
        start = time.perf_counter()
        agno_response = await agno_runner(state)
        agno_time = time.perf_counter() - start
        outputs["agno"]["output"] = agno_response
        outputs["agno"]["response-time"] = float(f"{agno_time:.2f}")

        # --- LangGraph (async) ---
        start = time.perf_counter()
        langgraph_response = await langgraph_runner(state)
        langgraph_time = time.perf_counter() - start
        outputs["langgraph"]["output"] = langgraph_response
        outputs["langgraph"]["response-time"] = float(f"{langgraph_time:.2f}")
        outputs = dict(outputs)
        self.register_response(state, outputs)


if __name__ == "__main__":
    config = load_config(f"{CONFIG_PATH}/evaluation_scenarios.yaml")

    evaluator = SmartHouseEvaluator()

    for power_level in POWER_THRESHOLDS:
        power = PowerConsumption()
        power.set_level(power_level)

        # --- RL preference models ---
        personas = ["man", "dogs", "woman", "kids"]
        climate_env = PreferenceEnv(personas, config["temperatures-preferences"], (20, 26))
        climate_rl_agent = PreferenceRLAgent(climate_env)
        climate_rl_agent.train(plot=True)

        light_env = PreferenceEnv(personas, config["brightness-preferences"], (1, 10))
        light_rl_agent = PreferenceRLAgent(light_env, scale_column=10)
        light_rl_agent.train(episodes=100, plot=True)

        for location, location_attrs in config["locations"].items():
            state = {}
            state["PowerConsumption"] = power.get_state()

            state["TempPreferencesDistributions"] = climate_rl_agent.get_state()
            state["BrightnessPreferencesDistributions"] = light_rl_agent.get_state()

            for attr in location_attrs:
                attr_state_key = attr.upper() if attr == "ac" else attr.replace("_", " ").title().replace(" ", "")
                state[attr_state_key] = location_attrs[attr]

            currently_home = state.pop("CurrentlyAtHome").split(",")
            for resident in list(state["TempPreferencesDistributions"].keys()):
                if resident not in currently_home:
                    del state["TempPreferencesDistributions"][resident]
                    del state["BrightnessPreferencesDistributions"][resident]

            try:
                asyncio.run(evaluator.run(state))
            except Exception:
                print(f"ERROR: exception for location: {location}")
