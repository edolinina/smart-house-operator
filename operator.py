import re
import os
import ast
import sys
import json
import yaml
import asyncio
import argparse
import random

from typing import Dict, Any
from datetime import datetime

from clients.preferences_rl import PreferenceEnv, PreferenceRLAgent
from clients.smart_device_base import SmartDevice, SmartInput
from clients.weather_client import Weather
from clients.power_client import PowerConsumption
from clients.smart_ac import SmartAC
from clients.smart_lights import SmartLights
from clients.smart_fridge import SmartFridge
from clients.smart_shutter import SmartShutter
from clients.constants import CONFIG_PATH
from clients.agno_team import run as agno_runner
from clients.smart_crew import run as crew_runner
from clients.lang_graph import run as langgraph_runner
from clients.utils import load_config


class SmartHouseOperator:
    """
    Main operator class for managing smart home devices and adaptive agents.

    This class orchestrates:
    - State collection from smart devices and inputs
    - Decision-making via multi-agent systems (Agno, CrewAI, LangGraph)
    - Applying decisions to device states

    Attributes:
        inputs (Dict[str, SmartInput]): Registered environment or user inputs.
        devices (Dict[str, SmartDevice]): Registered smart home devices.
        engine (str): Name of the agent reasoning engine to use.
        dry_run (bool): Whether to simulate decisions without applying them.
    """

    def __init__(self, engine: str, dry_run: bool = False):
        self.inputs: Dict[str, SmartInput] = {}
        self.devices: Dict[str, SmartDevice] = {}
        self.engine: str = engine
        self.dry_run: bool = dry_run

    # --- Registration ---
    def register_input(self, name: str, input_obj: SmartInput) -> None:
        """
        Register a SmartInput by name.

        Args:
            name (str): Input identifier name.
            input_obj (SmartInput): Input object providing state information.
        """
        self.inputs[name] = input_obj

    def register_device(self, name: str, device_obj: SmartDevice) -> None:
        """
        Register a SmartDevice by name.

        Args:
            name (str): Device identifier name.
            device_obj (SmartDevice): Device object providing and applying state.
        """
        self.devices[name] = device_obj

    # --- State Aggregation ---
    async def collect_states(self) -> Dict[str, dict]:
        """
        Collect current states from all registered inputs and devices.

        Returns:
            Dict[str, dict]: Combined state dictionary for the environment.
        """
        state: Dict[str, dict] = {}

        # Collect input states (e.g., weather, power consumption, RL agents)
        for name, input_obj in self.inputs.items():
            state[name] = input_obj.get_state()

        # Collect device states (may include async operations)
        for name, device_obj in self.devices.items():
            if isinstance(device_obj, SmartLights):
                async with SmartLights() as lights:
                    state[name] = await lights.get_state()
            elif isinstance(device_obj, SmartShutter):
                async with SmartShutter(device_obj.name) as shutter:
                    state[name] = await shutter.get_state()
            else:
                state[name] = device_obj.get_state()

        return state

    # --- Execution ---
    async def run(self, config: Dict[str, Any]) -> None:
        """
        Main execution loop for the SmartHouseOperator.

        1. Collects current system state
        2. Passes it to the selected reasoning engine
        3. Applies the returned decisions to devices

        Args:
            config (Dict[str, Any]): Configuration parameters for the environment.

        Raises:
            RuntimeError: If the provided engine type is unsupported.
        """
        # Step 1: Collect current states
        current_state = await self.collect_states()
        current_state["DayTime"] = datetime.now().isoformat(timespec="seconds")
        print(f"Collected State:\n{current_state}")

        # Step 2: Send data to the selected agent team
        team_response = None
        if self.engine == "crewai":
            team_response = crew_runner(current_state)
        elif self.engine == "agno":
            team_response = await agno_runner(current_state)
        elif self.engine == "langgraph":
            team_response = await langgraph_runner(current_state)
        else:
            raise RuntimeError(f"Team {self.engine} is not supported")

        # Parse agent team response if in string form
        if isinstance(team_response, str):
            try:
                team_response = json.loads(team_response)
            except json.JSONDecodeError:
                print(f"Failed to parse agent response: {team_response}")
                sys.exit(1)

        # Extract explanation and print recommended states
        explanation = team_response.pop("Explanation", "No explanation provided.")
        print(f"Agent Team Recommended State:\n{team_response}")
        print(f"Agent Team Explanation: {explanation}")

        # Step 3: Optionally apply the recommendations
        if self.dry_run:
            print("Not applying recommended state because --dry-run is set.")
            sys.exit(0)

        for name, state in team_response.items():
            if name == "Alert" and 'message' in state:
                # Handle alert message from the agent
                print(f"⚠️  ALERT: {state['level'].upper()} - {state['message']}")
            elif name in self.devices:
                # Apply new state to the device
                if isinstance(self.devices[name], SmartLights):
                    async with SmartLights() as lights:
                        await lights.set_state(state)
                elif isinstance(self.devices[name], SmartShutter):
                    async with SmartShutter(self.devices[name].name) as shutter:
                        await shutter.set_state(state)
                else:
                    self.devices[name].set_state(state)
                print(f"New state {state} for device {name} was applied successfully.")
            else:
                print(f"[WARN] Unknown target: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart House Operator")
    parser.add_argument(
        "--engine",
        "-e",
        required=False,
        default="langgraph",
        choices=["agno", "crewai", "langgraph"],
        help="Engine to use in order to run the Agents team.",
    )
    parser.add_argument(
        "--dry-run",
        required=False,
        action="store_true",
        help="Run in dry-run mode for simulation/testing only.",
    )

    args = parser.parse_args()
    engine: str = args.engine
    dry_run: bool = args.dry_run

    # Load environment configuration
    config = load_config()

    # Randomly select a subset of personas currently at home
    personas = list(config["temperatures-preferences"].keys())
    current_at_home = random.sample(personas, k=random.randint(1, len(personas)))
    print(f"Currently at home: {current_at_home}")

    # Initialize operator
    operator = SmartHouseOperator(engine, dry_run)

    # --- Reinforcement learning preference modeling ---
    # Train temperature preference agent
    climate_env = PreferenceEnv(current_at_home, config["temperatures-preferences"], (20, 26))
    climate_rl_agent = PreferenceRLAgent(climate_env)
    climate_rl_agent.train()
    operator.register_input("TempPreferencesDistributions", climate_rl_agent)

    # Train brightness preference agent
    light_env = PreferenceEnv(current_at_home, config["brightness-preferences"], (1, 10))
    light_rl_agent = PreferenceRLAgent(light_env, scale_column=10)
    light_rl_agent.train(episodes=100)
    operator.register_input("BrightnessPreferencesDistributions", light_rl_agent)

    # --- Register environmental inputs ---
    latitude: float = config["location"]["latitude"]
    longitude: float = config["location"]["longitude"]
    weather = Weather(latitude, longitude)
    weather.measure()

    power = PowerConsumption()
    power.set_level("Moderate")

    # --- Register smart devices ---
    fridge = SmartFridge()
    lights = SmartLights(config["lights"]["space"])
    ac = SmartAC()
    shutter = SmartShutter(config["shutter"]["name"])

    operator.register_input("Weather", weather)
    operator.register_input("PowerConsumption", power)
    operator.register_device("Fridge", fridge)
    operator.register_device("Lights", lights)
    operator.register_device("AC", ac)
    operator.register_device("Shutter", shutter)

    # --- Execute decision cycle ---
    asyncio.run(operator.run(config))
