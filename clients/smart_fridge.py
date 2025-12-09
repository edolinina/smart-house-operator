import os
import json
import requests
from datetime import datetime, timezone
from typing import Optional, Any, Dict

from clients.smart_device_base import SmartDevice

SMARTTHINGS_API_URL = "https://api.smartthings.com/v1"
SMARTTHINGS_TOKEN = os.getenv("SMARTTHINGS_TOKEN")


class SmartThingsClient:
    """
    Client for interacting with SmartThings cloud API.
    """

    def __init__(self, access_token: Optional[str] = None) -> None:
        """
        Initialize SmartThingsClient.

        Args:
            access_token (Optional[str]): Access token for SmartThings API.
                If None, will use environment variable SMARTTHINGS_TOKEN.

        Raises:
            ValueError: If no access token is provided or available in environment.
        """
        self.access_token: str = access_token or SMARTTHINGS_TOKEN
        if not self.access_token:
            raise ValueError("SmartThings access token is required")
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

    def get_devices(self) -> Optional[dict]:
        """
        Fetch all devices registered in SmartThings account.

        Returns:
            dict: JSON response with device list or None on failure.
        """
        try:
            response = requests.get(f"{SMARTTHINGS_API_URL}/devices", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching devices: {e}")
            return None

    def get_device_id(self, device_name: str) -> Optional[str]:
        """
        Retrieve device ID by matching device name.

        Args:
            device_name (str): Name of the device.

        Returns:
            Optional[str]: Device ID if found, otherwise None.
        """
        all_devices = self.get_devices()
        if not all_devices:
            return None
        for device in all_devices.get("items", []):
            if device_name.lower() in device["name"].lower():
                return device["deviceId"]
        return None

    def get_device_status(self, device_id: str) -> Optional[dict]:
        """
        Fetch current status of a specific device.

        Args:
            device_id (str): SmartThings device ID.

        Returns:
            dict: JSON response of device status or None on failure.
        """
        try:
            response = requests.get(f"{SMARTTHINGS_API_URL}/devices/{device_id}/status", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching device {device_id} status: {e}")
            return None

    def send_command(self, device_id: str, component: str, capability: str, command: str, arguments: Optional[list] = None) -> Optional[dict]:
        """
        Send a command to a SmartThings device.

        Args:
            device_id (str): Device ID.
            component (str): Device component name.
            capability (str): Capability name.
            command (str): Command name.
            arguments (Optional[list]): Command arguments.

        Returns:
            dict: JSON response from SmartThings or None on failure.
        """
        try:
            payload = {"commands": [{"component": component, "capability": capability, "command": command, "arguments": arguments or []}]}
            response = requests.post(
                f"{SMARTTHINGS_API_URL}/devices/{device_id}/commands",
                headers={**self.headers, "Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error sending command: {e}")
            return None


class SmartFridge(SmartDevice):
    """
    Smart fridge device controlled via SmartThings.
    """

    def __init__(self) -> None:
        """
        Initialize SmartFridge instance and discover device ID.
        """
        super().__init__("Fridge")
        self.client = SmartThingsClient()
        self.device_id: Optional[str] = self.client.get_device_id("refrigerator")
        self.state: Dict[str, Optional[float]] = {
            "cooler": None,
            "freezer": None
        }

    def get_state(self) -> Dict[str, Optional[float]]:
        """
        Retrieve current temperatures of cooler and freezer.

        Returns:
            dict: Dictionary with 'cooler' and 'freezer' temperatures.
        """
        status = self.client.get_device_status(self.device_id)
        if not status:
            return self.state

        for state_name in self.state:
            temp = status.get("components", {}).get(state_name, {}).get("temperatureMeasurement", {}).get("temperature", {}).get("value")
            self.state[state_name] = temp
        return self.state

    def set_state(self, state: Dict[str, float]) -> None:
        """
        Set target temperatures for cooler and freezer.

        Args:
            state (dict): Example: {"cooler": 4, "freezer": -18}
        """
        for state_name, temp in state.items():
            self.client.send_command(self.device_id, state_name, "thermostatCoolingSetpoint", "setCoolingSetpoint", [temp])
        super().set_state(state)


if __name__ == "__main__":
    fridge = SmartFridge()
    print("Current state:", fridge.get_state())
    fridge.set_state({"cooler": 3, "freezer": -19})
    print("Updated state:", fridge.get_state())
