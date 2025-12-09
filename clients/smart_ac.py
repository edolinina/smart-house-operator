import os
import requests
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from clients.smart_device_base import SmartDevice

SENSIBO_SERVER = "https://home.sensibo.com/api/v2"
SENSIBO_API_KEY = os.getenv("SENSIBO_API_KEY")


class SmartAC(SmartDevice):
    """
    Smart AC controller using Sensibo API.
    """

    def __init__(self, name: str = "Main", api_key: str = SENSIBO_API_KEY) -> None:
        """
        Initialize SmartAC instance and discover device ID.

        Args:
            name (str): Room name or device name.
            api_key (str): Sensibo API key.
        """
        super().__init__("AC")
        self.api_key: str = api_key
        self.name: str = name
        self.base_url: str = SENSIBO_SERVER
        self.default_params: Dict[str, str] = {"apiKey": self.api_key}
        self.device_id: Optional[str] = self._get_device_id(name)

    # ---- HTTP helpers ----
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> dict:
        """
        Perform GET request to Sensibo API.

        Args:
            path (str): API endpoint path.
            params (Optional[dict]): Query parameters.

        Returns:
            dict: JSON response parsed as dictionary.

        Raises:
            requests.HTTPError: If HTTP request fails.
        """
        url = f"{self.base_url}{path}"
        merged_params = {**self.default_params, **(params or {})}
        response = requests.get(url, params=merged_params)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, data: dict) -> dict:
        """
        Perform POST request to Sensibo API.

        Args:
            path (str): API endpoint path.
            data (dict): Payload to send as JSON.

        Returns:
            dict: JSON response parsed as dictionary.

        Raises:
            requests.HTTPError: If HTTP request fails.
        """
        url = f"{self.base_url}{path}"
        response = requests.post(
            url, params=self.default_params, data=json.dumps(data),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    # ---- Device discovery ----
    def _list_devices(self) -> Dict[str, str]:
        """
        List all Sensibo devices for the user.

        Returns:
            dict: Mapping of room name to device ID.
        """
        params = {"fields": "id,room"}
        result = self._get("/users/me/pods", params=params)
        return {x["room"]["name"]: x["id"] for x in result["result"]}

    def _get_device_id(self, room_name: str) -> Optional[str]:
        """
        Get the device ID for a given room.

        Args:
            room_name (str): Name of the room.

        Returns:
            Optional[str]: Device ID if found, else None.
        """
        devices = self._list_devices()
        return devices.get(room_name)

    # ---- SmartDevice overrides ----
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set AC state (temperature, fan, on/off).

        Args:
            state (dict): Example {"temperature": 24, "fan": "medium", "on": True}.

        Raises:
            RuntimeError: If device is not found.
        """
        if not self.device_id:
            raise RuntimeError(f"{self.name} AC device not found")

        ac_state: Dict[str, Any] = {}
        if "temperature" in state:
            ac_state["targetTemperature"] = state["temperature"]
        if "fan" in state:
            ac_state["fanLevel"] = state["fan"]
        if "on" in state:
            ac_state["on"] = state["on"]

        if ac_state:
            payload = {"acState": ac_state}
            self._post(f"/pods/{self.device_id}/acStates", payload)
            self.state.update(state)
            self.last_updated_at = datetime.now(timezone.utc)

        super().set_state(state)

    def get_state(self) -> Dict[str, Any]:
        """
        Fetch the latest AC state from Sensibo API.

        Returns:
            dict: Current AC state including "temperature", "fan", "mode" and "on".

        Raises:
            RuntimeError: If device is not found.
            requests.HTTPError: If API request fails.
        """
        if not self.device_id:
            raise RuntimeError(f"{self.name} AC device not found")

        try:
            result = self._get(f"/pods/{self.device_id}/acStates")
            ac_state = result["result"][0]["acState"]

            state = {
                "temperature": ac_state.get("targetTemperature"),
                "fan": ac_state.get("fanLevel"),
                "on": ac_state.get("on"),
                "mode": ac_state.get("mode"),
            }
            self.state.update(state)
            self.last_updated_at = datetime.now(timezone.utc)
            return self.state
        except Exception as e:
            print(f"⚠️ Failed to fetch state from Sensibo: {e}")
            raise


if __name__ == "__main__":
    ac = SmartAC()
    ac.set_state({"temperature": 24, "fan": "low", "on": True})
    print(ac.get_state())
