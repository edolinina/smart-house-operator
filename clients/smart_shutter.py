import os
import asyncio
from dataclasses import asdict
from typing import Optional, Dict, Any

from aioswitcher.bridge import SwitcherBridge
from aioswitcher.device import DeviceType, SwitcherBase
from aioswitcher.api import Command, SwitcherApi

from clients.smart_device_base import SmartDevice

SWITCHER_TOKEN = os.getenv("SWITCHER_TOKEN")


class SmartShutter(SmartDevice):
    """
    SmartShutter provides asynchronous control of a Switcher smart shutter device.
    It can discover devices on the local network, fetch their state, and update their position.
    """

    def __init__(self, name: str, access_token: str = SWITCHER_TOKEN) -> None:
        """
        Initialize a SmartShutter instance.

        Args:
            name (str): The name of the Switcher device to control.
        """
        super().__init__("Shutter")
        self.name: str = name
        self.token: Optional[str] = access_token
        if not self.token:
            raise ValueError("Switcher access token is required")
        self.device: Optional[SwitcherBase] = None

    async def get_device(self, device_name: str, timeout: int = 10) -> Optional[SwitcherBase]:
        """
        Discover a Switcher device by name using the asynchronous bridge.

        Args:
            device_name (str): The exact name of the Switcher device to find.
            timeout (int): Maximum time in seconds to wait for discovery (default: 10).

        Returns:
            Optional[SwitcherBase]: The discovered Switcher device object, or None if not found.
        """
        found_event = asyncio.Event()
        found_device: Optional[SwitcherBase] = None

        def on_device_found_callback(device: SwitcherBase) -> None:
            """Callback triggered for each discovered device."""
            nonlocal found_device
            if device.name == device_name:
                found_device = device
                found_event.set()  # Signal that the desired device was found.

        async with SwitcherBridge(on_device_found_callback):
            try:
                # Wait until the device is found or timeout expires
                await asyncio.wait_for(found_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                print(f"⚠️ Device '{device_name}' not found within {timeout} seconds.")
                return None

        return found_device

    async def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state (position) of the smart shutter.

        Example:
            await set_state({"position": 30})

        Args:
            state (Dict[str, Any]): Dictionary containing desired shutter state.
                                   Expected key: "position" (int, e.g., 0–100).
        """
        if not self.device:
            print(f"❌ Unable to set state: device '{self.name}' not found.")
            return

        async with SwitcherApi(
            self.device.device_type,
            self.device.ip_address,
            self.device.device_id,
            self.device.device_key,
            self.token,
        ) as api:
            await api.set_position(state.get("position"))

        super().set_state(state)

    async def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the current state (position) of the smart shutter.

        Returns:
            Dict[str, Any]: Dictionary containing the current shutter position.
                            Example: {"position": 75}
        """
        if not self.device:
            print(f"⚠️ Unable to retrieve state: device '{self.name}' not found.")
            return self.state

        # Extract the current position from the device
        self.state["position"] = self.device.position[0]
        return self.state

    # --- Async context manager support ---
    async def __aenter__(self):
        if not self.device:
            self.device = await self.get_device(self.name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# --- Example usage ---
async def main() -> None:
    async with SmartShutter("Switcher Runner_286D") as switcher:
        await switcher.set_state({"position": 77})
        current_position = await switcher.get_state()
        print("Switcher position:", current_position)


if __name__ == "__main__":
    asyncio.run(main())
