import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from pywizlight import wizlight, PilotBuilder, discovery
from clients.smart_device_base import SmartDevice


class SmartLights(SmartDevice):
    """
    Smart lighting device controller using pywizlight for WiZ bulbs.
    """

    def __init__(self, broadcast_ip: str = "192.168.1.255") -> None:
        """
        Initialize SmartLights instance.

        Args:
            broadcast_ip (str): Broadcast IP for bulb discovery.
        """
        super().__init__("Lights")
        self.broadcast_ip: str = broadcast_ip
        self.bulbs: List[Any] = []
        self._bulb_instances: Dict[str, wizlight] = {}

    async def _ensure_bulbs(self) -> None:
        """Discover bulbs if not already discovered."""
        if not self.bulbs:
            self.bulbs = await discovery.discover_lights(broadcast_space=self.broadcast_ip)

    async def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set state for all bulbs.

        Args:
            state (dict): Example {"on": True, "brightness": 50, "warm": True}.
                - "on" (bool): Turn bulbs on/off.
                - "brightness" (int): Brightness value (0-100).
                - "warm" (bool): True for warm white, False for cold white.
        """
        await self._ensure_bulbs()
        tasks = []

        for bulb in self.bulbs:
            if bulb.ip not in self._bulb_instances:
                self._bulb_instances[bulb.ip] = wizlight(bulb.ip)

            bulb_instance = self._bulb_instances[bulb.ip]

            if not state.get("on", False):
                tasks.append(bulb_instance.turn_off())
                continue

            # Build Pilot configuration
            brightness = state.get("brightness", 100)
            if state.get("warm", True):
                pilot = PilotBuilder(warm_white=brightness)
            else:
                pilot = PilotBuilder(cold_white=brightness)

            tasks.append(bulb_instance.turn_on(pilot))

        if tasks:
            await asyncio.gather(*tasks)

        super().set_state(state)

    async def get_state(self) -> Dict[str, Any]:
        """
        Fetch current live state from bulbs.

        Returns:
            dict: Current state of lights (on/off, brightness, warm/cold).
                  Falls back to last known state if fetching fails.
        """
        await self._ensure_bulbs()
        live_state: Dict[str, Any] = {}

        try:
            for bulb in self.bulbs:
                if bulb.ip not in self._bulb_instances:
                    self._bulb_instances[bulb.ip] = wizlight(bulb.ip)

                bulb_instance = self._bulb_instances[bulb.ip]
                pilot = await bulb_instance.updateState()
                if pilot is None:
                    continue

                live_state = {
                    "on": pilot.get_state(),
                    "brightness": pilot.get_brightness(),
                    "warm": not pilot.get_cold_white(),
                }
                break  # only read one bulb for simplicity

            if live_state:
                self.state.update(live_state)
                self.last_updated_at = datetime.now(timezone.utc)
                return self.state

        except Exception as e:
            print(f"⚠️ Failed to fetch light state: {e}")

        return self.state  # fallback

    async def close(self) -> None:
        """Close bulb connections properly."""
        for bulb_instance in self._bulb_instances.values():
            try:
                bulb_instance.transport.close()
                bulb_instance.transport = None
            except Exception:
                pass
        self._bulb_instances.clear()

    # --- Async context manager support ---
    async def __aenter__(self) -> "SmartLights":
        await self._ensure_bulbs()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


# --- Example usage ---
async def main():
    async with SmartLights() as lights:
        await lights.set_state({"on": True, "brightness": 50, "warm": True})
        print("Lights state:", await lights.get_state())
        await asyncio.sleep(2)
        await lights.set_state({"on": False})


if __name__ == "__main__":
    asyncio.run(main())
