from clients.smart_device_base import SmartInput

# Thresholds for testing
POWER_THRESHOLDS = {
    "Normal": 1200,
    "Moderate": 2000,
    "Critical": 2800
}


class PowerConsumption(SmartInput):
    """
    Simulated power consumption sensor/agent.
    """

    def __init__(self) -> None:
        """
        Initialize the PowerConsumption sensor with empty state.
        """
        super().__init__("PowerConsumption")
        self.state: dict[str, int | str] = {}

    def set_level(self, level: str) -> dict[str, int | str]:
        """
        Set the simulated power consumption level.

        Args:
            level (str): One of 'Normal', 'Moderate', or 'Critical'.

        Returns:
            dict: The updated state including 'level' and 'watt'.
        
        Raises:
            ValueError: If the provided level is not valid.
        """
        if level not in POWER_THRESHOLDS:
            raise ValueError(f"Invalid level '{level}'. Must be one of {list(POWER_THRESHOLDS.keys())}")
        
        self.state = {
            "level": level,
            "watt": POWER_THRESHOLDS[level]
        }
        return self.state

    def get_state(self) -> dict[str, int | str]:
        """
        Get the current power state.

        Returns:
            dict: Current state including 'level' and 'watt'.
        """
        return self.state

    def get_watt(self) -> int | None:
        """
        Get the current power consumption in watts.

        Returns:
            int | None: Wattage if set, else None.
        """
        return self.state.get("watt")

    def get_level(self) -> str | None:
        """
        Get the current power consumption level.

        Returns:
            str | None: Level if set, else None.
        """
        return self.state.get("level")


if __name__ == "__main__":
    power_sensor = PowerConsumption()

    # Example usage
    print(power_sensor.set_level("Normal"))    # {'level': 'Normal', 'watt': 1200}
    print(power_sensor.set_level("Moderate"))  # {'level': 'Moderate', 'watt': 2000}
    print(power_sensor.set_level("Critical"))  # {'level': 'Critical', 'watt': 2800}

    print("Current state:", power_sensor.get_state())
