from datetime import datetime
from typing import Optional, Any, Dict


class SmartDevice:
    """
    Base class for smart devices.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a SmartDevice instance.

        Args:
            name (str): Name of the device.
        """
        self.name: str = name
        self.last_updated_at: Optional[datetime] = None
        self.last_updated_by: Optional[str] = None
        self.state: Dict[str, Any] = {}

    def set_state(self, state: Dict[str, Any], user: str = "auto") -> None:
        """
        Update the device state and record the timestamp.

        Args:
            state (Dict[str, Any]): New state for the device.
            user (str): Identifier of the user/agent updating the state. Default is "auto".

        Raises:
            ValueError: If state is not a dictionary.
        """
        if not isinstance(state, dict):
            raise ValueError("State must be a dictionary")

        self.state = state
        self.last_updated_at = datetime.utcnow()
        self.last_updated_by = user

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current device state.

        Returns:
            Dict[str, Any]: Current state dictionary.
        """
        return self.state

    def __repr__(self) -> str:
        return (
            f"<SmartDevice name={self.name}, "
            f"state={self.state}, "
            f"last_updated_at={self.last_updated_at}, "
            f"last_updated_by={self.last_updated_by}>"
        )


class SmartInput:
    """
    Base class for smart sensors/inputs.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a SmartInput instance.

        Args:
            name (str): Name of the input.
        """
        self.name: str = name
        self.last_measured_at: Optional[datetime] = None
        self.values: Dict[str, Any] = {}

    def set_values(self, values: Dict[str, Any]) -> None:
        """
        Update input values and record the timestamp.

        Args:
            values (Dict[str, Any]): Dictionary of measured values.

        Raises:
            ValueError: If values is not a dictionary.
        """
        if not isinstance(values, dict):
            raise ValueError("Values must be a dictionary")

        self.values = values
        self.last_measured_at = datetime.utcnow()

    def get_values(self) -> Dict[str, Any]:
        """
        Get the current input values.

        Returns:
            Dict[str, Any]: Current measured values.
        """
        return self.values

    def __repr__(self) -> str:
        return (
            f"<SmartInput name={self.name}, "
            f"values={self.values}, "
            f"last_measured_at={self.last_measured_at}>"
        )
