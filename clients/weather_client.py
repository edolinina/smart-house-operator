import requests
from typing import Optional, Dict, Any
from clients.smart_device_base import SmartInput

WEATHER_OPEN_API = "https://api.open-meteo.com/v1/forecast"
DEFAULT_LATITUDE = 32.08
DEFAULT_LONGITUDE = 34.78

CURRENT_WEATHER_ATTRIBUTES = [
    "temperature_2m",
    "wind_direction_10m",
    "wind_speed_10m",
    "relative_humidity_2m",
    "cloud_cover",
    "rain",
]


class Weather(SmartInput):
    """
    Weather sensor input using Open-Meteo API.
    """

    def __init__(self, latitude: float = DEFAULT_LATITUDE, longitude: float = DEFAULT_LONGITUDE) -> None:
        """
        Initialize Weather sensor.

        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
        """
        super().__init__("Weather")
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.base_url: str = WEATHER_OPEN_API
        self.params: Dict[str, Any] = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "timezone": "auto",
            "daily": "sunset,sunrise",
            "current": ",".join(CURRENT_WEATHER_ATTRIBUTES)
        }
        self.state: Dict[str, Any] = {}

    def _fetch_weather(self, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Internal method to fetch weather data from Open-Meteo.

        Args:
            extra_params (dict, optional): Additional parameters to merge with default API params.

        Returns:
            dict: Parsed JSON response from Open-Meteo.
        """
        merged_params = {**self.params, **(extra_params or {})}
        response = requests.get(self.base_url, params=merged_params)
        response.raise_for_status()
        return response.json()

    def measure(self) -> Dict[str, Any]:
        """
        Fetch current weather and daily data, and update internal state.

        Returns:
            dict: Updated weather state including temperature, wind, humidity, cloud cover, sunrise, and sunset.
        """
        current_weather = self._fetch_weather().get("current", {})
        daily_data = self._fetch_weather().get("daily", {})
        self.set_state({
            "temperature": current_weather.get("temperature_2m"),
            "wind-speed": current_weather.get("wind_speed_10m"),
            "wind-direction": current_weather.get("wind_direction_10m"),
            "humidity": current_weather.get("relative_humidity_2m"),
            "cloud-cover": current_weather.get("cloud_cover"),
            "sunset": daily_data.get("sunset")[0] if daily_data.get("sunset") else None,
            "sunrise": daily_data.get("sunrise")[0] if daily_data.get("sunrise") else None,
        })
        return self.state

    # --- SmartInput interface ---
    def set_state(self, state: Dict[str, Any]) -> None:
        """Update the internal state dictionary."""
        self.state = state

    def get_state(self) -> Dict[str, Any]:
        """Return the current state of the weather sensor."""
        return self.state


if __name__ == "__main__":
    weather_sensor = Weather()
    weather_sensor.measure()  # updates internal state
    print("Weather state:", weather_sensor.get_state())
