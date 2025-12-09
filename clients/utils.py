import yaml
from typing import Dict, Any

from clients.constants import CONFIG_PATH


# --------------------
# HELPERS
# --------------------
def get_inputs(current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load shared guidelines and combine with current smart home state.

    Args:
        current_state: Dictionary containing device, weather, and power states.

    Returns:
        Dict containing agent input state and guideline rules.
    """
    with open(f'{CONFIG_PATH}/guidelines.yaml', 'r', encoding='utf-8') as file:
        guidelines = yaml.safe_load(file)

    return {
        'power_saving_thresholds': guidelines.get('power_saving_thresholds'),
        'ac_guidelines': guidelines.get('ac_guidelines'),
        'lights_guidelines': guidelines.get('lights_guidelines'),
        'fridge_guidelines': guidelines.get('fridge_guidelines'),
        'shutter_guidelines': guidelines.get('shutter_guidelines'),
        'current_ac_state': current_state.get('AC'),
        'current_lights_state': current_state.get('Lights'),
        'current_fridge_state': current_state.get('Fridge'),
        'current_shutter_state': current_state.get('Shutter'),
        'current_weather_state': current_state.get('Weather'),
        'current_power_state': current_state.get('PowerConsumption'),
        'current_daytime': current_state.get('DayTime'),
        'temp_preferences_distr': current_state.get('TempPreferencesDistributions'),
        'brightness_preferences_distr': current_state.get('BrightnessPreferencesDistributions'),
    }


def load_config(config_path: str = f"{CONFIG_PATH}/env_spec.yaml") -> dict:
    """
    Load YAML configuration file for the Smart House environment.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If the YAML file cannot be parsed.
        Exception: For any other unexpected errors.
    """
    data = {}
    try:
        with open(config_path) as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except Exception as e:
        print(f"Unexpected error loading spec from '{config_path}'")
        raise