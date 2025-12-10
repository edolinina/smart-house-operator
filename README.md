# Smart House Operator for Power Saving

The **Smart House Operator** is a multi-agent system designed to **optimize energy consumption** in a household while maintaining comfort and safety.
It integrates intelligent agents for **Air Conditioning (AC)**, **Lighting**, **Fridge**, **Shutter** and a **Manager Agent** that coordinates decisions across the system.

---

### Agents Inputs & Environmental Context

The Smart Home system integrates multiple **contextual and learned inputs** to support adaptive, closed-loop decision-making across all agents.

#### **1. Persona-Based Preference Learning**
Each simulation randomly selects a subset of personas currently at home.  
Two reinforcement learning environments are used to model personal comfort preferences:

- **Temperature Preferences (`TempPreferencesDistributions`)**  
  Learned by a `PreferenceRLAgent` trained on individual comfort ranges (e.g., 20–26 °C).  
- **Brightness Preferences (`BrightnessPreferencesDistributions`)**  
  Learned by a second `PreferenceRLAgent`, trained on personal lighting preferences (scale 1–10).

The `PreferenceRLAgent` is a tabular, model-free agent that learns preferences for different personas through direct interaction with the environment. It applies value-based reinforcement learning, implementing a simplified form of *Q-learning* where the agent directly learns the action-value function *Q(s,a)* from experience — specifically from observed states, actions and rewards. A *Q-table* is maintained to estimate the expected return for each state–action pair, while the *ε-greedy* policy balances exploration and exploitation. This implementation considers only immediate rewards, without using a discount factor or estimating future values.

These learned preferences distributions allow agents to personalize temperature and lighting behavior dynamically.

#### **2. Environmental & System Inputs**
Additional real-time or simulated inputs provide environmental awareness and operational constraints:

- **`Weather`** – Current temperature, humidity, wind and daylight (from Open-Meteo API).  
- **`PowerConsumption`** – Current household power load categorized as *Normal*, *Moderate* or *Critical*.  

#### **3. Smart Devices (Controllable Agents)**
Registered actuators that provide current state of the contolled devices:

- **`Lights`** – WiZ smart bulbs (brightness, color warmth, on/off).  
- **`AC`** – Sensibo-controlled air conditioning (temperature, fan speed, mode, power state).  
- **`Shutter`** – Smart blinds position (percentage open/closed).  
- **`Fridge`** – SmartThings-controlled refrigerator (temperature for fridge and freezer).  

Each device and input module contributes to the **adaptive decision loop**, ensuring agents reason about real conditions, learned preferences and system constraints before taking action.

---

## 🏠 Team's Agents Descriptions and Roles

### 1. **AC Optimization Agent**
- **Goal**: Maintain comfort while minimizing energy use.  
- **Inputs**: Current AC state, weather, power thresholds, personal temperature preferences, time of day.
- **Output**:  
```json
	  {
	    "AC": {
	      "temperature": 24,
	      "fan": "low",
	      "on": true,
	      "mode": "cool"
	    }
	  }
```

### 2. **Lighting Optimization Agent**
- **Goal**: Manage lighting for circadian rhythm and efficiency.
- **Inputs**: Current lights state, time, weather, power consumption, personal brightness preferences.
- **Output**:  
  ```json
	  {
	  "Lights": {
	    "on": true,
	    "brightness": 50,
	    "warm": true
	  }
	}
```

### 3. **Fridge Optimization Agent**
- **Goal**: Ensure food safety while reducing energy consumption.
- **Inputs**: Current fridge/freezer state, weather, power consumption.
- **Output**:  
```json
	  {
	  "Fridge": {
	    "cooler": 4,
	    "freezer": -18
	  }
	}
```

### 4. **Shutter Optimization Agent**
- **Goal**: Regulate natural light, privacy, and energy efficiency by adjusting shutter positions.
- **Inputs**: Current shutter state, weather data (sunrise/sunset time, cloud cover, wind speed), power consumption, and time-of-day context.
- **Output** (percentage open (0–100)):
```json
  {
    "Shutter": {
      "position": 77
    }
  }
```

### 5. **Manager Agent**
- **Goal**: Coordinate other agents decisions into a unified smart-home strategy.
- **Inputs**: Outputs from all individual agents.
- **Additional Behavior**: May raise alerts when thresholds are violated.
- **Output**:  
```python
	  {
	  "AC": {...},
	  "Lights": {...},
	  "Fridge": {...},
	  "Shutter": {...},
	  "Alert": {
	    "level": "warning",
	    "message": "Power consumption is critical, adjustments applied."
	  }
	}
```

### 🧩 Multi-Agent Orchestration Engines
The Smart House Operator supports **three interchangeable engines** for orchestrating multi-agent collaboration:

- **CrewAI** – for structured, role-based teamwork with agent specialization.  
- **LangGraph** – for dynamic graph-based agent workflows and contextual memory.  
- **Agno Team** – for lightweight, modular coordination and asynchronous task execution.  

Each engine can be configured to run the same Smart agents (lights, AC, shutter, fridge), enabling flexible experimentation with different coordination strategies.

## ⚙️ Orchestration Flow
The SmartHouseOperator coordinates the system by running the following flow:
1. Collect States -> gathers inputs (weather, power, personal preferences, current device states).
2. Pass States to Agents -> invokes the agents (via CrewAI, LangGraph or Agno team).
3. Integrate Decisions -> Manager Agent combines outputs.
4. Apply Actions -> updates device states (AC, Lights, Fridge, Shutter).
5. Alert User (if necessary).

## 🔧 Environment Requirements
Before running the SmartHouseOperator, you must set the following environment variables to authenticate with connected devices:

### Required Environment Variables

| Variable Name       | Description                                      |
|---------------------|--------------------------------------------------|
| `SENSIBO_API_KEY`   | API key for controlling Sensibo AC devices.      |
| `SMARTTHINGS_TOKEN` | Token for accessing Samsung SmartThings devices. |
| `SWITCHER_TOKEN`    | Token for Switcher smart plugs and shutters.     |

#### Example of Setting the Variables

```bash
export SENSIBO_API_KEY=<your_sensibo_api_key>
export SMARTTHINGS_TOKEN=<your_smartthings_token>
export SWITCHER_TOKEN=<your_switcher_token>
```

## 🚀 Running the Operator
1. Configure environment metadata in *env_spec.yaml* (location, ac and lights settings).
2. Install required dependencies:
```bash
pip install -r requirements.txt

```
3. Run orchestrator:
```bash
python smart_operator.py --engine crewai

```
4. Observe collected states, agent decisions, and applied changes in the console.

*Note*: If --dry-run is set, the system will simulate decisions without applying them.

### Results Evaluation Methodology

The evaluation consists of two complementary parts designed to ensure reliability despite real-home variability (seasonal changes, occupancy, appliance updates):

1. **Common-sense reasoning evaluation**
   The MAS’s reasoning abilities are tested through multiple controlled simulations, executed with every supported multi-agent engine (LangGraph, CrewAI, Agno). These scenarios assess how consistently and accurately each configuration produces safe, logical, and energy-aware decisions.

2. **Energy-efficiency assessment:**
   The impact on household energy use is measured using the AC, the most power-consuming device. The power estimation model follows standard HVAC energy-performance principles, where consumption depends on the temperature deviation from a reference setpoint and includes additional fan loads.

### Evaluation Code & Data

- **Evaluation dataset generation:** [`evaluate.py`](./evaluate.py)
- **Results assessment:** [`evaluation/`](./evaluation/)
