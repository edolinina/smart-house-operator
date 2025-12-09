# Smart House Operator for Power Saving

The **Smart House Operator** is a multi-agent system designed to **optimize energy consumption** in a household while maintaining comfort and safety.
It integrates intelligent agents for **Air Conditioning (AC)**, **Lighting**, **Fridge**, **Shutter** and a **Manager Agent** that coordinates decisions across the system.

---

## 🧠 Agent Concepts

Smart House Operator is designed using the following principles:

### **Agent Inputs:**
- **Agent's abilities**:
 * Sense and measure the current state of the environment (e.g. weather conditions) and devices (AC, Lights, Fridge, Shutter).
 * Learn the personal preferences of personas currently at home using reinforcement learning.
 * Control smart-home devices via their respective APIs.  
 * Analyze and reason about necessary adjustments based on rules, thresholds, and learned patterns.  
 * Apply corrective actions by sending commands to the corresponding device APIs.  
- **Agent's stimuli** -> the agent’s *current observation* of the environment (e.g., weather, power consumption, current devices' state).  
- **Agent's prior knowledge** -> pre-loaded **rules, guidelines, and thresholds** defining safe and efficient operations and agent's **action space**.
- **Agent's past experience** -> previous agent's actions and their outcomes, stored in agent's memory
- **Agent's goals and preferences** -> reduce **power consumption** while balancing comfort, safety and user-defined rules.

### **Agent output:**
Changes applied to the device states (e.g., new AC temperature, dimmed lights, shutter and fridge adjustments) and alert to the user if unusual or high power consumption values surpass the specified thresholds.

### Agent Characteristcs:**
- **Rational agent** -> decisions maximize performance by balancing power efficiency with comfort and safety.  
- **Reactive agent** -> adapts immediately to changes in environment and states.  
- **Pro-active agent** -> sends **alerts** to the user if critical conditions arise.  
- **Social agent** -> communicates with humans through alerts and explanations, and coordinates with other agents by delegating relevant tasks.
- **Reasoning**:
  - *Deductive reasoning*: applies **rule-based logic** (e.g., circadian lighting rules).  
  - *Inductive reasoning*: leverages **machine learning and pattern recognition** using Ollama LLM for inferring new states.  
- **Multi-agent reasoning** -> agents collaborate through the **Manager Agent** to produce coherent global decisions.  

### Agent Environment Characteristics
- **Partially observable** - agents do not have full information (only current states).
- **Stochastic** - external factors (weather, user actions) are unpredictable.
- **Sequential** - current decisions impact future states and energy use.
- **Dynamic** - environment continuously changes over time (weather, user activity).
- **Continuous** - states like temperature, brightness vary on scales.
- **Known** - guidelines and thresholds are predefined and available to agents.

### Agent Power Saving Thresholds
- **Normal**: < 1500 W
- **Moderate**: 1500–2500 W
- **Critical**: > 2500 W

Agents adjust their outputs depending on these thresholds to maintain efficiency.

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

### Ollama Server
The project's AI backend runs on **Ollama**, a powerful local model server hosted on a **dedicated machine with 88 CPUs, 1 GPU and 128 GB RAM**. Ollama provides a flexible and efficient way to run and manage large language models (LLMs) locally, offering better control over performance, data privacy and latency compared to cloud-based APIs.  

While Ollama can host **any compatible LLM**, for this project **`gpt-oss:20b`** was selected as it demonstrated the best balance between **instruction following, reasoning quality and performance** in the smart-house control context.  

Ollama was chosen for its **ease of integration**, **local execution capabilities** and **modular model management**, making it ideal for experimentation and deployment in environments where **real-time AI reasoning** and **data security** are critical.

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


### 🧠 Adaptive & Decision Making

The Smart House Operator operates as a **closed-loop decision-making system**, where each cycle involves **collecting inputs, reasoning and acting** based on the current environment.

In each one-shot operation, the system:
- **Collects contextual inputs** such as weather, power usage, and user preferences  
- **Delegates reasoning** to a multi-agent decision team (Agno, CrewAI, or LangGraph)  
- **Executes actions** to adjust devices like AC, lights, fridge and shutters accordingly  

Although each decision loop is independent, the system maintains **adaptive behavior** through model-driven reasoning:
- Agents respond to **dynamic and non-stationary conditions** (e.g., changing weather or occupancy)  
- Decisions are optimized using **reinforcement-based personalization** and rule-based constraints  
- The loop ensures **real-time responsiveness** while maintaining safety, comfort and energy efficiency  

This closed-loop decision process enables Smart Home agents to function as **reactive and context-aware controllers**, achieving intelligent adaptability within each operational cycle.


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

