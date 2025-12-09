# Ollama model configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "gpt-oss:20b" # llm model
OLLAMA_MODEL_TEMPERATURE = 0.2 # for more deterministic responses
OLLAMA_MODEL_SEED = 42 # for repetitive responses
OLLAMA_MODEL_GPUS = 1
OLLAMA_MODEL_THREADS = 20
# embedding model for knowledge encoding with CrewAI
OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large:335m"
# embedding model for knowledge encoding with LangGraph
HUGGING_FACE_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

# File paths
CONFIG_PATH = "config" # configuration specs root folder
KB_PATH = "knowledge" # knowledge-base root folder

# Test state sample for simulation
TRAIN_SET_STATE = {
  'Weather': {'temperature': 30.9, 'wind-speed': 8.7, 'wind-direction': 275},
  'PowerConsumption': {'level': 'Critical', 'watt': 2800},
  'AC': {'temperature': 23, 'fan': 'low', 'mode': 'cool', 'on': True},
  'Lights': {'on': True, 'brightness': 50, 'warm': False},
  'Fridge': {'cooler': 3, 'freezer': -20},
  'Shutter': {'position': 77},
  'DayTime': {'datetime': '2025-09-14T15:32:00'},
}
