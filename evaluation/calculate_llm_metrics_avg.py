import yaml
from collections import defaultdict

YAML_PATH = "evaluation_results.yaml"

# metrics to average
METRICS = [
    "response-time",
    "personalization-score",
    "contextual-score",
    "interpretability-score"
]

# storage for sums and counts
stats = {
    "crewai": defaultdict(float),
    "agno": defaultdict(float),
    "langgraph": defaultdict(float)
}

counts = {
    "crewai": 0,
    "agno": 0,
    "langgraph": 0
}

with open(YAML_PATH, "r") as f:
    data = list(yaml.safe_load_all(f))

for entry in data:
    outputs = entry.get("output", {})
    
    for engine in ["crewai", "agno", "langgraph"]:
        if engine not in outputs:
            continue
        
        engine_data = outputs[engine]
        counts[engine] += 1
        
        for metric in METRICS:
            value = engine_data.get(metric)
            if isinstance(value, (int, float)):
                stats[engine][metric] += value


print("\n=== AVERAGE METRICS BY ENGINE ===\n")

for engine in stats:
    if counts[engine] == 0:
        print(f"{engine}: no data\n")
        continue

    print(f"{engine}:")
    for metric in METRICS:
        avg = stats[engine][metric] / counts[engine]
        print(f"  {metric}: {avg:.2f}")
    print()
