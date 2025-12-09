import os
import re
import sys
import yaml
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

from ollama import Client as OllamaClient
from agno.agent import Agent
from agno.team.team import Team
from agno.knowledge.knowledge import Knowledge
from agno.db.sqlite import SqliteDb
from agno.memory.manager import MemoryManager
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.models.ollama import Ollama
from agno.utils.pprint import pprint_run_response
from agno.utils.log import set_log_level_to_warning

from clients.constants import *
from clients.utils import get_inputs

KNOWLEDGE_DB_TABLE = "agents_knowledge"

set_log_level_to_warning()

# --------------------
# MODEL LOADING
# --------------------
def load_model() -> Ollama:
    """
    Load and return an Ollama model instance.

    Returns:
        Ollama: Model instance initialized with OLLAMA backend.
    """
    client_config = {
        "temperature": str(OLLAMA_MODEL_TEMPERATURE),
        "num_gpu":str(OLLAMA_MODEL_GPUS),
        "num_thread": str(OLLAMA_MODEL_THREADS),
        "seed": str(OLLAMA_MODEL_SEED),
    }
    client = OllamaClient(
        host=OLLAMA_BASE_URL,
        headers=client_config
    )
    return Ollama(id=OLLAMA_MODEL_NAME, client=client)


# --------------------
# KNOWLEDGE BASE
# --------------------
def setup_knowledge() -> Tuple[LanceDb, SqliteDb]:
    """
    Initialize and return a LanceDb vector database for embeddings.

    Returns:
        LanceDb: Vector database configured for hybrid search.
        SqliteDb: Content database.
    """
    cwd = Path(__file__).parent
    tmp_dir = cwd.joinpath("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    contents_db = SqliteDb(db_file=tmp_dir.joinpath("data.db"))
    vector_db = LanceDb(
        uri=str(tmp_dir.joinpath("agents_knowledge")),
        table_name=KNOWLEDGE_DB_TABLE,
        search_type=SearchType.hybrid,
        embedder=SentenceTransformerEmbedder(),
    )
    return vector_db, contents_db


async def create_knowledge() -> Knowledge:
    """
    Create and return a Knowledge object backed by LanceDb.
    It will automatically load all files from the knowledge_base/ folder.
    """
    vector_db, contents_db = setup_knowledge()
    knowledge = Knowledge(
        vector_db=vector_db,
        contents_db=contents_db,
        name=KNOWLEDGE_DB_TABLE
    )
    
    knowledge_dir = Path(KB_PATH)
    knowledge_files = [str(p) for p in knowledge_dir.rglob("*.md") if p.is_file()]

    if knowledge_files:
        await knowledge.add_contents_async(paths=knowledge_files, skip_if_exists=True, 
            name=KNOWLEDGE_DB_TABLE, description=KNOWLEDGE_DB_TABLE)

    return knowledge

# --------------------
# AGENT MEMORY
# --------------------
def create_memory(model: Ollama) -> MemoryManager:
    """
    Create and return a MemoryManager object backed by SqliteDb for local storage.
    """
    cwd = Path(__file__).parent
    tmp_dir = cwd.joinpath("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    memory_db = SqliteDb(db_file=str(tmp_dir.joinpath("agents_memory.db")))
    return MemoryManager(
        model=model,
        db=memory_db,
        add_memories=True,
        update_memories=True,
    )

# --------------------
# AGENT FACTORY
# --------------------
def create_agent(
        model: Ollama,
        knowledge: Knowledge,
        inputs: Dict[str, Any],
        name: str,
        role: str,
        guideline_key: str,
        state_key: str,
    ) -> Agent:
    """
    General-purpose factory function to create agents.

    Args:
        model: LLM model instance.
        knowledge: Shared knowledge base.
        inputs: Current state and guidelines.
        name: Agent's display name.
        role: Agent's role/purpose description.
        guideline_key: Key in inputs for guideline section.
        state_key: Key in inputs for current state section.

    Returns:
        Agent: Configured agent instance.
    """
    return Agent(
        name=name,
        role=role,
        model=model,
        knowledge=knowledge,
        instructions=[
            f"guidelines: {inputs[guideline_key]}",
            f"current_state: {inputs[state_key]}",
            f"weather: {inputs['current_weather_state']}",
            f"current_daytime: {inputs['current_daytime']}",
        ],
    )


# --------------------
# TEAM
# --------------------
def create_team(model: Ollama, knowledge: Knowledge, memory_manager: MemoryManager, inputs: Dict[str, Any]) -> Team:
    """
    Create and return the Smart Agents Team.

    Args:
        model: Model instance for agents.
        knowledge: Knowledge instance for shared rules.
        memory_manager: MemoryManager instance for shared memory.
        inputs: Current state and guidelines.

    Returns:
        Team: Configured multi-agent team.
    """
    with open(f"{CONFIG_PATH}/tasks.yaml", "r", encoding="utf-8") as file:
        tasks = yaml.safe_load(file)

    ac_agent = create_agent(
        model, knowledge, inputs,
        name="AC Agent",
        role="Optimize AC usage based on state and guidelines",
        guideline_key="ac_guidelines",
        state_key="current_ac_state",
    )

    lights_agent = create_agent(
        model, knowledge, inputs,
        name="Lights Agent",
        role="Optimize lighting conditions for energy efficiency",
        guideline_key="lights_guidelines",
        state_key="current_lights_state",
    )

    fridge_agent = create_agent(
        model, knowledge, inputs,
        name="Fridge Agent",
        role="Optimize fridge and freezer settings for efficiency",
        guideline_key="fridge_guidelines",
        state_key="current_fridge_state",
    )

    shutter_agent = create_agent(
        model, knowledge, inputs,
        name="Shutter Agent",
        role="Adjust the window shutter position",
        guideline_key="shutter_guidelines",
        state_key="current_shutter_state",
    )

    return Team(
        name="Team of Smart Agents",
        members=[ac_agent, lights_agent, fridge_agent, shutter_agent],
        instructions=[
            f"Your inputs: {inputs}",
            f"Your task: {tasks['manager_task']['description']}",
            f"Expected output: {tasks['manager_task']['expected_output']}",
        ],
        model=model,
        knowledge=knowledge,
        db=memory_manager.db,
        memory_manager=memory_manager,
        add_history_to_context=True,
        enable_agentic_memory=True
    )

# --------------------
# RESPONSE PARSER
# --------------------
def extract_json_from_markdown(response_text: str) -> dict:
    """
    Extract JSON object from a markdown ```json ... ``` block in the agent response.
    
    Args:
        response_text: Raw string returned by the agent, containing markdown JSON block.
        
    Returns:
        Parsed Python dictionary if successful, else empty dict.
    """
    # Match ```json ... ``` block
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1)

    return response_text

# --------------------
# MAIN EXECUTION
# --------------------
async def run(current_state: Dict[str, Any] = TRAIN_SET_STATE) -> Any:
    """
    Run the Smart Agents Team with the given state.

    Args:
        current_state: Smart home state dictionary.

    Returns:
        Team response after execution.
    """
    inputs = get_inputs(current_state)
    model = load_model()
    knowledge = await create_knowledge()
    memory = create_memory(model)
    team = create_team(model, knowledge, memory, inputs)
    response = team.run("Tell me the recommended state according to the current conditions. \
Integrate outputs from all agents into a single JSON and add an alert if needed.")
    pprint_run_response(response, markdown=True, show_time=True)
    response_json = extract_json_from_markdown(response.to_dict()["content"])
    return response_json


if __name__ == "__main__":
    response = run()
    print(response)
