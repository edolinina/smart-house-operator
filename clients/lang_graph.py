import os
import yaml
import re
import asyncio
from typing import Any
from pathlib import Path
from functools import partial

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from clients.constants import *
from clients.utils import get_inputs

WORKER_AGENTS = ["AC", "Lights", "Fridge", "Shutter"]

# --------------------
# MODEL LOADING
# --------------------
def load_model() -> ChatOllama:
    """
    Load the Ollama LLM model.

    Returns:
        ChatOllama: Configured Ollama model instance.
    """
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL_NAME,
        temperature=OLLAMA_MODEL_TEMPERATURE
    )

# --------------------
# KNOWLEDGE BASE
# --------------------
async def create_knowledge() -> LanceDB:
    """
    Build a vector database from local Markdown files.

    Returns:
        LanceDB: Vector store populated with embeddings.
    """
    cwd = Path(__file__).parent
    tmp_dir = cwd / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(
        str(Path(KB_PATH)),
        glob="*.md",
        loader_cls=TextLoader
    )
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name=HUGGING_FACE_EMBEDDING
    )

    vectordb = LanceDB.from_documents(
        docs, embeddings, uri=str(tmp_dir / "lancedb")
    )
    return vectordb

# --------------------
# MEMORY
# --------------------
def create_memory() -> AsyncSqliteSaver:
    """
    Create an async SQLite-based checkpointer.

    Returns:
        AsyncSqliteSaver: Memory manager instance for LangGraph.
    """
    tmp_dir = Path(__file__).parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(":memory:")

# --------------------
# AGENT NODES
# --------------------
async def worker_agent(state: dict, agent_name: str) -> dict:
    """
    Worker agent node: analyse how to optimize usage of a specific device.

    Args:
        state (dict): Current shared workflow state (must contain 'model',
            guidelines, device states, etc.).
        agent_name (str): Name of the device/agent ("AC", "Lights", etc.).

    Returns:
        dict: Updated state with `{agent}_recommendation`.
    """
    model = state["model"]
    agent_index = agent_name.lower()
    agent_guidelines = state[f"{agent_index}_guidelines"]
    current_state =  f"""current {agent_index} state: {state[f"current_{agent_index}_state"]},
current weather: {state["current_daytime"]}, current day/time: {state["current_daytime"]}, 
current power consumption: {state["current_power_state"], "power_saving_thresholds"}
"""
    task = f"Optimize {agent_name} usage. Guidelines: {agent_guidelines}, {current_state}"
    answer = await model.ainvoke(task)
    state[f"{agent_index}_recommendation"] = answer.content
    return state

async def manager_agent(state: dict) -> dict:
    """
    Manager agent node: coordinates and aggregates worker outputs.

    Args:
        state (dict): Current shared workflow state, including 'model'.

    Returns:
        dict: Response containing "manager_summary".
    """
    with open(f"{CONFIG_PATH}/tasks.yaml", "r", encoding="utf-8") as file:
        tasks = yaml.safe_load(file)

    model = state["model"]
    task = f"""Your inputs: {state}
Your task: {tasks['manager_task']['description']},
Expected output: {tasks['manager_task']['expected_output']}
Add alert if needed."""
    response = await model.ainvoke(task)
    return {"manager_summary": response.content}

# --------------------
# BUILD LANGGRAPH
# --------------------
def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow.

    Returns:
        StateGraph: Configured workflow graph with workers and manager.
    """
    graph = StateGraph(dict)

    # Add agent nodes
    [graph.add_node(f"{worker_name.lower()}_agent", partial(worker_agent, agent_name=worker_name)) \
        for worker_name in WORKER_AGENTS]
    graph.add_node("manager_agent", manager_agent)

    # the trigger entry point
    graph.set_entry_point("ac_agent")

    # manager's inbound edges - all workers feed back into manager
    graph.add_edge("ac_agent", "manager_agent")
    graph.add_edge("lights_agent", "manager_agent")
    graph.add_edge("shutter_agent", "manager_agent")
    graph.add_edge("fridge_agent", "manager_agent")

    # stop the graph after manager final aggregation
    graph.add_edge("manager_agent", END)

    return graph

# --------------------
# MAIN EXECUTION
# --------------------
async def run(current_state: dict = TRAIN_SET_STATE) -> dict:
    """
    Run the workflow.

    Args:
        current_state (dict, optional): Initial training/test state.
            Defaults to TRAIN_SET_STATE.

    Returns:
        Any: Manager summary dict from the workflow execution.
    """
    state = get_inputs(current_state)
    model = load_model()
    knowledge = await create_knowledge()
    memory = create_memory()

    # update of state with shared model and knowledge for all graph nodes
    state.update({
        "model": model,
    })

    workflow = build_graph()

    # compilation with persistent checkpointer in case of interruptions
    async with create_memory() as memory:
        memory.thread_id = "main_thread"
        app = workflow.compile(checkpointer=memory)
        config = {
            "configurable": {
                "thread_id": "main_thread",
                "knowledge": knowledge,
            }
        }
        result = await app.ainvoke(state, config=config)

    return result.get("manager_summary")

if __name__ == "__main__":
    response = run()
    print(response)
