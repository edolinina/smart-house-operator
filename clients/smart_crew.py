import os
import sys
import yaml
from pathlib import Path
from typing import List, Dict, Any

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

from clients.constants import *
from clients.utils import get_inputs

os.environ["POSTHOG_DISABLED"] = "1"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

@CrewBase
class SmartHomeCrew:
    """Smart Home Crew setup for multi-agent energy optimization."""

    # Config file paths (YAML)
    agents_config: str = f'../{CONFIG_PATH}/agents.yaml'
    tasks_config: str = f'../{CONFIG_PATH}/tasks.yaml'

    # Upload knowledge files
    knowledge_dir = Path(KB_PATH)
    files = [f for f in os.listdir(knowledge_dir)]

    knowledge_source: TextFileKnowledgeSource = TextFileKnowledgeSource(
        file_paths=files
    )

    # Initialize config for knowledge embeddeding
    embedder_config: Dict = {
        "provider": "ollama",
        "config": {
            "model": OLLAMA_EMBEDDING_MODEL,
            "url": f"{OLLAMA_BASE_URL}/api/embeddings"
        }
    }

    # Initialize shared LLM instance (Ollama backend)
    ollama_llm: LLM = LLM(
        model=f"ollama/{OLLAMA_MODEL_NAME}",
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_MODEL_TEMPERATURE,
        seed=OLLAMA_MODEL_SEED,
        reasoning_effort="medium"
    )

    # --------------------
    # AGENTS
    # --------------------
    @agent
    def ac_agent(self) -> Agent:
        """AC agent to optimize comfort and efficiency."""
        return Agent(
            config=self.agents_config['ac_agent'],
            allow_delegation=False,
            verbose=True,
            llm=self.ollama_llm
        )

    @agent
    def lights_agent(self) -> Agent:
        """Lights agent to optimize lighting conditions."""
        return Agent(
            config=self.agents_config['lights_agent'],
            allow_delegation=False,
            verbose=True,
            llm=self.ollama_llm
        )

    @agent
    def fridge_agent(self) -> Agent:
        """Fridge agent to ensure food safety while saving energy."""
        return Agent(
            config=self.agents_config['fridge_agent'],
            allow_delegation=False,
            verbose=True,
            llm=self.ollama_llm
        )

    @agent
    def shutter_agent(self) -> Agent:
        """Shutter agent to adjust the window shutter position."""
        return Agent(
            config=self.agents_config['shutter_agent'],
            allow_delegation=False,
            verbose=True,
            llm=self.ollama_llm
        )

    @agent
    def manager_agent(self) -> Agent:
        """Manager agent to coordinate across all sub-agents."""
        return Agent(
            config=self.agents_config['manager_agent'],
            allow_delegation=False,
            verbose=True,
            llm=self.ollama_llm
        )

    # --------------------
    # TASKS
    # --------------------
    @task
    def ac_task(self) -> Task:
        """Task for AC agent."""
        return Task(
            config=self.tasks_config['ac_task'],
            agent=self.ac_agent()
        )

    @task
    def lights_task(self) -> Task:
        """Task for Lights agent."""
        return Task(
            config=self.tasks_config['lights_task'],
            agent=self.lights_agent(),
        )

    @task
    def fridge_task(self) -> Task:
        """Task for Fridge agent."""
        return Task(
            config=self.tasks_config['fridge_task'],
            agent=self.fridge_agent()
        )

    @task
    def shutter_task(self) -> Task:
        """Task for Shutter agent."""
        return Task(
            config=self.tasks_config['shutter_task'],
            agent=self.shutter_agent()
        )

    @task
    def manager_task(self) -> Task:
        """
        Manager task coordinates sub-tasks from AC, Lights, Shutter and Fridge.
        Context must include Task instances, not functions.
        """
        return Task(
            config=self.tasks_config['manager_task'],
            agent=self.manager_agent(),
            context=[self.ac_task(), self.lights_task(), self.fridge_task(), self.shutter_task()]
        )

    # --------------------
    # CREW
    # --------------------
    @crew
    def crew(self) -> Crew:
        """Creates and returns the SmartHome Crew with agents and tasks."""
        return Crew(
            agents=self.agents,  # Populated by @agent decorator
            tasks=self.tasks,    # Populated by @task decorator
            process=Process.sequential,
            async_execution=True,
            model=self.ollama_llm,
            knowledge_sources=[self.knowledge_source],
            memory=True,
            embedder=self.embedder_config
        )


def run(current_state: Dict[str, Any] = TRAIN_SET_STATE) -> Any:
    """
    Run the SmartHome Crew with the given state.

    Args:
        current_state: Smart home state dictionary (defaults to TRAIN_SET_STATE).

    Returns:
        Crew response after kickoff.
    """
    inputs = get_inputs(current_state)
    response = SmartHomeCrew().crew().kickoff(inputs=inputs)
    return response.raw


def train(n_iterations: int, filename: str) -> None:
    """
    Train the crew with repeated runs.

    Args:
        n_iterations: Number of training iterations.
        filename: Output filename for storing training results.
    """
    inputs = get_inputs(TRAIN_SET_STATE)
    SmartHomeCrew().crew().train(
        n_iterations=int(n_iterations),
        filename=filename,
        inputs=inputs
    )


if __name__ == "__main__":
    # Usage: python smart_crew.py <n_iterations> <filename>
    run()
    #train(sys.argv[1], sys.argv[2])
