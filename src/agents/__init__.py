"""Agents module - orchestrator and specialized agents."""
from src.agents.shared import Deps, create_llm_client, create_embedding_client
from src.agents.orchestrator_agent import stream_orchestrated, update_memory
from src.agents.retrieval_agent import retrieval_agent, run_retrieval
from src.agents.web_agent import web_agent, run_web_task
from src.agents.code_agent import code_agent, run_code_task

# Keep backward compatibility with original agent
from src.agents.assistant_agent import stream_messages, agent

__all__ = [
    # Orchestrated approach (recommended)
    'stream_orchestrated',
    'update_memory',
    # Specialized agents
    'retrieval_agent',
    'web_agent',
    'code_agent',
    'run_retrieval',
    'run_web_task',
    'run_code_task',
    # Shared
    'Deps',
    'create_llm_client',
    'create_embedding_client',
    # Legacy
    'stream_messages',
    'agent',
]
