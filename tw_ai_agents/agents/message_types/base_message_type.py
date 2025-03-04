from typing import Dict, Any

from langgraph.graph import MessagesState


class State(MessagesState):
    """State for the agent system, extending MessagesState with metadata for tool tracking."""

    next: str
    remaining_steps: int
    metadata: Dict[str, Any]


class SubagentState(State):
    """State for the agent system, extending MessagesState with metadata for tool tracking."""

    message_for_subagent: str
