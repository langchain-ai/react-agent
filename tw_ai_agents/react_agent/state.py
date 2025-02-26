"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class StateWithRemainingSteps(MessagesState):
    """State schema for the agent.

    This defines the state schema that LangGraph expects for React agents.
    It must include 'messages' and 'remaining_steps' as per LangGraph requirements.
    """

    remaining_steps: int
    """Number of remaining steps before termination."""

    is_last_step: bool
    """Indicates whether the current step is the last one before the graph raises an error."""

    # Additional attributes can be added here as needed
    # Each attribute should have a proper type annotation
