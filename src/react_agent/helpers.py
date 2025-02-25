"""Helper functions for React agents."""

from typing import Dict, Any, List, Optional

from langchain_core.messages import BaseMessage

from react_agent.state import State


def create_initial_state() -> State:
    """Create an initial state for the agent with proper default values.
    
    Returns:
        A State object with default values.
    """
    return {
        "messages": [],
        "remaining_steps": 10,
        "is_last_step": False,
    }


def prepare_supervisor_state(messages: List[BaseMessage]) -> Dict[str, Any]:
    """Create a state object for the supervisor system.
    
    This function creates a state object with the minimum required fields for the supervisor.
    
    Args:
        messages: The list of messages to include in the state.
        
    Returns:
        A dict with the required state fields.
    """
    return {
        "messages": messages,
        "remaining_steps": 10,
    } 