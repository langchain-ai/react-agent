"""Helper functions for React agents."""

from typing import Dict, Any, List

from langchain_core.messages import BaseMessage


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