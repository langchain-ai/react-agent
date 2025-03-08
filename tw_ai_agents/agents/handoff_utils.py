from tw_ai_agents.config_handler.constants import (
    WHITESPACE_RE,
)


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize an agent name to be used inside the tool name.

    Args:
        agent_name: The name of the agent to normalize.

    Returns:
        A normalized version of the agent name.
    """
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()
