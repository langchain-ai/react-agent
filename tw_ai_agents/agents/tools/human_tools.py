from langchain_core.tools import tool
from langgraph.types import interrupt

COMPLETE_HANDOFF_STRING = "Handoff the full conversation to a real agent."


@tool("real_human_agent_execute_actions")
def real_human_agent_execute_actions(query: str) -> str:
    """Make a real agent execute actions.

    This tool makes a real human agent execute actions based on the provided query.

    Args:
        query: The query to make the real human agent execute.

    Returns:
        A string containing the information from the real human agent. It can be a confirmation or a negative answer.
    """
    a = 1

    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        query,
    )
    print(f"> Received an input from the interrupt: {answer}")
    return answer


@tool("handoff_conversation_to_real_agent")
def handoff_conversation_to_real_agent() -> str:
    """
    Handoff the full conversation to a real agent.
    """
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        COMPLETE_HANDOFF_STRING,
    )
    return answer
