from langchain_core.tools import tool
from langgraph.types import interrupt

from tw_ai_agents.agents.message_types.base_message_type import (
    AgentMessageMode,
    InterruptBaseModel,
)

COMPLETE_HANDOFF_STRING = "Handoff the full conversation to a real agent."


@tool("real_human_agent_execute_actions")
def real_human_agent_execute_actions(action_request: str) -> str:
    """
    Make a real agent execute actions.
    This tool makes a real human agent execute actions based on the provided query.

    Args:
        action_request: The action request to make the real human agent execute.

    Returns:
        A string containing the information from the real human agent. It can be a confirmation or a negative answer.
    """
    answer = interrupt(
        InterruptBaseModel(
            user_message=action_request,
            tools_called=[],
            destination="agent",
            agent_message_mode=AgentMessageMode.ACTION_REQUEST,
        ).dict(),
    )
    print(f"> Received an input from the interrupt: {answer}")
    return answer


@tool("handoff_conversation_to_real_agent")
def handoff_conversation_to_real_agent() -> str:
    """
    Handoff the full conversation to a real agent.
    """
    answer = interrupt(
        InterruptBaseModel(
            user_message=COMPLETE_HANDOFF_STRING,
            tools_called=[],
            destination="agent",
            agent_message_mode=AgentMessageMode.COMPLETE_HANDOFF,
        ).dict(),
    )
    return answer


@tool("get_information_from_real_agent")
def get_information_from_real_agent(query: str) -> str:
    """
    Get information from a real human agent.
    This tool sends a question to a real human agent and returns their response.

    :params
        query: The question to ask the real human agent.
    :return
        The information provided by the real human agent.
    """
    answer = interrupt(
        InterruptBaseModel(
            user_message=query,
            tools_called=[],
            destination="agent",
            agent_message_mode=AgentMessageMode.QUESTION,
        ).dict(),
    )
    print(f"> Received information from the real agent: {answer}")
    return answer
