"""Handoff functionality for supervisor agent system.

This module provides tools for transferring control between agents in a multi-agent system.
"""

import re
import uuid
from typing import Tuple, List

from langchain_core.messages import AIMessage, ToolMessage, ToolCall, BaseMessage, HumanMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from typing_extensions import Annotated


WHITESPACE_RE = re.compile(r"\s+")


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize an agent name to be used inside the tool name.
    
    Args:
        agent_name: The name of the agent to normalize.
        
    Returns:
        A normalized version of the agent name.
    """
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def create_handoff_tool(*, agent_name: str) -> BaseTool:
    """Create a tool that can handoff control to the requested agent.

    Args:
        agent_name: The name of the agent to handoff control to, i.e.
            the name of the agent node in the multi-agent graph.
            Agent names should be simple, clear and unique, preferably in snake_case,
            although you are only limited to the names accepted by LangGraph
            nodes as well as the tool names accepted by LLM providers
            (the tool name will look like this: `transfer_to_<agent_name>`).
            
    Returns:
        A tool that can be used to transfer control to another agent.
    """
    tool_name = f"transfer_to_{_normalize_agent_name(agent_name)}"

    @tool(tool_name)
    def handoff_to_agent(
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        message = AIMessage(
            content=f"Successfully transferred to {agent_name}",
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": [message]},
        )

    return handoff_to_agent


def create_handoff_back_messages(agent_name: str, supervisor_name: str) -> List[BaseMessage]:
    """Create messages to indicate a handoff back to the supervisor.
    
    Args:
        agent_name: The name of the agent handing back control.
        supervisor_name: The name of the supervisor receiving control.
        
    Returns:
        A list of messages indicating the handoff.
    """
    return [
        AIMessage(content=f"I've completed my task as {agent_name} and am handing control back to {supervisor_name}."),
        HumanMessage(content=f"Acknowledged. {supervisor_name} is now in control of the conversation.")
    ] 