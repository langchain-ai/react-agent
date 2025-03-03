"""Handoff functionality for supervisor agent system.

This module provides tools for transferring control between agents in a multi-agent system.
"""

import re
import uuid
from typing import Literal, Callable, Dict, Optional, Tuple, List

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    ToolCall,
    BaseMessage,
    HumanMessage,
)
from langchain_core.tools import tool, BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel
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


def create_handoff_tool(
    *, agent_name: str, agent_description: str = ""
) -> BaseTool:
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

    class BaseArgsSchema(BaseModel):
        tool_call_id: Annotated[str, InjectedToolCallId]
        message_for_subagent: str

    @tool(tool_name, description=agent_description, args_schema=BaseArgsSchema)
    def handoff_to_agent(
        tool_call_id: Annotated[str, InjectedToolCallId],
        message_for_subagent: str,
    ) -> Command:
        """Ask another agent for help."""
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}\n\n"
            f"## Message from the supervisor\n"
            f"{message_for_subagent}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": [tool_message]},
        )

    return handoff_to_agent


def create_handoff_back_messages(
    agent_name: str, supervisor_name: str
) -> tuple[AIMessage, ToolMessage]:
    """Create a pair of (AIMessage, ToolMessage) to add to the message history when returning control to the supervisor."""
    tool_call_id = str(uuid.uuid4())
    tool_name = f"transfer_back_to_{_normalize_agent_name(supervisor_name)}"
    tool_calls = [ToolCall(name=tool_name, args={}, id=tool_call_id)]
    return (
        AIMessage(
            content=f"Transferring back to {supervisor_name}",
            tool_calls=tool_calls,
            name=agent_name,
        ),
        ToolMessage(
            content=f"Successfully transferred back to {supervisor_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        ),
    )


OutputMode = Literal["full_history", "last_message"]


def _make_call_agent(
    agent: CompiledStateGraph,
    output_mode: OutputMode,
    add_handoff_back_messages: bool,
    supervisor_name: str,
    input_mode: OutputMode = "last_message",
) -> Callable[[Dict], Dict]:
    """
    Create a function that calls an agent and processes its output.
    This function is what is actually executed when the handoff to a sub-agent happens.
    Here is were we process messages going to and coming from sub-agents.

    Args:
        agent: The agent to call.
        output_mode: How to handle the agent's message history.
        add_handoff_back_messages: Whether to add handoff back messages.
        supervisor_name: The name of the supervisor agent.

    Returns:
        A callable that invokes the agent and processes its output.
    """
    if output_mode not in OutputMode.__args__:  # type: ignore
        raise ValueError(
            f"Invalid agent output mode: {output_mode}. "
            f"Needs to be one of {OutputMode.__args__}"  # type: ignore
        )

    def _process_output(
        output: Dict, old_messages: Optional[Dict] = None
    ) -> Dict:
        messages = output["messages"]
        if output_mode == "full_history":
            messages = old_messages + messages
        elif output_mode == "last_message":
            messages = messages[-1:]
        else:
            raise ValueError(
                f"Invalid agent output mode: {output_mode}. "
                f"Needs to be one of {OutputMode.__args__}"  # type: ignore
            )

        if add_handoff_back_messages:
            # Add handoff back messages using AIMessage and HumanMessage
            messages.extend(
                create_handoff_back_messages(agent.name, supervisor_name)
            )

        # Add From Supervisor to the last message
        last_message = messages[-1]
        last_message.content = (
            f"## Response from sub-agent\n{last_message.content}"
        )
        messages[-1] = last_message

        return {
            **output,
            "messages": messages,
        }

    def _process_input(input: Dict) -> Tuple[Dict, Optional[List[BaseMessage]]]:
        if input_mode == "last_message":
            # return on the last ToolMessage, convert it to a HumanMessage
            last_message = input["messages"][-1]
            other_messages = input["messages"][:-1]
            if isinstance(last_message, ToolMessage):
                last_message = HumanMessage(last_message.content)
            input["messages"] = [last_message]
            return input, other_messages
        elif input_mode == "full_history":
            return input, None
        else:
            raise ValueError(f"Invalid input mode: {input_mode}")

    def call_agent(state: Dict) -> Dict:
        state, old_messages = _process_input(state)
        output = agent.invoke(state)
        return _process_output(output, old_messages)

    async def acall_agent(state: Dict) -> Dict:
        state, old_messages = _process_input(state)
        output = await agent.ainvoke(state)
        return _process_output(output, old_messages)

    return RunnableCallable(call_agent, acall_agent)
