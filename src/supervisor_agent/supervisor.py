"""Supervisor agent for orchestrating multiple specialized agents.

This module provides functionality for creating a supervisor agent that can manage
and coordinate multiple specialized agents in a multi-agent system.
"""

import inspect
from typing import Callable, Dict, List, Literal, Optional, Type, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import LanguageModelLike
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    StateSchemaType,
    Prompt,
    create_react_agent,
)
from langgraph.utils.runnable import RunnableCallable

from supervisor_agent.handoff import (
    create_handoff_tool,
    create_handoff_back_messages,
)


OutputMode = Literal["full_history", "last_message"]
"""Mode for adding agent outputs to the message history in the multi-agent workflow

- `full_history`: add the entire agent message history
- `last_message`: add only the last message
"""


def _make_call_agent(
    agent: CompiledStateGraph,
    output_mode: OutputMode,
    add_handoff_back_messages: bool,
    supervisor_name: str,
) -> Callable[[Dict], Dict]:
    """Create a function that calls an agent and processes its output.
    
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

    def _process_output(output: Dict) -> Dict:
        messages = output["messages"]
        if output_mode == "full_history":
            pass
        elif output_mode == "last_message":
            messages = messages[-1:]
        else:
            raise ValueError(
                f"Invalid agent output mode: {output_mode}. "
                f"Needs to be one of {OutputMode.__args__}"  # type: ignore
            )

        if add_handoff_back_messages:
            # Add handoff back messages using AIMessage and HumanMessage
            messages.extend(create_handoff_back_messages(agent.name, supervisor_name))

        return {
            **output,
            "messages": messages,
        }

    def call_agent(state: Dict) -> Dict:
        output = agent.invoke(state)
        return _process_output(output)

    async def acall_agent(state: Dict) -> Dict:
        output = await agent.ainvoke(state)
        return _process_output(output)

    return RunnableCallable(call_agent, acall_agent)


def create_supervisor(
    agents: List[CompiledStateGraph],
    *,
    model: LanguageModelLike,
    tools: Optional[List[Union[Callable, BaseTool]]] = None,
    prompt: Optional[Prompt] = None,
    state_schema: StateSchemaType = AgentState,
    output_mode: OutputMode = "last_message",
    add_handoff_back_messages: bool = True,
    supervisor_name: str = "supervisor",
) -> StateGraph:
    """Create a multi-agent supervisor.

    Args:
        agents: List of agents to manage
        model: Language model to use for the supervisor
        tools: Tools to use for the supervisor
        prompt: Optional prompt to use for the supervisor. Can be one of:
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.
        state_schema: State schema to use for the supervisor graph.
        output_mode: Mode for adding managed agents' outputs to the message history in the multi-agent workflow.
            Can be one of:
            - `full_history`: add the entire agent message history
            - `last_message`: add only the last message (default)
        add_handoff_back_messages: Whether to add a pair of (AIMessage, ToolMessage) to the message history
            when returning control to the supervisor to indicate that a handoff has occurred.
        supervisor_name: Name of the supervisor node.
        
    Returns:
        A StateGraph representing the supervisor agent system.
    """
    agent_names = set()
    for agent in agents:
        if agent.name is None or agent.name == "LangGraph":
            raise ValueError(
                "Please specify a name when you create your agent, either via `create_react_agent(..., name=agent_name)` "
                "or via `graph.compile(name=name)`."
            )

        if agent.name in agent_names:
            raise ValueError(
                f"Agent with name '{agent.name}' already exists. Agent names must be unique."
            )

        agent_names.add(agent.name)

    handoff_tools = [create_handoff_tool(agent_name=agent.name) for agent in agents]
    all_tools = (tools or []) + handoff_tools

    if (
        hasattr(model, "bind_tools")
        and "parallel_tool_calls" in inspect.signature(model.bind_tools).parameters
    ):
        model = model.bind_tools(all_tools, parallel_tool_calls=False)

    supervisor_agent = create_react_agent(
        name=supervisor_name,
        model=model,
        tools=all_tools,
        prompt=prompt,
        state_schema=state_schema,
    )

    builder = StateGraph(state_schema)
    builder.add_node(supervisor_agent, destinations=tuple(agent_names))
    builder.add_edge(START, supervisor_agent.name)
    for agent in agents:
        builder.add_node(
            agent.name,
            _make_call_agent(
                agent,
                output_mode,
                add_handoff_back_messages,
                supervisor_name,
            ),
        )
        builder.add_edge(agent.name, supervisor_agent.name)

    return builder 