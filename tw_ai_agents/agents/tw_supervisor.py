import inspect
import sqlite3
from typing import Callable, Dict, List, Optional, Union

from langchain.prompts import Prompt
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState, StateSchemaType

from tw_ai_agents.agents.handoff import (
    create_handoff_tool,
    OutputMode,
    _make_call_agent,
)
from tw_ai_agents.agents.message_types.base_message_type import State

conn = sqlite3.connect("checkpoints.sqlite")
memory = SqliteSaver(conn)


class TWSupervisor:
    def __init__(
        self,
        agents: List[Union[CompiledStateGraph, "TWSupervisor"]],
        model: LanguageModelLike,
        description: str,
        memory=None,
        tools: Optional[List[Union[Callable, BaseTool]]] = None,
        prompt: Optional[Prompt] = None,
        state_schema: StateSchemaType = AgentState,
        output_mode: OutputMode = "last_message",
        add_handoff_back_messages: bool = False,
        supervisor_name: str = "supervisor",
    ):
        """

        Args:
            agents: List of agents or supervisors to manage
            model: Language model to use for the supervisor
            tools: Tools to use for the supervisor
            output_mode: Mode for adding managed agents' outputs to the message history in the multi-agent workflow.
                Can be one of:
                - `full_history`: add the entire agent message history
                - `last_message`: add only the last message (default)
            add_handoff_back_messages: Whether to add a pair of (AIMessage, ToolMessage) to the message history
                when returning control to the supervisor to indicate that a handoff has occurred.
            supervisor_name: Name of the supervisor node.
        """
        self.agents = agents
        self.model = model
        self.tools = tools
        self.prompt = prompt
        self.state_schema = state_schema
        self.output_mode = output_mode
        self.add_handoff_back_messages = add_handoff_back_messages
        self.supervisor_name = supervisor_name
        self.compiled_graph = None
        self.description = description
        self.memory = memory

    def _process_agent(self, agent):
        """Process an agent which could be a CompiledStateGraph or another TWSupervisor."""
        if isinstance(agent, TWSupervisor):
            return agent.get_supervisor_compiled_graph()
        return agent

    def get_graph(
        self,
    ) -> StateGraph:
        """Create a multi-agent supervisor.

        Returns:
            A StateGraph representing the supervisor agent system.
        """
        processed_agents = [
            (self._process_agent(agent), agent.description)
            for agent in self.agents
        ]

        agent_names = set()
        for agent, description in processed_agents:
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

        handoff_tools = [
            create_handoff_tool(
                agent_name=agent.name,
                agent_description=description,
            )
            for agent, description in processed_agents
        ]
        all_tools = (self.tools or []) + handoff_tools

        if (
            hasattr(self.model, "bind_tools")
            and "parallel_tool_calls"
            in inspect.signature(self.model.bind_tools).parameters
        ):
            self.model = self.model.bind_tools(
                all_tools, parallel_tool_calls=False
            )

        # Convert tools to the expected format if needed
        tool_executor = ToolNode(all_tools) if all_tools else None

        supervisor_agent = create_react_agent(
            name=self.supervisor_name,
            model=self.model,
            tools=tool_executor,  # Pass the tool executor instead of the raw tools list
            prompt=self.prompt,
            state_schema=self.state_schema,
        )

        builder = StateGraph(self.state_schema)
        builder.add_node(supervisor_agent, destinations=tuple(agent_names))
        builder.add_edge(START, supervisor_agent.name)
        for agent, description in processed_agents:
            builder.add_node(
                agent.name,
                _make_call_agent(
                    agent,
                    self.output_mode,
                    self.add_handoff_back_messages,
                    self.supervisor_name,
                ),
            )
            builder.add_edge(agent.name, supervisor_agent.name)

        return builder

    def get_supervisor_compiled_graph(self) -> CompiledStateGraph:
        if self.compiled_graph is None:
            self.compiled_graph = self.get_graph().compile(
                debug=False,
                name=self.supervisor_name,
                checkpointer=self.memory,
            )
        return self.compiled_graph

    def get_pretty_description(self):
        return f"{self.supervisor_name}: {self.description}"

    def _process_output_chunk(self, chunk, result):
        """Process a chunk of output from the graph execution and update the result dictionary.

        :params
            chunk: Output chunk from graph execution
            result: Dictionary containing messages and metadata to update
        :return
            Updated result dictionary
        """
        if "__interrupt__" in chunk:
            # human interruption
            new_message = AIMessage(content=chunk["__interrupt__"][0].value)
            metadata = {
                "ns": chunk["__interrupt__"][0].ns,
                "target_entity": "agent",
            }
            result["messages"].append(new_message)
            result["metadata"].update(metadata)
        else:
            for key, values in chunk.items():
                if isinstance(values, list):
                    for value in values:
                        result["messages"].extend(value.get("messages", []))
                        result["metadata"].update(value.get("metadata", {}))
                else:
                    result["messages"].extend(values.get("messages", []))
                    result["metadata"].update(values.get("metadata", {}))
        return result

    def _extract_tool_calls(self, messages):
        """Extract tool calls from a list of messages.

        :params
            messages: List of messages to process
        :return
            List of extracted tool calls
        """
        tool_calls = []
        for message in messages:
            if (
                hasattr(message, "additional_kwargs")
                and message.additional_kwargs
            ):
                if "tool_calls" in message.additional_kwargs:
                    for tool_call in message.additional_kwargs["tool_calls"]:
                        tool_calls.append(
                            {
                                "tool_name": tool_call.get("function", {}).get(
                                    "name", "unknown"
                                ),
                                "tool_input": tool_call.get("function", {}).get(
                                    "arguments", "{}"
                                ),
                                "tool_id": tool_call.get("id", "unknown"),
                            }
                        )
        return tool_calls

    def run_supervisor(self, state: State, config) -> Dict:
        """Run the supervisor agent system with metadata tracking synchronously.

        :params
            state: The initial state containing messages and other context
            config: Configuration for the graph execution
        :return
            The updated state with results and metadata about the execution
        """
        graph = self.get_supervisor_compiled_graph()
        result = {"messages": [], "metadata": {"tool_calls": []}}

        for chunk in graph.stream(state, config=config):
            result = self._process_output_chunk(chunk, result)

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["tool_calls"] = self._extract_tool_calls(
            result["messages"]
        )
        return result

    async def arun_supervisor(self, state: State, config) -> Dict:
        """Run the supervisor agent system with metadata tracking asynchronously.

        :params
            state: The initial state containing messages and other context
            config: Configuration for the graph execution
        :return
            The updated state with results and metadata about the execution
        """
        graph = self.get_supervisor_compiled_graph()
        result = {"messages": [], "metadata": {"tool_calls": []}}

        async for chunk in graph.astream(state, config=config):
            result = self._process_output_chunk(chunk, result)

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["tool_calls"] = self._extract_tool_calls(
            result["messages"]
        )
        return result
