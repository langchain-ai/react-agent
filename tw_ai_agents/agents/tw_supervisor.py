import inspect
import sqlite3
from typing import Callable, Dict, List, Optional, Union, Tuple, Any

from langchain.prompts import Prompt
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import END
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState, StateSchemaType

from tw_ai_agents.agents.handoff import (
    OutputMode,
    _make_call_agent,
    create_handoff_tool,
    SUBAGENT_TOOL_NAME_PREFIX,
)
from tw_ai_agents.agents.message_types.base_message_type import (
    State,
    ToolMessageInfo,
    InterruptBaseModel,
)
from tw_ai_agents.agents.tools.human_tools import (
    COMPLETE_HANDOFF_STRING,
)

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
        dependant_agents: Optional[List[Any]] = None,
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
            dependant_agents: List of agents which receive all messages and metadata from the supervisor.
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
        self.dependant_agents = dependant_agents or []

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
        ] + [agent.create_handoff_tool() for agent in self.dependant_agents]
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

        edges_from_nodes = []

        if self.dependant_agents:
            for node in self.dependant_agents:
                edges_from_nodes.append(node.name)

        builder.add_node(
            supervisor_agent,
            destinations=tuple(list(agent_names) + edges_from_nodes),
        )
        builder.add_edge(START, supervisor_agent.name)

        # Add all other agents - they can only go back to supervisor
        for agent, description in processed_agents:
            builder.add_node(
                agent.name,  # this string has to be same used in the Command object inside the handoff_tool
                _make_call_agent(
                    agent,
                    self.add_handoff_back_messages,
                    self.supervisor_name,
                ),
            )
            builder.add_edge(agent.name, supervisor_agent.name)

        for agent in self.dependant_agents:
            builder.add_node(
                agent.name,  # this string has to be same used in the Command object inside the handoff_tool
                agent.make_call_dependant_agent(),
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
            interrupt_content_base = chunk["__interrupt__"][0].value
            if isinstance(interrupt_content_base, str):
                # When it is a complete handoff
                interrupt_content = InterruptBaseModel.parse_obj(
                    {"user_message": interrupt_content_base}
                )
            else:
                interrupt_content = InterruptBaseModel.parse_obj(
                    chunk["__interrupt__"][0].value
                )

            new_message = AIMessage(content=interrupt_content.user_message)
            metadata = {
                "ns": chunk["__interrupt__"][0].ns,
                "target_entity": interrupt_content.destination,
                "complete_handoff": COMPLETE_HANDOFF_STRING
                == new_message.content,
            }
            result["messages"].append(new_message)
            result["metadata"].update(metadata)
            result["tools_called"].extend(interrupt_content.tools_called)
        else:
            for key, values in chunk.items():
                # we always want the last
                if isinstance(values, list):
                    for value in values:
                        result["messages"].extend(value.get("messages", []))
                        result["tools_called"].extend(
                            value.get("tools_called", [])
                        )
                        result["metadata"].update(value.get("metadata", {}))
                else:
                    result["messages"].extend(values.get("messages", []))
                    result["tools_called"].extend(
                        values.get("tools_called", [])
                    )
                    result["metadata"].update(values.get("metadata", {}))
        return result

    def _extract_tool_calls(
        self, tool_message_infos: List[ToolMessageInfo]
    ) -> List[Dict]:
        """Extract tool calls from a list of ToolMessageInfo objects.

        :params tool_message_infos: List of ToolMessageInfo objects
        :return List of dictionaries containing tool call information
        """
        tool_calls = []
        for tool_message_info in tool_message_infos:
            if not isinstance(tool_message_info, ToolMessageInfo):
                raise ValueError(
                    f"Expected ToolMessageInfo object, got {type(tool_message_info)}"
                )
            a = 1
            # Filter out our sub-agents from the tool call list.
            # The ones whose name starts with SUBAGENT_TOOL_NAME_PREFIX
            if (
                tool_message_info.name is not None
                and not tool_message_info.name.startswith(
                    SUBAGENT_TOOL_NAME_PREFIX
                )
            ):
                a = 1
                tool_calls.append(
                    {
                        "tool_name": tool_message_info.name,
                        "tool_input": tool_message_info.parameters,
                        "tool_id": tool_message_info.tool_call_id,
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

        result = None
        for chunk in graph.stream(state, config=config):
            # We only want the latest result
            result = {"messages": [], "metadata": {}, "tools_called": []}
            result = self._process_output_chunk(chunk, result)

        if result is None:
            raise ValueError("No output from the graph execution")

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["tool_calls"] = self._extract_tool_calls(
            result["tools_called"]
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

        async for chunk in graph.astream(state, config=config):
            result = {"messages": [], "metadata": {}, "tools_called": []}
            result = self._process_output_chunk(chunk, result)

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["tool_calls"] = self._extract_tool_calls(
            result["messages"]
        )
        return result
