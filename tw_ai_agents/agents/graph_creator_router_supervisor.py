"""Test script for the orchestrator.

This script tests the orchestrator to ensure it works correctly.
"""

import asyncio
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import Annotated

from tw_ai_agents.agents.message_types.base_message_type import (
    InterruptBaseModel,
    ToolMessageInfo,
)
from tw_ai_agents.agents.tools.actions_retriever import AGENT_LIST
from tw_ai_agents.agents.tools.human_tools import (
    COMPLETE_HANDOFF_STRING,
    get_information_from_real_agent,
)

WHITESPACE_RE = re.compile(r"\s+")
SUBAGENT_TOOL_NAME_PREFIX = f"transfer_to_"
SUBAGENT_TOOL_NAME_SUFFIX = "_agent"

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.constants import END
from langgraph.graph import START
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import StateSchemaType

from tw_ai_agents.agents.handoff import (
    SUBAGENT_TOOL_NAME_PREFIX,
    _normalize_agent_name,
    create_handoff_tool,
    _make_call_agent,
)
from tw_ai_agents.agents.llm_models_loader import get_llm_model
from tw_ai_agents.agents.message_types.base_message_type import State
from tw_ai_agents.agents.tools.human_tools import (
    AskUserTool,
    handoff_conversation_to_real_agent,
    real_human_agent_execute_actions,
)
from tw_ai_agents.agents.tools.tools import get_knowledge_info
from tw_ai_agents.agents.tw_supervisor import TWSupervisor

# Load environment variables
load_dotenv()

ROUTER_NODE_NAME = "router_node"
TW_SUPERVISOR_NAME = "tw_supervisor"


def initial_router_node(state: State) -> State:
    """
    Initial entry point for the graph.
    Decides if to go to the supervisor or the agent.
    """
    return state


def initial_router_sorting_condition(state: State) -> str:
    """
    Initial entry point for the graph.
    Decides if to go to the supervisor or the agent.
    """
    next_node = state.metadata.get("next_node", TW_SUPERVISOR_NAME)
    return next_node


class CaseAgent:
    def __init__(
        self,
        agents: List[Union[CompiledStateGraph, "TWSupervisor"]],
        tools: List[Union[Callable, BaseTool]],
        model: LanguageModelLike,
        prompt: Optional[str],
        state_schema: StateSchemaType,
        name: str,
        description: str,
        memory=None,
        dependant_agents: Optional[List[AskUserTool]] = None,
    ):
        """Initialize a CaseAgent.

        :params
            agents: List of agents or supervisors to manage
            tools: Tools to use for the agent
            model: Language model to use for the agent
            prompt: Prompt to use for the agent
            state_schema: Schema for the agent state
            name: Name of this agent for identification
            description: Description of this agent's capabilities
            memory: Memory store for the agent
            dependant_agents: List of AskUserTool instances this agent depends on
        :return
            None
        """
        self.agents = agents
        self.tools = tools
        self.model = model
        self.prompt = prompt
        self.state_schema = state_schema
        self.name = name
        self.description = description
        self.memory = memory
        self.dependant_agents = dependant_agents

        self.compiled_graph = None

    def get_pretty_description(self):
        """Return a formatted description of this agent.

        :return
            String description of the agent
        """
        return f"{self.name}: {self.description}"

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

        go_back_to_supervisor_tools = self.create_go_back_to_supervisor_tools()

        handoff_tools = [
            create_handoff_tool(
                agent_name=agent.name,
                agent_description=description,
            )
            for agent, description in processed_agents
        ]
        all_tools = (
            (self.tools or []) + handoff_tools + [go_back_to_supervisor_tools]
        )

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

        this_case_agent = create_react_agent(
            name=self.name,
            model=self.model,
            tools=tool_executor,  # Pass the tool executor instead of the raw tools list
            prompt=self.prompt,
            state_schema=self.state_schema,
        )

        builder = StateGraph(self.state_schema)
        builder.add_node(
            this_case_agent,
            destinations=tuple([END, ROUTER_NODE_NAME]),
        )

        builder.add_edge(START, this_case_agent.name)

        # Add all other agents - they can only go back to supervisor
        for agent, description in processed_agents:
            builder.add_node(
                agent.name,  # this string has to be same used in the Command object inside the handoff_tool
                _make_call_agent(
                    agent,
                    add_handoff_back_messages=False,
                    supervisor_name=self.name,
                ),
            )
            builder.add_edge(agent.name, this_case_agent.name)

        builder.add_node(ROUTER_NODE_NAME, initial_router_node)
        builder.add_edge(this_case_agent.name, END)

        return builder

    def get_supervisor_compiled_graph(self) -> CompiledStateGraph:
        if self.compiled_graph is None:
            self.compiled_graph = self.get_graph().compile(
                debug=False,
                name=self.name,
                checkpointer=self.memory,
            )
        return self.compiled_graph

    def create_go_back_to_supervisor_tools(self):
        tool_name = "go_back_to_supervisor"
        go_back_to_supervisor_description = (
            "This tool is used to go back to the supervisor."
        )

        class BaseArgsSchema(BaseModel):
            tool_call_id: Annotated[str, InjectedToolCallId]

        @tool(
            tool_name,
            description=go_back_to_supervisor_description,
            args_schema=BaseArgsSchema,
        )
        def handoff_to_agent(
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Ask another agent for help."""
            tool_message = ToolMessage(
                content=f"Successfully transferred back to the supervisor\n\n",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return Command(
                goto=ROUTER_NODE_NAME,
                graph=Command.PARENT,
                update={
                    "messages": [tool_message],
                    "metadata": {
                        "next_node": TW_SUPERVISOR_NAME,
                    },
                },
            )

        return handoff_to_agent


class Supervisor:
    def __init__(
        self,
        agents: List[Union[CompiledStateGraph, "TWSupervisor"]],
        model: LanguageModelLike,
        prompt: Optional[str],
        state_schema: StateSchemaType,
        supervisor_name: str,
        description: str,
        memory=None,
        tools: Optional[List[Union[Callable, BaseTool]]] = None,
    ):
        """Initialize a Supervisor.

        :params
            agents: List of agents or supervisors to manage
            model: Language model to use for the supervisor
            prompt: Prompt to use for the supervisor
            state_schema: Schema for the supervisor state
            supervisor_name: Name of this supervisor for identification
            description: Description of this supervisor's capabilities
            memory: Memory store for the supervisor
            tools: Tools to use for the supervisor
        :return
            None
        """
        self.agents = agents
        self.model = model
        self.prompt = prompt
        self.state_schema = state_schema
        self.supervisor_name = supervisor_name
        self.description = description
        self.memory = memory
        self.tools = tools or []

        self.compiled_graph = None

    def get_graph(
        self,
    ) -> StateGraph:
        """Create a multi-agent supervisor.

        Returns:
            A StateGraph representing the supervisor agent system.
        """

        handoff_tools = [
            self.create_handoff_tool_to_router_node(
                agent_name=agent.name,
                agent_description=agent.description,
            )
            for agent in self.agents
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
        builder.add_node(
            supervisor_agent,
            destinations=tuple([END, ROUTER_NODE_NAME]),
        )

        builder.add_edge(START, supervisor_agent.name)

        builder.add_node(ROUTER_NODE_NAME, initial_router_node)
        builder.add_edge(supervisor_agent.name, END)

        return builder

    def get_supervisor_compiled_graph(self) -> CompiledStateGraph:
        if self.compiled_graph is None:
            self.compiled_graph = self.get_graph().compile(
                debug=False,
                name=self.supervisor_name,
                checkpointer=self.memory,
            )
        return self.compiled_graph

    def create_handoff_tool_to_router_node(
        self, agent_name: str, agent_description: str = ""
    ) -> BaseTool:
        """
        Create a tool that can handoff control to router node, which will give the control to the requested agent.

        Args:
            agent_name: The name of the agent to handoff control to.
            agent_description: The description of the agent to handoff control to.

        Returns:
            A tool that can be used to transfer control to router node.
        """
        tool_name = (
            SUBAGENT_TOOL_NAME_PREFIX
            + f"{_normalize_agent_name(agent_name)}"
            + SUBAGENT_TOOL_NAME_SUFFIX
        )

        class BaseArgsSchema(BaseModel):
            tool_call_id: Annotated[str, InjectedToolCallId]

        @tool(
            tool_name, description=agent_description, args_schema=BaseArgsSchema
        )
        def handoff_to_agent(
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Ask another agent for help."""
            tool_message = ToolMessage(
                content=f"Successfully transferred to {agent_name}\n\n",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return Command(
                goto=ROUTER_NODE_NAME,
                graph=Command.PARENT,
                update={
                    "messages": [tool_message],
                    "metadata": {
                        "next_node": agent_name,
                    },
                },
            )

        return handoff_to_agent

    def get_pretty_description(self):
        """Return a formatted description of this supervisor.

        :return
            String description of the supervisor
        """
        return f"{self.supervisor_name}: {self.description}"


def get_complete_graph_router_supervisor(
    model, configs: dict, memory, channel_type_id: str
) -> CompiledStateGraph:
    """Test the orchestrator system with a simple query."""
    # Load channel configs
    correct_channel = next(
        (
            channel
            for channel in configs["channels"]
            if channel["channelTypeId"] == channel_type_id
        ),
        None,
    )
    if correct_channel is None:
        raise ValueError("Channel type not found in the configuration data.")
    channel_type = correct_channel["channelType"]["name"]
    channel_rules = correct_channel["instructions"]["text"]
    writer_function_input = {
        "channel_type": channel_type,
        "channel_rules": channel_rules,
    }

    # Load share tools
    supervisor_tools = []
    subagents_list = []
    shared_agents = [
        # Now, create a second supervisor that could potentially be called by the main supervisor
        TWSupervisor(
            agents=[],
            model=model,
            tools=[get_knowledge_info],
            prompt="You are an agent specialized in knowledge information lookup.",
            state_schema=State,
            supervisor_name="knowledge_handler",
            description="Agent able to lookup knowledge information.",
            memory=memory,
        ),
    ]

    shared_tools = [
        handoff_conversation_to_real_agent,
        real_human_agent_execute_actions,
        get_information_from_real_agent,
    ]

    for config in configs["caseCategories"]:
        description = config["description"]
        name = _normalize_agent_name(config["name"])
        instructions_with_tools = config["instructions"]
        instructions = instructions_with_tools["text"]
        action_list = instructions_with_tools["actions"]
        agent_list_as_tools = []
        for action in action_list:
            new_agent = AGENT_LIST[action["id"]]()
            agent_list_as_tools.append(
                TWSupervisor(
                    agents=[],
                    model=model,
                    tools=new_agent.get_tools(),
                    prompt=new_agent.system_prompt,
                    state_schema=State,
                    supervisor_name=f"{name}_{new_agent.node_name}",
                    description=new_agent.description,
                )
            )

        handoff_conditions = config["handoffConditions"]["text"]
        agent_prompt = hub.pull("case_agent_initial_prompt").format(
            instructions=instructions,
            handoff_conditions=handoff_conditions,
            channel_type=channel_type,
            channel_rules=channel_rules,
        )
        subagents_list.append(
            CaseAgent(
                agents=agent_list_as_tools + shared_agents,
                tools=shared_tools,
                model=model,
                prompt=agent_prompt,
                state_schema=State,
                name=name,
                description=description,
                memory=memory,
            )
        )

    # Main Supervisor
    starting_supervisor_prompt = hub.pull("tw-supervisor-system-prompt")
    final_supervisor_prompt = starting_supervisor_prompt.format(
        agents="- "
        + "\n - ".join(
            [subagent.get_pretty_description() for subagent in subagents_list]
        ),
        tools="- "
        + "\n - ".join(
            [tool.get_pretty_description() for tool in supervisor_tools]
        ),
        channel_type=channel_type,
        channel_rules=channel_rules,
    )

    supervisor_system = Supervisor(
        agents=subagents_list,
        model=model,
        prompt=final_supervisor_prompt,
        state_schema=State,
        supervisor_name=TW_SUPERVISOR_NAME,
        description="Agent able to handle the flow of the conversation.",
        memory=memory,
        tools=[handoff_conversation_to_real_agent],
    )

    # Build the whole graph
    builder = StateGraph(State)

    builder.add_node(ROUTER_NODE_NAME, initial_router_node)
    builder.add_node(
        supervisor_system.supervisor_name,
        supervisor_system.get_supervisor_compiled_graph(),
    )
    for case_agent in subagents_list:
        builder.add_node(
            case_agent.name, case_agent.get_supervisor_compiled_graph()
        )
        builder.add_edge(case_agent.name, END)

    builder.add_edge(START, ROUTER_NODE_NAME)
    builder.add_conditional_edges(
        ROUTER_NODE_NAME, initial_router_sorting_condition
    )
    builder.add_edge(supervisor_system.supervisor_name, END)

    complete_graph = builder.compile(
        checkpointer=memory,
        name="TYPEWISE_ALL_GRAPHS",
    )
    return complete_graph


class GraphRunner:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    def run_supervisor(self, state: State, config) -> Dict:
        """Run the supervisor agent system with metadata tracking synchronously.

        :params
            state: The initial state containing messages and other context
            config: Configuration for the graph execution
        :return
            The updated state with results and metadata about the execution
        """

        result = None
        last_message = None
        i = 0
        while (
            last_message is None or isinstance(last_message, ToolMessage)
        ) and i < 5:
            for chunk in self.graph.stream(state, config=config):
                # We only want the latest result
                result = {"messages": [], "metadata": {}, "tools_called": []}
                result = self._process_output_chunk(chunk, result)

                last_message = result["messages"][-1]
                i += 1

        if result is None:
            raise ValueError("No output from the graph execution")

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["tool_calls"] = self._extract_tool_calls(
            result["tools_called"]
        )
        return result

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
                "agent_message_mode": interrupt_content.agent_message_mode,
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


def get_input_configs() -> Dict[str, Any]:
    """
    Fetch configuration data from the API endpoint.

    Returns:
        Dict[str, Any]: The configuration data including company info, channels, and case categories.

    Raises:
        Exception: If there's an error fetching or processing the configuration data.
    """
    response = requests.get(
        "https://gqyrvmzupb.eu-central-1.awsapprunner.com/api/config/all"
    )
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()


if __name__ == "__main__":
    # Run the test
    model = get_llm_model()

    messages = [
        HumanMessage(
            # content="Hello, how are you?\nI'd like to cancel the order I just did!"
            # content="My order number is 423423, and my email f.roberts@gmail.com."
            # content="It is because the desk arrived completly broken. I want to return it."
            content="Hello, how are you?\nI'd like to change the shipping address for my ticket 14983 to Heinrichstrasse 237, Zurich, Switzerland. Please make sure to double check that this was actually done!"
        ),
    ]

    async def main():
        async with AsyncSqliteSaver.from_conn_string(
            "checkpoints.sqlite"
        ) as saver:
            graph = get_complete_graph(model, get_input_configs(), memory=saver)
            # Your code here
            compiled_supervisor = graph.get_supervisor_compiled_graph()
            config = {"configurable": {"thread_id": "thread-5e54334"}}
            # async for event in graph.astream_events(..., config, version="v1"):
            # print(event)

            result = await compiled_supervisor.ainvoke(
                input=State(messages=messages),
                config=config,
            )

            # result_2 = await compiled_supervisor.ainvoke(
            #     input=Command(
            #         resume="Stop. We can't accept the return. We are not doing return for this things."
            #     ),
            #     config=config,
            # )

            a = 1

        last_message = result["messages"][-1]
        print(last_message)

    # Run the async function
    asyncio.run(main())

    a = 1
