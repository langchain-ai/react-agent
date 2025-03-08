import inspect
import sqlite3
from typing import Callable, List, Optional, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import END
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import StateSchemaType
from langgraph.types import Command
from pydantic import BaseModel
from tw_ai_agents.agents.handoff_utils import (
    _normalize_agent_name,
)
from tw_ai_agents.config_handler.constants import (
    SUBAGENT_TOOL_NAME_PREFIX,
    SUBAGENT_TOOL_NAME_SUFFIX,
)
from typing_extensions import Annotated

from tw_ai_agents.agents.router_entrypoint import (
    ROUTER_NODE_NAME,
    initial_router_node,
)

conn = sqlite3.connect("checkpoints.sqlite")
memory = SqliteSaver(conn)


class Supervisor:
    def __init__(
        self,
        agents: List[Union[CompiledStateGraph, "CaseAgent"]],
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
