import copy
import inspect
import re
from typing import Callable, Dict, List, Optional, Tuple
from typing import Union, Any

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.prebuilt.chat_agent_executor import StateSchemaType
from langgraph.types import Command
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel
from typing_extensions import Annotated

from tw_ai_agents.agents.handoff_utils import _normalize_agent_name
from tw_ai_agents.agents.message_types.base_message_type import (
    ToolMessageInfo,
    State,
)
from tw_ai_agents.agents.router_entrypoint import (
    ROUTER_NODE_NAME,
    initial_router_node,
)
from tw_ai_agents.config_handler.constants import (
    SUBAGENT_TOOL_NAME_PREFIX,
    SUBAGENT_TOOL_NAME_SUFFIX,
    TW_SUPERVISOR_NAME,
)


class CaseAgent:
    def __init__(
        self,
        agents: List[Union[CompiledStateGraph, "ToolAgent"]],
        tools: List[Union[Callable, BaseTool]],
        model: LanguageModelLike,
        prompt: Optional[str],
        state_schema: StateSchemaType,
        name: str,
        description: str,
        memory=None,
        dependant_agents: Optional[List[Any]] = None,
    ):
        """
        Initialize a sub-agent which handles a specific case.

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

        self.is_case_agent = True

        self.compiled_graph = None

    def get_pretty_description(self):
        """Return a formatted description of this agent.

        :return
            String description of the agent
        """
        return f"{self.name}: {self.description}"

    def _process_agent(self, agent):
        """Process an agent which could be a CompiledStateGraph or another TWSupervisor."""
        if isinstance(agent, CaseAgent):
            return agent.get_supervisor_compiled_graph()
        return agent

    def get_graph(
        self,
    ) -> StateGraph:
        """
        Create a case agent graph.

        Returns:
            A StateGraph representing the case agent.
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

        if self.is_case_agent:
            # It's is true on the CaseAgent, but not on the ToolAgent
            go_back_to_supervisor_tools = [
                self.create_go_back_to_supervisor_tools()
            ]
        else:
            go_back_to_supervisor_tools = []

        handoff_tools = [
            self.create_handoff_tool(
                agent_name=agent.name,
                agent_description=description,
            )
            for agent, description in processed_agents
        ]
        all_tools = (
            (self.tools or []) + handoff_tools + go_back_to_supervisor_tools
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
        this_agent_destinations = [END]
        if self.is_case_agent:
            this_agent_destinations.append(ROUTER_NODE_NAME)
        builder.add_node(
            this_case_agent,
            destinations=tuple(this_agent_destinations),
        )

        builder.add_edge(START, this_case_agent.name)

        # Add all other agents - they can only go back to supervisor
        for agent, description in processed_agents:
            builder.add_node(
                agent.name,
                self._make_call_agent(
                    agent,
                ),
            )
            builder.add_edge(agent.name, this_case_agent.name)

        if self.is_case_agent:
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
        """
        Create a tool to go back to the supervisor.
        This happens when the case agent realizes it should not handle the case and should let the supervisor handle it.
        """
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

    def _make_call_agent(
        self,
        agent: CompiledStateGraph,
    ) -> Callable[[Dict], Dict]:
        """
        Create a function that calls an agent and processes its output.
        This function is what is actually executed after the handoff tool is called.
        Here is where we process messages going to and coming from sub-agents.

        This function passes only the last ToolMessage to the sub-agent (see _process_input), which contains the message from the supervisor.
        The case agent will receive only the last message from the sub-agent (see _process_output), inside the response of the ToolMessage.

        Args:
            agent: The agent to call.

        Returns:
            A callable that invokes the agent and processes its output.
        """

        def _process_output(
            output: Dict, old_messages: Optional[Dict] = None
        ) -> agent.output_schema:
            output = output.copy()
            old_messages = old_messages or {}
            old_messages = old_messages.copy()
            all_messages = old_messages + output["messages"]

            # Remove last three messages, which are
            # 1. the Tool message generated by the handoff function
            # 2. the HumanMessage we generate in process_input
            # 3. the AI Message generated as response from the Tool/Sub-agent
            # Replace the content of the tool message with the reply from the subagent

            # Find the last ToolMessage that comes before the last HumanMessage
            last_tool_idx = None

            # Iterate in reverse to find the first HumanMessage and check if the previous one is a ToolMessage
            for i in range(len(all_messages) - 1, 0, -1):
                if (
                    isinstance(all_messages[i], HumanMessage)
                    and getattr(all_messages[i], "from_user", True) == False
                ):
                    if isinstance(all_messages[i - 1], ToolMessage):
                        last_tool_idx = i - 1
                    break

            if last_tool_idx is None:
                raise ValueError(
                    "Could not find appropriate ToolMessage to update"
                )

            # Update the content of that ToolMessage with the final AI response, to give them back to UI
            new_tools_called = []
            potential_user_messages = []
            for idx in range(last_tool_idx, len(all_messages)):
                message = all_messages[idx]
                if isinstance(message, ToolMessage):
                    # get the previous ai message, if available. It contains the params of the tool call
                    previous_ai_message = (
                        all_messages[idx - 1] if idx > 0 else None
                    )
                    parameters = (
                        previous_ai_message.tool_calls[0].get("args", {})
                        if previous_ai_message
                        else {}
                    )

                    new_tools_called.append(
                        ToolMessageInfo(
                            content=message.content,
                            name=message.name,
                            tool_call_id=message.tool_call_id,
                            id=message.id,
                            parameters=parameters,
                        )
                    )

            to_return_message = all_messages[last_tool_idx]
            to_return_message.content = f"{all_messages[-1].content}"

            return {
                **output,
                "messages": to_return_message,
                # we add new tools before because this code is called when going back up on the graph.
                "tools_called": new_tools_called + output["tools_called"],
                # Remove last message from supervisor, as it was used in the supervisor processing
                "message_from_supervisor": [None],
            }

        def _process_input(
            input: State,
        ) -> Tuple[agent.input_schema, Optional[List[BaseMessage]]]:
            # return on the last ToolMessage, convert it to a HumanMessage
            last_message = input["messages"][-1]
            other_messages = input["messages"]
            # qua come content devo prendere il message from subagent
            if isinstance(last_message, ToolMessage):
                last_message = HumanMessage(
                    last_message.content, from_user=False
                )
            else:
                raise ValueError(
                    f"Expected last message to be a ToolMessage, got {type(last_message)}"
                )
            input["messages"] = [last_message]
            return input, other_messages

        def call_agent(input_state: Dict) -> Dict:
            state, old_messages = _process_input(input_state)
            output = agent.invoke(state)
            return _process_output(output, old_messages)

        async def acall_agent(state: Dict) -> Dict:
            state, old_messages = _process_input(state)
            output = await agent.ainvoke(state)
            return _process_output(output, old_messages)

        return RunnableCallable(call_agent, acall_agent)

    def create_handoff_tool(
        self, agent_name: str, agent_description: str = ""
    ) -> BaseTool:
        """Create a tool that can handoff control to the requested agent.

        Args:
            agent_name: The name of the agent to handoff control to
            agent_description: The description of the agent to handoff control to. Will be used by the model to determine what the sub-agent is good at.
        Returns:
            A tool that can be used to transfer control to another agent.
        """
        tool_name = (
            SUBAGENT_TOOL_NAME_PREFIX
            + f"{_normalize_agent_name(agent_name)}"
            + SUBAGENT_TOOL_NAME_SUFFIX
        )

        class BaseArgsSchema(BaseModel):
            tool_call_id: Annotated[str, InjectedToolCallId]
            message_for_subagent: str

        @tool(
            tool_name, description=agent_description, args_schema=BaseArgsSchema
        )
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
                update={
                    "messages": [tool_message],
                    "message_from_supervisor": [message_for_subagent],
                },
            )

        return handoff_to_agent
