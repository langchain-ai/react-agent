import inspect
from typing import Callable, Dict, List, Literal, Optional, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    Prompt,
    StateSchemaType,
    create_react_agent,
)
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.utils.runnable import RunnableCallable

from tw_ai_agents.agents.base_agent import State
from tw_ai_agents.agents.supervisor_utils import (
    SUPERVISOR_PROMPT,
    OutputMode,
    _make_call_agent,
)
from tw_ai_agents.agents.zendesk_agent_tools import ZendeskAgentWithTools
from tw_ai_agents.react_agent.utils import load_chat_model
from tw_ai_agents.supervisor_agent.handoff import (
    create_handoff_back_messages,
    create_handoff_tool,
)
from tw_ai_agents.supervisor_agent.specialized_agents import (
    create_knowledge_lookup_agent,
    create_zendesk_retrieval_agent,
    create_zendesk_setter_agent,
)
from tw_ai_agents.supervisor_agent.tools import (
    SUPERVISOR_TOOLS,
    get_knowledge_info,
    set_ticket_info,
    set_ticket_shipping_address,
)


class TWSupervisor:
    def __init__(
        self,
        agents: List[Union[CompiledStateGraph, "TWSupervisor"]],
        model: LanguageModelLike,
        description: str,
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
            state_schema: State schema to use for the supervisor graph.
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
        # all_tools = handoff_tools
        if (
            hasattr(self.model, "bind_tools")
            and "parallel_tool_calls"
            in inspect.signature(self.model.bind_tools).parameters
        ):
            self.model = self.model.bind_tools(
                all_tools, parallel_tool_calls=False
            )

        # Convert tools to the expected format if needed
        tool_executor = ToolExecutor(all_tools) if all_tools else None

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

    def get_supervisor_compiled_graph(self):
        if self.compiled_graph is None:
            self.compiled_graph = self.get_graph().compile(
                debug=True, name=self.supervisor_name
            )
        return self.compiled_graph


model_name: str = "openai/gpt-4o"
knowledge_agent = create_knowledge_lookup_agent(model_name)
# zendesk_retrieval_agent = create_zendesk_retrieval_agent(model_name)
zendesk_setter_agent = create_zendesk_setter_agent(model_name)

# Create the supervisor model
model = load_chat_model(model_name)
# Define Zendesk setter tools here

# Define a specialized system prompt for Zendesk data setting
system_prompt = """You are a specialized Zendesk data setting agent. 
Your role is to update customer information, ticket status, and other relevant data 
in Zendesk to help resolve customer inquiries. Always ensure you're setting the 
most accurate information and following company policies for data updates. 
Double-check all information before making changes to ensure accuracy.
"""

# Create and return the agent
# zendesk_handler_system = create_react_agent(
#     name="zendesk_handler",
#     model=model,
#     tools=[ZendeskAgentWithTools().get_agent(), zendesk_setter_agent],
#     prompt="You are an agent able to handle Zendesk ticket. You can get information about tickets and comments or set some specific fields.",
#     state_schema=State,
# )
zst = ZendeskAgentWithTools()
zendesk_getter_with_tools = TWSupervisor(
    agents=[],
    model=model,
    tools=zst.get_tools(),
    prompt=zst.system_prompt,
    state_schema=State,
    supervisor_name=zst.node_name,
    description=zst.description,
)
zendesk_setter_with_tools = TWSupervisor(
    agents=[],
    model=model,
    tools=[set_ticket_info, set_ticket_shipping_address],
    prompt=zst.system_prompt,
    state_schema=State,
    supervisor_name="Zendesk Ticket Info Setter",
    description="Agent able to set information in Zendesk about tickets, address, etc.",
)

# Define prompt for account_address_update_case separately
subagents = [zendesk_getter_with_tools, zendesk_setter_with_tools]
account_address_update_prompt = "You are an agent able to call tools to read info from Zendesk tickets about addresses and update them with a Zendesk Ticket setter tool.\nYou can't solve the issue directly, but you can call specialized agents to help you."
account_address_update_prompt += f"\nPossible tools are: \n"
for subagent in subagents:
    account_address_update_prompt += (
        f" - {subagent.supervisor_name}: {subagent.description}\n"
    )

account_address_update_case = TWSupervisor(
    agents=subagents,
    model=model,
    # tools=SUPERVISOR_TOOLS,
    prompt=account_address_update_prompt,
    state_schema=State,
    supervisor_name="account_address_update_case",
    description="Agent able to update address information in the Zendesk ticket.",
)

# Now, create a second supervisor that could potentially be called by the main supervisor
knowledge_handler_system = TWSupervisor(
    agents=[],
    model=model,
    tools=[get_knowledge_info],
    prompt="You are an agent specialized in knowledge information lookup.",
    state_schema=State,
    supervisor_name="knowledge_handler",
    description="Agent able to lookup knowledge information.",
)

# Create bidirectional relationship by passing each supervisor to the other
# Note: We need to be careful about circular dependencies, so we'll use the
# supervisors as they are, and they'll compile their graphs when needed
supervisor_system = TWSupervisor(
    agents=[
        knowledge_handler_system,  # Pass the supervisor directly
        account_address_update_case,  # Pass the supervisor directly
    ],
    model=model,
    prompt=SUPERVISOR_PROMPT,
    state_schema=State,
    supervisor_name="tw_supervisor",
    description="Agent able to handle the flow of the conversation.",
)
# supervisor_system = create_react_agent(
#     name="tw_supervisor",
#     model=model,
#     tools=[knowledge_agent, zendesk_handler_system],
#     prompt=SUPERVISOR_PROMPT,
#     state_schema=State,
# )

# Compile the supervisor system
compiled_supervisor = supervisor_system.get_supervisor_compiled_graph()

graph = compiled_supervisor.get_graph()
png = graph.draw_mermaid_png()
with open("tw_supervisor.png", "wb") as f:
    f.write(png)

account_address_update_case_graph = (
    account_address_update_case.get_supervisor_compiled_graph().get_graph()
)
png = account_address_update_case_graph.draw_mermaid_png()
with open("account_address_update_case.png", "wb") as f:
    f.write(png)

if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": "Hello, how are you?\nI'd like to change the shipping address for my ticket 14983 to Heinrichstrasse 237, Zurich, Switzerland.",
        },
    ]
    state = State(messages=messages)
    result = compiled_supervisor.invoke(state)
    print(result)
