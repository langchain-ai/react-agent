import asyncio  # Added import for asyncio
import inspect
from typing import Callable, List, Optional, Union, Dict

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage
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

from tw_ai_agents.agents.base_agent import State
from tw_ai_agents.agents.handoff import create_handoff_tool
from tw_ai_agents.agents.supervisor_utils import (
    SUPERVISOR_PROMPT,
    OutputMode,
    _make_call_agent,
)
from tw_ai_agents.agents.utils import load_chat_model
from tw_ai_agents.agents.zendesk_agent_tools import ZendeskAgentWithTools
from tw_ai_agents.tools.tools import (
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

# Create the supervisor model
model = load_chat_model(model_name)

# Define a specialized system prompt for Zendesk data setting
system_prompt = """You are a specialized Zendesk data setting agent. 
Your role is to update customer information, ticket status, and other relevant data 
in Zendesk to help resolve customer inquiries. Always ensure you're setting the 
most accurate information and following company policies for data updates. 
Double-check all information before making changes to ensure accuracy.
"""

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
    supervisor_name="zendesk_info_setter",
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


# Compile the supervisor system
compiled_supervisor = supervisor_system.get_supervisor_compiled_graph()


async def run_supervisor(state: State) -> Dict:
    """Run the supervisor agent system with metadata tracking.
    
    This function executes the supervisor graph and tracks metadata like tool usage
    throughout the execution.
    
    Args:
        state: The initial state containing messages and other context
        
    Returns:
        The updated state with results and metadata about the execution
    """
    try:
        result = await compiled_supervisor.ainvoke(state)
        
        if "metadata" not in result:
            result["metadata"] = {}

        tool_calls = []
        for message in result["messages"]:
            # Check if this is a tool-related message
            if hasattr(message, "additional_kwargs") and message.additional_kwargs:
                # Extract tool calls from OpenAI format
                if "tool_calls" in message.additional_kwargs:
                    for tool_call in message.additional_kwargs["tool_calls"]:
                        tool_calls.append({
                            "tool_name": tool_call.get("function", {}).get("name", "unknown"),
                            "tool_input": tool_call.get("function", {}).get("arguments", "{}"),
                            "tool_id": tool_call.get("id", "unknown")
                        })
            
        result["metadata"]["available_agents"] = available_agents
        result["metadata"]["tool_calls"] = tool_calls
            
        return result
        
    except Exception as e:
        # If there's an error, return a valid state with error information
        return {
            "messages": state.get("messages", []),
            "next": "FINISH",
            "metadata": {
                "error": str(e),
                "status": "error"
            }
        }


if __name__ == "__main__":
    messages = [
        HumanMessage(
            content="Hello, how are you?\nI'd like to change the shipping address for my ticket 14983 to Heinrichstrasse 237, Zurich, Switzerland. Please make sure to double check that this was actually done!"
        ),
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": "Hello, how are you?\nI'd like to change the shipping address for my ticket 14983 to Heinrichstrasse 237, Zurich, Switzerland.",
    #     },
    # ]
    state = State(messages=messages)

    async def main():
        result = await run_supervisor(state)
        print(result)

    # Run the async function using asyncio
    asyncio.run(main())
