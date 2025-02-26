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


class TWSupervisor:
    def __init__(
        self,
        agents: List[CompiledStateGraph],
        model: LanguageModelLike,
        tools: Optional[List[Union[Callable, BaseTool]]] = None,
        prompt: Optional[Prompt] = None,
        state_schema: StateSchemaType = AgentState,
        output_mode: OutputMode = "last_message",
        add_handoff_back_messages: bool = True,
        supervisor_name: str = "supervisor",
    ):
        """

        Args:
            agents: List of agents to manage
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
        self.prompt = SUPERVISOR_PROMPT
        self.state_schema = state_schema
        self.output_mode = output_mode
        self.add_handoff_back_messages = add_handoff_back_messages
        self.supervisor_name = supervisor_name

    def _create_supervisor(
        self,
    ) -> StateGraph:
        """Create a multi-agent supervisor.

        Returns:
            A StateGraph representing the supervisor agent system.
        """
        agent_names = set()
        for agent in self.agents:
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
            create_handoff_tool(agent_name=agent.name) for agent in self.agents
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
        for agent in self.agents:
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
        return self._create_supervisor().compile(debug=True)


model_name: str = "openai/gpt-4o"
knowledge_agent = create_knowledge_lookup_agent(model_name)
zendesk_retrieval_agent = create_zendesk_retrieval_agent(model_name)
zendesk_setter_agent = create_zendesk_setter_agent(model_name)

# Create the supervisor model
model = load_chat_model(model_name)
supervisor_system = TWSupervisor(
    agents=[knowledge_agent, zendesk_retrieval_agent, zendesk_setter_agent],
    model=model,
    # tools=SUPERVISOR_TOOLS,
    state_schema=State,
    supervisor_name="tw_supervisor",
)

# Compile the supervisor system before wrapping it
# checkpointer = InMemorySaver()
# store = InMemoryStore()
compiled_supervisor = supervisor_system.get_supervisor_compiled_graph()


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    state = State(messages=messages)
    result = compiled_supervisor.invoke(state)
    print(result)
