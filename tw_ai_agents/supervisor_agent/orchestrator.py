"""Customer service orchestrator agent system.

This module sets up a supervisor agent system for customer service, with specialized agents
for different tasks such as knowledge lookup, Zendesk data retrieval, and Zendesk data setting.
"""

from typing import Any, Dict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from tw_ai_agents.react_agent.helpers import prepare_supervisor_state
from tw_ai_agents.agents.base_agent import State

from tw_ai_agents.react_agent.utils import load_chat_model
from tw_ai_agents.supervisor_agent.specialized_agents import (
    create_knowledge_lookup_agent,
    create_zendesk_retrieval_agent,
    create_zendesk_setter_agent,
)
from tw_ai_agents.supervisor_agent.supervisor import create_supervisor
from tw_ai_agents.supervisor_agent.tools import SUPERVISOR_TOOLS


class OrchestratorSystem:
    """Orchestrator system that wraps the supervisor agent system with conversation state handling."""

    def __init__(self, supervisor_system: Any):
        """Initialize the orchestrator system.

        Args:
            supervisor_system: The supervisor agent system to wrap.
        """
        self.supervisor_system = supervisor_system

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the orchestrator with conversation state handling.

        This method wraps the supervisor system to handle conversation state,
        including tracking the current category, flow, and flow step.

        Args:
            state: The current state, including messages and conversation metadata.

        Returns:
            The updated state with the orchestrator's response.
        """
        # Extract conversation state
        discussion_id = state.get("discussion_id", "")
        current_category = state.get("current_category")
        current_flow = state.get("current_flow")
        flow_step = state.get("flow_step")
        metadata = state.get("metadata", {})

        # Create a state object for the supervisor system
        supervisor_state = prepare_supervisor_state(state["messages"])

        print(f"Supervisor state: {supervisor_state}")
        # Invoke the supervisor system
        result = self.supervisor_system.invoke(supervisor_state, debug=True)
        print(f"Supervisor result: {result}")

        # Extract the response message
        response_message = result["messages"][-1]

        # Analyze the response to update conversation state
        content = response_message.content.lower()

        # Update category if it's been identified
        if current_category is None and any(
            cat in content
            for cat in [
                "billing",
                "technical",
                "account",
                "product",
                "shipping",
            ]
        ):
            for cat in [
                "billing",
                "technical",
                "account",
                "product",
                "shipping",
            ]:
                if cat in content:
                    current_category = cat
                    break

        # Update flow if it's been identified
        if current_category and current_flow is None:
            flow_keywords = {
                "billing": ["refund", "subscription", "payment"],
                "technical": ["troubleshooting", "installation"],
                "account": ["reset", "update"],
                "product": ["information", "compatibility"],
                "shipping": ["tracking", "return"],
            }

            if current_category in flow_keywords:
                for keyword in flow_keywords[current_category]:
                    if keyword in content:
                        current_flow = f"{current_category}_{keyword}"
                        flow_step = 1
                        break

        # Update flow step if we're in a flow
        if current_flow and flow_step is not None:
            # Check if we're waiting for user input
            waiting_for_user = (
                "please provide" in content
                or "could you tell me" in content
                or "i need to know" in content
            )

            # If we're not waiting for user input and we're in a flow, increment the step
            if not waiting_for_user:
                flow_step += 1

            # Return the updated state
            return {
                "messages": result["messages"],
                "discussion_id": discussion_id,
                "current_category": current_category,
                "current_flow": current_flow,
                "flow_step": flow_step,
                "metadata": metadata,
                "waiting_for_user": waiting_for_user,
            }

        # If we're not in a flow yet, just return the response
        return {
            "messages": result["messages"],
            "discussion_id": discussion_id,
            "current_category": current_category,
            "current_flow": current_flow,
            "flow_step": flow_step,
            "metadata": metadata,
            "waiting_for_user": False,
        }


def create_orchestrator_system(
    model_name: str = "openai/gpt-4o",
) -> OrchestratorSystem:
    """Create the customer service orchestrator system.

    This function sets up a supervisor agent system for customer service, with specialized agents
    for different tasks such as knowledge lookup, Zendesk data retrieval, and Zendesk data setting.

    Args:
        model_name: The name of the language model to use for the supervisor and specialized agents.

    Returns:
        A compiled supervisor agent system.
    """
    # Create specialized agents
    knowledge_agent = create_knowledge_lookup_agent(model_name)
    zendesk_retrieval_agent = create_zendesk_retrieval_agent(model_name)
    zendesk_setter_agent = create_zendesk_setter_agent(model_name)

    # Create the supervisor model
    model = load_chat_model(model_name)

    # Define the supervisor system prompt
    supervisor_prompt = """You are an orchestrator AI for a customer service system.

Your role is to:
1. Analyze incoming customer service requests
2. Classify the request into the appropriate category
3. Identify the most suitable flow to handle the request
4. Follow the steps in the flow to resolve the request
5. Delegate specific tasks to specialized agents when needed
6. Maintain conversation context across multiple interactions

You have access to the following specialized agents:
- knowledge_lookup: For searching the company's knowledge base
- zendesk_retrieval: For retrieving customer data from Zendesk
- zendesk_setter: For updating customer data in Zendesk

You also have tools to:
- get_request_categories: Get the list of available request categories
- get_category_flows: Get the list of flows for a specific category
- get_flow_details: Get detailed information about a specific flow

Always follow this process:
1. Understand the customer's request
2. Use get_request_categories to identify the appropriate category
3. Use get_category_flows to find available flows for that category
4. Select the most appropriate flow based on the customer's specific needs
5. Use get_flow_details to get detailed instructions for the selected flow
6. Follow the flow steps, delegating to specialized agents when needed
7. Provide a clear, helpful response to the customer

If you need more information from the customer to proceed with a flow, ask for it and set the "waiting_for_user" flag to true.
When you receive the customer's response, continue from where you left off.

Remember to be professional, empathetic, and solution-oriented in all interactions.
"""

    # Create the supervisor agent system
    supervisor_system = create_supervisor(
        agents=[knowledge_agent, zendesk_retrieval_agent, zendesk_setter_agent],
        model=model,
        tools=SUPERVISOR_TOOLS,
        prompt=supervisor_prompt,
        state_schema=State,
        output_mode="full_history",
        add_handoff_back_messages=True,
        supervisor_name="orchestrator",
    )

    # Compile the supervisor system before wrapping it
    # checkpointer = InMemorySaver()
    # store = InMemoryStore()
    compiled_supervisor = supervisor_system.compile()
    # from IPython.display import Image, display
    # from langchain_core.runnables.graph import MermaidDrawMethod

    # display(
    #     Image(
    #         compiled_supervisor.get_graph().draw_mermaid_png(
    #             draw_method=MermaidDrawMethod.API,
    #         )
    #     )
    # )

    # Wrap the compiled supervisor system in our orchestrator class
    return OrchestratorSystem(compiled_supervisor)


# Create the orchestrator system
# orchestrator = create_orchestrator_system()
# orchestrator_graph = orchestrator.supervisor_system
