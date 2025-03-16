"""Module for defining the agent's workflow graph and human interaction nodes."""

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt

from agent.state import State


async def human_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Call the interrupt function to pause the graph and handle user interaction.

    Once resumed, it will log the type of action which was returned from
    the interrupt function.
    """
    # Define the interrupt request
    action_request = ActionRequest(
        action="Confirm Joke",
        args={"joke": "What did the engineer say to the manager?"},
    )

    interrupt_config = HumanInterruptConfig(
        allow_ignore=True,  # Allow the user to `ignore` the interrupt.
        allow_respond=True,  # Allow the user to `respond` to the interrupt.
        allow_edit=True,  # Allow the user to `edit` the interrupt's args.
        allow_accept=True,  # Allow the user to `accept` the interrupt's args.
    )

    # The description will be rendered as markdown in the UI, so you may use markdown syntax.
    description = (
        "# Confirm Joke\n"
        + "Please carefully example the joke, and decide if you want to accept, edit, or ignore the joke."
        + "If you accept, the joke will be added to the conversation."
        + "If you edit and submit, the edited joke will be added to the conversation."
        + "If you ignore, the conversation will continue without adding the joke."
        + "If you respond, the response will be used to generate a new joke."
    )

    request = HumanInterrupt(
        action_request=action_request, config=interrupt_config, description=description
    )

    human_response: HumanResponse = interrupt([request])[0]

    if human_response.get("type") == "response":
        message = f"User responded with: {human_response.get('args')}"
        return {"interrupt_response": message}
    elif human_response.get("type") == "accept":
        message = f"User accepted with: {human_response.get('args')}"
        return {"interrupt_response": message}
    elif human_response.get("type") == "edit":
        message = f"User edited with: {human_response.get('args')}"
        return {"interrupt_response": message}
    elif human_response.get("type") == "ignore":
        message = "User ignored interrupt."
        return {"interrupt_response": message}

    return {
        "interrupt_response": "Unknown interrupt response type: " + str(human_response)
    }


# Define a new graph
workflow = StateGraph(State)

# Add the node to the graph. This node will interrupt when it is invoked.
workflow.add_node("human_node", human_node)

# Set the entrypoint as `human_node` so the first node will interrupt
workflow.add_edge("__start__", "human_node")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "Agent Inbox Example"  # This defines the custom name in LangSmith