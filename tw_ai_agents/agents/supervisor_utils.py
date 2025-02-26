from typing import Callable, Dict, Literal

from langgraph.graph.state import CompiledStateGraph
from langgraph.utils.runnable import RunnableCallable

from tw_ai_agents.supervisor_agent.handoff import create_handoff_back_messages

OutputMode = Literal["full_history", "last_message"]
"""Mode for adding agent outputs to the message history in the multi-agent workflow

- `full_history`: add the entire agent message history
- `last_message`: add only the last message
"""


def _make_call_agent(
    agent: CompiledStateGraph,
    output_mode: OutputMode,
    add_handoff_back_messages: bool,
    supervisor_name: str,
) -> Callable[[Dict], Dict]:
    """Create a function that calls an agent and processes its output.

    Args:
        agent: The agent to call.
        output_mode: How to handle the agent's message history.
        add_handoff_back_messages: Whether to add handoff back messages.
        supervisor_name: The name of the supervisor agent.

    Returns:
        A callable that invokes the agent and processes its output.
    """
    if output_mode not in OutputMode.__args__:  # type: ignore
        raise ValueError(
            f"Invalid agent output mode: {output_mode}. "
            f"Needs to be one of {OutputMode.__args__}"  # type: ignore
        )

    def _process_output(output: Dict) -> Dict:
        messages = output["messages"]
        if output_mode == "full_history":
            pass
        elif output_mode == "last_message":
            messages = messages[-1:]
        else:
            raise ValueError(
                f"Invalid agent output mode: {output_mode}. "
                f"Needs to be one of {OutputMode.__args__}"  # type: ignore
            )

        if add_handoff_back_messages:
            # Add handoff back messages using AIMessage and HumanMessage
            messages.extend(
                create_handoff_back_messages(agent.name, supervisor_name)
            )

        return {
            **output,
            "messages": messages,
        }

    def call_agent(state: Dict) -> Dict:
        output = agent.invoke(state)
        return _process_output(output)

    async def acall_agent(state: Dict) -> Dict:
        output = await agent.ainvoke(state)
        return _process_output(output)

    return RunnableCallable(call_agent, acall_agent)


SUPERVISOR_PROMPT = """
You are an orchestrator AI for a customer service system.

You are given a customer service request. 
Your goal is to determine the best course of action to resolve the request. You will give the request to the appropriate agent to resolve.
If you need more information from the customer to proceed with a flow, you can ask further questions to the customer.
Your role is to:
1. Analyze incoming customer service requests
2. Classify the request into the appropriate category
3. Identify the most suitable flow to handle the request
4. Follow the steps in the flow to resolve the request
5. Delegate specific tasks to specialized agents when needed
6. Maintain conversation context across multiple interactions

## Tools
You have access to the following tools:
{tools}

## Sub Agents
These are the specialized agents you can delegate tasks to when you have determined the appropriate agent for the task:
{agents}

When you receive the customer's response, continue from where you left off.

Remember to be professional, empathetic, and solution-oriented in all interactions.
"""
