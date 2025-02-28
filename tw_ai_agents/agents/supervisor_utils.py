from typing import Callable, Dict, Literal, Tuple, Optional, List

from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.utils.runnable import RunnableCallable

from tw_ai_agents.agents.handoff import create_handoff_back_messages

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
    input_mode: OutputMode = "last_message",
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

    def _process_output(
        output: Dict, old_messages: Optional[Dict] = None
    ) -> Dict:
        messages = output["messages"]
        if output_mode == "full_history":
            messages = old_messages + messages
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

    def _process_input(input: Dict) -> Tuple[Dict, Optional[List[BaseMessage]]]:
        if input_mode == "last_message":
            # return on the last ToolMessage, convert it to a HumanMessage
            last_message = input["messages"][-1]
            other_messages = input["messages"][:-1]
            if isinstance(last_message, ToolMessage):
                last_message = HumanMessage(last_message.content)
            input["messages"] = [last_message]
            return input, other_messages
        elif input_mode == "full_history":
            return input, None
        else:
            raise ValueError(f"Invalid input mode: {input_mode}")

    def call_agent(state: Dict) -> Dict:
        state, old_messages = _process_input(state)
        output = agent.invoke(state)
        return _process_output(output, old_messages)

    async def acall_agent(state: Dict) -> Dict:
        state, old_messages = _process_input(state)
        output = await agent.ainvoke(state)
        return _process_output(output, old_messages)

    return RunnableCallable(call_agent, acall_agent)
