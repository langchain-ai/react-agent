import pytest
from langsmith import unit
from langchain_core.messages import HumanMessage

from react_agent import graph
from react_agent.state import InputState


@unit
async def test_react_agent_simple_passthrough() -> None:
    message = HumanMessage(content="Who is the founder of LangChain?")
    input_state = InputState(messages=[message])
    res = await graph.ainvoke(
        input_state,
        {"configurable": {"system_prompt": "You are a helpful AI assistant."}},
    )

    assert "harrison" in str(res["messages"][-1].content).lower()
