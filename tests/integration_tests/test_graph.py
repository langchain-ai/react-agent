import pytest

from react_agent import graph
from react_agent.context import Context

pytestmark = pytest.mark.anyio


async def test_react_agent_simple_passthrough() -> None:
    res = await graph.ainvoke(
        {"messages": [("user", "Who is the founder of LangChain?")]},  # type: ignore
        context=Context(system_prompt="You are a helpful AI assistant."),
    )

    assert "harrison" in str(res["messages"][-1].content).lower()
