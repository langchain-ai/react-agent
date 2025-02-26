from typing import (
    Callable,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

from langchain_core.language_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    HumanMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Command
from typing_extensions import TypedDict


class State(MessagesState):
    next: str

class BaseAgent:
    @classmethod
    def get_compiled_graph(cls):
        pass

    def get_node(
        self,
        llm,
        tool_list: List[Union[ToolExecutor, Sequence[BaseTool], ToolNode]],
        name: str,
        target_node_name: str,
    ) -> Callable[[State], Command]:
        agent = create_react_agent(llm, tools=tool_list)
        return_type = Literal[target_node_name]

        def node(state: State) -> Command[return_type]:
            result = agent.invoke(state)
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=result["messages"][-1].content, name=name
                        )
                    ]
                },
                # We want our workers to ALWAYS "report back" to the supervisor when done
                goto=target_node_name,
            )

        return node


def make_supervisor_node(
    llm: BaseChatModel, members: list[str], system_prompt: str
):
    options = ["FINISH"] + members

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: str  # Will be one of the options

    def supervisor_node(state: State) -> Command[str]:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node
