from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel


class ToolMessageInfo(BaseModel):
    name: str
    content: str
    tool_call_id: str
    id: str
    parameters: Optional[Dict[str, Any]] = None

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.content == other.content
            and self.tool_call_id == other.tool_call_id
            and self.id == other.id
            and self.parameters == other.parameters
        )


def custom_add(a, b):
    for b_item in b:
        if b_item not in a:
            a.append(b_item)

    return a


class State(MessagesState):
    """State for the agent system, extending MessagesState with metadata for tool tracking."""

    next: str
    remaining_steps: int
    metadata: Dict[str, Any]
    tools_called: Annotated[List[ToolMessageInfo], custom_add] = []


class SubagentState(State):
    """State for the agent system, extending MessagesState with metadata for tool tracking."""

    message_for_subagent: str
