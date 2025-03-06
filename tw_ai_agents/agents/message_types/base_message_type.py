from typing import Annotated, Any, Dict, List, Optional, Sequence, Union

from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph import MessagesState, add_messages
from langgraph.managed import IsLastStep, RemainingSteps
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


def custom_add_with_None(a, b):
    """
    Append to list, if the content to Append is None, then drop the last element of the list
    """
    for b_item in b:
        if b_item not in a:
            a.append(b_item)
        elif b_item is None:
            a.pop()
    return a


class StateTD(MessagesState):
    """State for the agent system, extending MessagesState with metadata for tool tracking."""

    next: str
    remaining_steps: int
    metadata: Dict[str, Any]
    tools_called: Annotated[List[ToolMessageInfo], custom_add] = []
    messages_to_from_user: Annotated[list[AnyMessage], add_messages]


class State(BaseModel):
    """State for the agent system, extending MessagesState with metadata for tool tracking."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    # next: str
    metadata: Dict[str, Any]
    tools_called: Annotated[List[ToolMessageInfo], custom_add] = []
    messages_to_from_user: Annotated[list[AnyMessage], add_messages]
    message_from_supervisor: Annotated[
        List[Union[str, None]], custom_add_with_None
    ] = []

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def add_messages_to_from_user(self, messages: List[AnyMessage]):
        self.messages_to_from_user = messages


class InterruptBaseModel(BaseModel):
    user_message: str
    tools_called: List[ToolMessageInfo]
