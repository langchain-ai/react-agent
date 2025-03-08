from collections import defaultdict
from enum import Enum
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph import MessagesState, add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field


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


def custom_dict_add(a, b):
    for key, value in b.items():
        a[key] = value
    return a


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


class State(BaseModel):
    """State for the agent system, extending MessagesState with metadata for tool tracking."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep = Field(
        default_factory=IsLastStep, description="Required field"
    )
    remaining_steps: RemainingSteps = Field(
        default_factory=RemainingSteps, description="Required field"
    )
    metadata: Annotated[Dict[str, Any], custom_dict_add] = Field(
        default_factory=dict,
        description="Metadata field. Expecially needed to store the initial router destination",
    )
    tools_called: Annotated[List[ToolMessageInfo], custom_add] = Field(
        default_factory=list, description="List of tools called by the agent"
    )
    message_from_supervisor: Annotated[
        List[Union[str, None]], custom_add_with_None
    ] = Field(
        default_factory=list,
        description="List of messages from an agent to a sub-agent, on any level.",
    )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def add_messages_to_from_user(self, messages: List[AnyMessage]):
        self.messages_to_from_user = messages


class AgentMessageMode(str, Enum):
    COMPLETE_HANDOFF = "complete_handoff"
    QUESTION = "question"
    CONFIRMATION = "confirmation"
    ACTION_REQUEST = "action_request"


class InterruptBaseModel(BaseModel):
    user_message: str
    agent_message_mode: AgentMessageMode
    tools_called: List[ToolMessageInfo] = []
    destination: Literal["agent", "user"] = "agent"
