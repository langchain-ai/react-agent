from typing import List, Optional

from pydantic import BaseModel


class FrontendMessage(BaseModel):
    message_type: str  # user, agent, ai
    message_text: str


class BaseMetadataGeneratorInput(BaseModel):
    discussion_id: str
    client: str
    message_list: List[FrontendMessage]


class BaseMetadataGeneratorOutput(BaseModel):
    metadata: dict


class TitleGeneratorInput(BaseMetadataGeneratorInput):
    current_title: Optional[str] = None


class TitleGeneratorOutput(BaseMetadataGeneratorOutput):
    title: str


class SentimentGeneratorInput(BaseMetadataGeneratorInput):
    current_sentiment: Optional[int] = None


class SentimentGeneratorOutput(BaseMetadataGeneratorOutput):
    sentiment: str


class PriorityGeneratorInput(BaseMetadataGeneratorInput):
    current_priority: Optional[int] = None


class PriorityGeneratorOutput(BaseMetadataGeneratorOutput):
    priority: str
