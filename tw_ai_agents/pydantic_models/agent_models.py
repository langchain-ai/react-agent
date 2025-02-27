"""Pydantic models for the agent API.

This module contains the Pydantic models used for request and response validation in the agent API.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class AgentResponseRequest(BaseModel):
    """Request model for the agent response endpoint.
    
    Attributes:
        message_type: Type of message, either 'user' or 'agent'.
        message_text: The content of the message.
        discussion_id: Unique identifier for the discussion.
        client: Client identifier.
    """
    message_type: str = Field(..., description="Type of message, either 'user' or 'agent'")
    message_text: str = Field(..., description="The content of the message")
    discussion_id: str = Field(..., description="Unique identifier for the discussion")
    client: str = Field(..., description="Client identifier")


class AgentResponseModel(BaseModel):
    """Response model for the agent response endpoint.
    
    Attributes:
        message_type: Type of message, either 'user' or 'agent'.
        message_text: The content of the message.
        message_id: Unique identifier for the message.
        metadata: Additional metadata associated with the message.
    """
    message_type: str = Field(..., description="Type of message, either 'user' or 'agent'")
    message_text: str = Field(..., description="The content of the message")
    message_id: str = Field(..., description="Unique identifier for the message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata associated with the message")
