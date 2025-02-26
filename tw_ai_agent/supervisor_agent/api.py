"""API for the customer service supervisor agent system.

This module provides a FastAPI implementation for interacting with the supervisor agent system.
It handles incoming messages, maintains conversation state, and generates responses using
OpenAI's GPT-4o model.
"""

from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from tw_ai_agent.react_agent.helpers import prepare_supervisor_state
from tw_ai_agent.supervisor_agent.orchestrator import orchestrator


# Models for API requests and responses
class Message(BaseModel):
    """A message in a conversation."""
    
    role: str = Field(..., description="The role of the message sender (user or assistant)")
    content: str = Field(..., description="The content of the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="When the message was sent")


class ConversationState(BaseModel):
    """The state of a conversation."""
    
    discussion_id: str = Field(..., description="Unique identifier for the conversation")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the conversation")
    current_category: Optional[str] = Field(None, description="The current category of the request")
    current_flow: Optional[str] = Field(None, description="The current flow being followed")
    flow_step: Optional[int] = Field(None, description="The current step in the flow")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the conversation")
    waiting_for_user: bool = Field(False, description="Whether the system is waiting for user input")
    active: bool = Field(True, description="Whether the conversation is active")


class MessageRequest(BaseModel):
    """Request model for sending a message."""
    
    discussion_id: Optional[str] = Field(None, description="Unique identifier for the conversation. If not provided, a new conversation will be created.")
    message: str = Field(..., description="The message content from the user")
    user_id: Optional[str] = Field(None, description="Identifier for the user")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the message or user")


class MessageResponse(BaseModel):
    """Response model for a message."""
    
    discussion_id: str = Field(..., description="Unique identifier for the conversation")
    response: str = Field(..., description="The response message from the assistant")
    waiting_for_user: bool = Field(False, description="Whether the system is waiting for user input")
    conversation_state: ConversationState = Field(..., description="The current state of the conversation")


# In-memory store for conversation states
# In a production environment, this would be replaced with a database
conversation_states: Dict[str, ConversationState] = {}


# Create FastAPI app
app = FastAPI(
    title="Customer Service Supervisor Agent API",
    description="API for interacting with the customer service supervisor agent system",
    version="1.0.0",
)


@app.post("/message", response_model=MessageResponse)
async def process_message(request: MessageRequest, background_tasks: BackgroundTasks) -> MessageResponse:
    """Process an incoming message from a user.
    
    Args:
        request: The message request containing the user's message and optional discussion ID.
        background_tasks: FastAPI background tasks for asynchronous processing.
        
    Returns:
        A response containing the assistant's message and updated conversation state.
    """
    # Get or create discussion ID
    discussion_id = request.discussion_id or str(uuid.uuid4())
    
    # Get or create conversation state
    if discussion_id not in conversation_states:
        conversation_states[discussion_id] = ConversationState(
            discussion_id=discussion_id,
            messages=[],
            metadata=request.metadata or {},
        )
    
    conversation_state = conversation_states[discussion_id]
    
    # Add user message to conversation
    conversation_state.messages.append(
        Message(
            role="user",
            content=request.message,
            timestamp=datetime.now(),
        )
    )
    
    # Update waiting state
    conversation_state.waiting_for_user = False
    
    # Prepare messages for the orchestrator
    langchain_messages = []
    
    # Add system message if this is a new conversation
    if len(conversation_state.messages) == 1:
        langchain_messages.append(SystemMessage(content="You are a helpful customer service assistant powered by OpenAI's GPT-4o model."))
    
    # Add conversation history
    for msg in conversation_state.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))
    
    # Create state for the orchestrator using the helper function
    orchestrator_state = prepare_supervisor_state(langchain_messages)
    
    # Add additional state information
    orchestrator_state.update({
        "discussion_id": discussion_id,
        "metadata": conversation_state.metadata,
        "current_category": conversation_state.current_category,
        "current_flow": conversation_state.current_flow,
        "flow_step": conversation_state.flow_step,
    })
    
    # Invoke the orchestrator
    try:
        result = await orchestrator.ainvoke(orchestrator_state)
        
        # Extract the response message
        response_message = result["messages"][-1]
        response_content = response_message.content
        
        # Update conversation state with orchestrator results
        conversation_state.messages.append(
            Message(
                role="assistant",
                content=response_content,
                timestamp=datetime.now(),
            )
        )
        
        # Update conversation state with any changes from the orchestrator
        if "current_category" in result:
            conversation_state.current_category = result["current_category"]
        if "current_flow" in result:
            conversation_state.current_flow = result["current_flow"]
        if "flow_step" in result:
            conversation_state.flow_step = result["flow_step"]
        if "waiting_for_user" in result:
            conversation_state.waiting_for_user = result["waiting_for_user"]
        if "metadata" in result:
            conversation_state.metadata.update(result["metadata"])
        
        # Save updated conversation state
        conversation_states[discussion_id] = conversation_state
        
        # Return response
        return MessageResponse(
            discussion_id=discussion_id,
            response=response_content,
            waiting_for_user=conversation_state.waiting_for_user,
            conversation_state=conversation_state,
        )
    
    except Exception as e:
        # Log the error
        print(f"Error processing message: {str(e)}")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}",
        )


@app.get("/conversation/{discussion_id}", response_model=ConversationState)
async def get_conversation(discussion_id: str) -> ConversationState:
    """Get the current state of a conversation.
    
    Args:
        discussion_id: The unique identifier for the conversation.
        
    Returns:
        The current state of the conversation.
    """
    if discussion_id not in conversation_states:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {discussion_id} not found",
        )
    
    return conversation_states[discussion_id]


@app.delete("/conversation/{discussion_id}")
async def delete_conversation(discussion_id: str) -> Dict[str, str]:
    """Delete a conversation.
    
    Args:
        discussion_id: The unique identifier for the conversation.
        
    Returns:
        A confirmation message.
    """
    if discussion_id not in conversation_states:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation with ID {discussion_id} not found",
        )
    
    del conversation_states[discussion_id]
    
    return {"message": f"Conversation with ID {discussion_id} deleted"}


@app.get("/conversations", response_model=List[str])
async def list_conversations() -> List[str]:
    """List all active conversation IDs.
    
    Returns:
        A list of conversation IDs.
    """
    return list(conversation_states.keys()) 