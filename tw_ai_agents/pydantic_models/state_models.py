"""Pydantic models for state management operations."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ResetStateRequest(BaseModel):
    """Request model for resetting the state of a discussion."""
    
    discussion_id: Optional[str] = Field(
        None, 
        description="Optional ID of the discussion to include in the response message"
    )


class StateInfoRequest(BaseModel):
    """Request model for getting information about stored states."""
    
    discussion_id: Optional[str] = Field(
        None, 
        description="Optional discussion ID to filter results"
    )


class StateOperationResponse(BaseModel):
    """Response model for state management operations."""
    
    success: bool = Field(
        ..., 
        description="Whether the operation was successful"
    )
    message: str = Field(
        ..., 
        description="A message describing the result of the operation"
    )
    deleted_file: Optional[str] = Field(
        None, 
        description="Path of the deleted database file"
    )
    file_size_bytes: Optional[int] = Field(
        None, 
        description="Size of the database file in bytes"
    )
    file_exists: Optional[bool] = Field(
        None, 
        description="Whether the database file exists"
    )