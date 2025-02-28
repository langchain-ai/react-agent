"""State management operations for TW AI Agents.

This module provides functions to manage the state of agent conversations
stored in the SQLite database.
"""
import os
from typing import Optional, Dict, Any

from tw_ai_agents.config.constants import DB_CHECKPOINT_PATH


async def reset_state(discussion_id: Optional[str] = None) -> Dict[str, Any]:
    """Reset the state by deleting the database file.
    
    Args:
        discussion_id: Optional ID of the discussion to include in the response message.
                     This parameter doesn't affect which records are deleted since
                     the entire database file is removed.
        
    Returns:
        Dict[str, Any]: Result of the operation with success status and details
    """
    try:
        # Check if the file exists
        if not os.path.exists(DB_CHECKPOINT_PATH):
            return {
                "success": False,
                "message": f"Database file not found: {DB_CHECKPOINT_PATH}",
            }
        
        # Get file size before deletion for reporting
        file_size = os.path.getsize(DB_CHECKPOINT_PATH)
        
        # Delete the database file
        os.remove(DB_CHECKPOINT_PATH)
        
        message = "Successfully reset state by deleting the database file"
        if discussion_id:
            message += f" (requested for discussion ID: {discussion_id})"
        
        return {
            "success": True,
            "message": message,
            "deleted_file": DB_CHECKPOINT_PATH,
            "file_size_bytes": file_size
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error deleting database file: {str(e)}",
            "error": str(e)
        }