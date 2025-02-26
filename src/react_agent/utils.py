"""Utility functions for the react agent.

This module provides utility functions for the react agent, such as loading chat models.
"""

from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    """Load a chat model based on the model name.
    
    Args:
        model_name: The name of the model to load, in the format "provider/model_name".
        **kwargs: Additional keyword arguments to pass to the model constructor.
        
    Returns:
        A chat model instance.
        
    Raises:
        ValueError: If the model provider is not supported.
    """
    # Split the model name into provider and model
    parts = model_name.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid model name format: {model_name}. Expected format: 'provider/model_name'")
    
    provider, model = parts
    
    # Load the appropriate model based on the provider
    if provider.lower() == "openai":
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Create OpenAI chat model
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1000),
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
