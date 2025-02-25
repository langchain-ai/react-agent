"""Specialized agents for the customer service system.

This module defines specialized agents for different tasks in the customer service system,
such as knowledge lookup, Zendesk data retrieval, and Zendesk data setting.
"""

from typing import Dict, List, Optional, Any

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.chat_agent_executor import create_react_agent

from react_agent.state import State
from react_agent.utils import load_chat_model


def create_knowledge_lookup_agent(
    model_name: str = "anthropic/claude-3-5-sonnet-20240620",
) -> Any:
    """Create an agent specialized in knowledge base lookup.
    
    Args:
        model_name: The name of the language model to use.
        
    Returns:
        A compiled agent graph for knowledge lookup.
    """
    model = load_chat_model(model_name)
    
    # Define knowledge lookup tools here
    tools = []
    
    # Define a specialized system prompt for knowledge lookup
    system_prompt = """You are a specialized knowledge lookup agent. 
    Your role is to search through the company's knowledge base to find accurate information 
    that can help resolve customer inquiries. Always provide the most up-to-date and relevant 
    information available in the knowledge base. If you cannot find the information requested, 
    clearly state that and suggest alternative sources or approaches.
    """
    
    # Create and return the agent
    agent = create_react_agent(
        name="knowledge_lookup",
        model=model,
        tools=tools,
        prompt=system_prompt,
        state_schema=State,
    )
    
    return agent


def create_zendesk_retrieval_agent(
    model_name: str = "anthropic/claude-3-5-sonnet-20240620",
) -> Any:
    """Create an agent specialized in Zendesk data retrieval.
    
    Args:
        model_name: The name of the language model to use.
        
    Returns:
        A compiled agent graph for Zendesk data retrieval.
    """
    model = load_chat_model(model_name)
    
    # Define Zendesk retrieval tools here
    tools = []
    
    # Define a specialized system prompt for Zendesk data retrieval
    system_prompt = """You are a specialized Zendesk data retrieval agent. 
    Your role is to retrieve customer information, ticket history, and other relevant data 
    from Zendesk to help resolve customer inquiries. Always ensure you're retrieving the 
    most accurate and up-to-date information. Respect customer privacy and only retrieve 
    information that is necessary for resolving the current inquiry.
    """
    
    # Create and return the agent
    agent = create_react_agent(
        name="zendesk_retrieval",
        model=model,
        tools=tools,
        prompt=system_prompt,
        state_schema=State,
    )
    
    return agent


def create_zendesk_setter_agent(
    model_name: str = "anthropic/claude-3-5-sonnet-20240620",
) -> Any:
    """Create an agent specialized in setting Zendesk data.
    
    Args:
        model_name: The name of the language model to use.
        
    Returns:
        A compiled agent graph for Zendesk data setting.
    """
    model = load_chat_model(model_name)
    
    # Define Zendesk setter tools here
    tools = []
    
    # Define a specialized system prompt for Zendesk data setting
    system_prompt = """You are a specialized Zendesk data setting agent. 
    Your role is to update customer information, ticket status, and other relevant data 
    in Zendesk to help resolve customer inquiries. Always ensure you're setting the 
    most accurate information and following company policies for data updates. 
    Double-check all information before making changes to ensure accuracy.
    """
    
    # Create and return the agent
    agent = create_react_agent(
        name="zendesk_setter",
        model=model,
        tools=tools,
        prompt=system_prompt,
        state_schema=State,
    )
    
    return agent 