"""Customer service orchestrator agent system.

This module sets up a supervisor agent system for customer service, with specialized agents
for different tasks such as knowledge lookup, Zendesk data retrieval, and Zendesk data setting.
"""

from typing import Dict, List, Optional, Any

from react_agent.state import State
from react_agent.utils import load_chat_model
from supervisor_agent.specialized_agents import (
    create_knowledge_lookup_agent,
    create_zendesk_retrieval_agent,
    create_zendesk_setter_agent,
)
from supervisor_agent.supervisor import create_supervisor
from supervisor_agent.tools import SUPERVISOR_TOOLS


def create_orchestrator_system(
    model_name: str = "anthropic/claude-3-5-sonnet-20240620",
) -> Any:
    """Create the customer service orchestrator system.
    
    This function sets up a supervisor agent system for customer service, with specialized agents
    for different tasks such as knowledge lookup, Zendesk data retrieval, and Zendesk data setting.
    
    Args:
        model_name: The name of the language model to use for the supervisor and specialized agents.
        
    Returns:
        A compiled supervisor agent system.
    """
    # Create specialized agents
    knowledge_agent = create_knowledge_lookup_agent(model_name)
    zendesk_retrieval_agent = create_zendesk_retrieval_agent(model_name)
    zendesk_setter_agent = create_zendesk_setter_agent(model_name)
    
    # Create the supervisor model
    model = load_chat_model(model_name)
    
    # Define the supervisor system prompt
    supervisor_prompt = """You are an orchestrator AI for a customer service system.

Your role is to:
1. Analyze incoming customer service requests
2. Classify the request into the appropriate category
3. Identify the most suitable flow to handle the request
4. Follow the steps in the flow to resolve the request
5. Delegate specific tasks to specialized agents when needed

You have access to the following specialized agents:
- knowledge_lookup: For searching the company's knowledge base
- zendesk_retrieval: For retrieving customer data from Zendesk
- zendesk_setter: For updating customer data in Zendesk

You also have tools to:
- get_request_categories: Get the list of available request categories
- get_category_flows: Get the list of flows for a specific category
- get_flow_details: Get detailed information about a specific flow

Always follow this process:
1. Understand the customer's request
2. Use get_request_categories to identify the appropriate category
3. Use get_category_flows to find available flows for that category
4. Select the most appropriate flow based on the customer's specific needs
5. Use get_flow_details to get detailed instructions for the selected flow
6. Follow the flow steps, delegating to specialized agents when needed
7. Provide a clear, helpful response to the customer

Remember to be professional, empathetic, and solution-oriented in all interactions.
"""
    
    # Create the supervisor agent system
    supervisor_system = create_supervisor(
        agents=[knowledge_agent, zendesk_retrieval_agent, zendesk_setter_agent],
        model=model,
        tools=SUPERVISOR_TOOLS,
        prompt=supervisor_prompt,
        state_schema=State,
        output_mode="last_message",
        add_handoff_back_messages=True,
        supervisor_name="orchestrator",
    )
    
    return supervisor_system


# Create the orchestrator system
orchestrator = create_orchestrator_system() 