"""Demo script for the customer service supervisor agent system.

This script demonstrates how to use the supervisor agent system for customer service.
"""

import asyncio
from typing import Dict, List, Any

from langchain_core.messages import HumanMessage

from supervisor_agent.orchestrator import orchestrator


async def main():
    """Run the customer service demo."""
    print("Customer Service Supervisor Agent Demo")
    print("=====================================")
    print("This demo shows how the supervisor agent system handles customer service requests.")
    print("The system will classify the request, identify the appropriate flow, and follow")
    print("the steps in the flow to resolve the request, delegating to specialized agents when needed.")
    print()
    
    # Example customer service requests
    requests = [
        "I need a refund for my recent purchase. The product doesn't work as advertised.",
        "I'm having trouble logging into my account. I've tried resetting my password but I'm not receiving the reset email.",
        "Can you tell me if your software is compatible with macOS Monterey?",
        "I need to update my shipping address for my recent order #12345.",
    ]
    
    for i, request in enumerate(requests):
        print(f"Example {i+1}: {request}")
        print("-" * 80)
        
        # Create initial state with the customer request
        state = {
            "messages": [HumanMessage(content=request)],
        }
        
        # Invoke the orchestrator
        result = await orchestrator.ainvoke(state)
        
        # Print the result
        print("Response:")
        for message in result["messages"]:
            print(f"{message.type}: {message.content}")
        
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main()) 