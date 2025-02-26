"""Test script for the orchestrator.

This script tests the orchestrator to ensure it works correctly.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from supervisor_agent.orchestrator import create_orchestrator_system

# Load environment variables
load_dotenv()

def test_orchestrator():
    """Test the orchestrator system with a simple query."""
    print("Testing orchestrator...")
    
    # Create the orchestrator
    orchestrator = create_orchestrator_system()
    
    # Initial state with a human message
    initial_state = {
        "discussion_id": "test-123",
        "messages": [
            HumanMessage(content="I'm having trouble with my account. Can you help me?")
        ],
        "metadata": {}
    }
    
    try:
        # Call the orchestrator
        response = orchestrator.invoke(initial_state)
        
        # Print the response
        print("\nOrchestrator response:")
        for msg in response["messages"]:
            print(f"{msg.type}: {msg.content}")
            
        print("\nConversation state:")
        print(f"  Discussion ID: {response['discussion_id']}")
        print(f"  Category: {response['current_category']}")
        print(f"  Flow: {response['current_flow']}")
        print(f"  Flow Step: {response['flow_step']}")
        print(f"  Waiting for User: {response['waiting_for_user']}")
        
        print("\nTest completed successfully!")
        return True
    except Exception as e:
        print(f"\nError testing orchestrator: {e}")
        return False

if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        exit(1)
    
    # Run the test
    success = test_orchestrator()
    
    # Exit with appropriate code
    exit(0 if success else 1) 