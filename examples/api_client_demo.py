"""Demo script for the customer service supervisor agent API.

This script demonstrates how to use the customer service supervisor agent API.
The API uses OpenAI's GPT-4o model for generating responses.
"""

import asyncio
import json
import httpx


async def main():
    """Run the API client demo."""
    print("Customer Service Supervisor Agent API Demo")
    print("=========================================")
    print("This demo shows how to use the API to interact with the supervisor agent system.")
    print("The API maintains conversation state across multiple interactions.")
    print("The system uses OpenAI's GPT-4o model for generating responses.")
    print()
    
    # Example customer service requests
    requests = [
        "I need a refund for my recent purchase. The product doesn't work as advertised.",
        "My order number is ABC123 and I purchased it on May 15th.",
        "Yes, I've already tried resetting the device as suggested in the manual.",
        "Thank you for processing my refund. When can I expect to see it in my account?",
    ]
    
    # API endpoint
    api_url = "http://localhost:8000"
    
    # Initialize discussion ID
    discussion_id = None
    
    async with httpx.AsyncClient() as client:
        for i, request in enumerate(requests):
            print(f"User: {request}")
            print("-" * 80)
            
            # Prepare request payload
            payload = {
                "message": request,
            }
            
            # Add discussion ID if we have one
            if discussion_id:
                payload["discussion_id"] = discussion_id
            
            # Send request to API
            response = await client.post(f"{api_url}/message", json=payload)
            
            # Check if request was successful
            if response.status_code == 200:
                # Parse response
                result = response.json()
                
                # Save discussion ID for next request
                discussion_id = result["discussion_id"]
                
                # Print response
                print(f"Assistant: {result['response']}")
                print()
                print(f"Conversation State:")
                print(f"  Discussion ID: {result['conversation_state']['discussion_id']}")
                print(f"  Category: {result['conversation_state']['current_category']}")
                print(f"  Flow: {result['conversation_state']['current_flow']}")
                print(f"  Flow Step: {result['conversation_state']['flow_step']}")
                print(f"  Waiting for User: {result['conversation_state']['waiting_for_user']}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
            
            print("\n" + "=" * 80 + "\n")
    
    # Get conversation history
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/conversation/{discussion_id}")
        
        if response.status_code == 200:
            conversation = response.json()
            
            print("Conversation History:")
            for msg in conversation["messages"]:
                print(f"{msg['role'].capitalize()}: {msg['content']}")
        else:
            print(f"Error retrieving conversation history: {response.status_code} - {response.text}")


if __name__ == "__main__":
    asyncio.run(main()) 