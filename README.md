# Customer Service Supervisor Agent System

This project implements a supervisor agent system for customer service, with specialized agents for different tasks such as knowledge lookup, Zendesk data retrieval, and Zendesk data setting. The system is exposed through a RESTful API that maintains conversation state across multiple interactions.

## Overview

The system consists of:

1. **Orchestrator (Supervisor) Agent**: The main agent that receives customer service requests, classifies them into categories, identifies the appropriate flow, and delegates tasks to specialized agents.

2. **Specialized Agents**:
   - **Knowledge Lookup Agent**: Searches the company's knowledge base for information.
   - **Zendesk Retrieval Agent**: Retrieves customer data from Zendesk.
   - **Zendesk Setter Agent**: Updates customer data in Zendesk.

3. **Tools**:
   - `get_request_categories`: Get the list of available request categories.
   - `get_category_flows`: Get the list of flows for a specific category.
   - `get_flow_details`: Get detailed information about a specific flow.

4. **API**:
   - RESTful API for interacting with the supervisor agent system.
   - Maintains conversation state across multiple interactions.
   - Supports pausing and resuming conversations when waiting for user input.

## Architecture

The system uses a supervisor agent pattern, where the orchestrator agent is the main entry point and can delegate tasks to specialized agents. The specialized agents can then return control back to the orchestrator when they've completed their tasks.

```
                  ┌─────────────────┐
                  │   Orchestrator  │
                  │      Agent      │
                  └────────┬────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
    ┌─────────▼────────┐   │    ┌───────▼─────────┐
    │  Knowledge Base  │   │    │  Zendesk Data   │
    │      Agent      │   │    │  Retrieval Agent │
    └─────────────────┘   │    └─────────────────┘
                          │
                  ┌───────▼─────────┐
                  │  Zendesk Data   │
                  │   Setter Agent  │
                  └─────────────────┘
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-service-agent.git
   cd customer-service-agent
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   PORT=8000
   ```

## Usage

### Running the API Server

You can run the API server with:

```bash
python src/server.py
```

This will start the server on port 8000 (or the port specified in your `.env` file).

### API Endpoints

The API provides the following endpoints:

- `POST /message`: Send a message to the supervisor agent system.
  ```json
  {
    "discussion_id": "optional-discussion-id",
    "message": "I need a refund for my recent purchase",
    "user_id": "optional-user-id",
    "metadata": {}
  }
  ```

- `GET /conversation/{discussion_id}`: Get the current state of a conversation.

- `DELETE /conversation/{discussion_id}`: Delete a conversation.

- `GET /conversations`: List all active conversation IDs.

### Example Client

You can use the example client to interact with the API:

```bash
python examples/api_client_demo.py
```

### Programmatic Usage

You can also use the system programmatically:

```python
from langchain_core.messages import HumanMessage
from supervisor_agent.orchestrator import orchestrator

# Create initial state with the customer request
state = {
    "messages": [HumanMessage(content="I need a refund for my recent purchase.")],
    "discussion_id": "unique-discussion-id",
}

# Invoke the orchestrator
result = await orchestrator.ainvoke(state)

# Print the result
for message in result["messages"]:
    print(f"{message.type}: {message.content}")
```

## Project Structure

- `src/supervisor_agent/`: The main package for the supervisor agent system.
  - `__init__.py`: Package initialization.
  - `handoff.py`: Tools for transferring control between agents.
  - `supervisor.py`: Implementation of the supervisor agent.
  - `specialized_agents.py`: Implementation of specialized agents.
  - `tools.py`: Tools for the supervisor agent.
  - `orchestrator.py`: Main orchestrator setup.
  - `api.py`: FastAPI implementation for the API.

- `src/server.py`: Server for running the API.

- `examples/`: Example scripts.
  - `customer_service_demo.py`: Demo script for the customer service system.
  - `api_client_demo.py`: Demo script for the API client.

## Conversation State

The system maintains conversation state across multiple interactions, including:

- **Discussion ID**: Unique identifier for the conversation.
- **Messages**: List of messages in the conversation.
- **Current Category**: The current category of the request.
- **Current Flow**: The current flow being followed.
- **Flow Step**: The current step in the flow.
- **Waiting for User**: Whether the system is waiting for user input.

This allows the system to pick up where it left off when a user responds to a question.

## License

This project is licensed under the MIT License - see the LICENSE file for details.