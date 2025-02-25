# Customer Service Supervisor Agent System

This project implements a supervisor agent system for customer service, with specialized agents for different tasks such as knowledge lookup, Zendesk data retrieval, and Zendesk data setting.

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

## Usage

You can use the system by importing the orchestrator and invoking it with a state containing messages:

```python
from langchain_core.messages import HumanMessage
from supervisor_agent.orchestrator import orchestrator

# Create initial state with the customer request
state = {
    "messages": [HumanMessage(content="I need a refund for my recent purchase.")],
}

# Invoke the orchestrator
result = orchestrator.invoke(state)

# Print the result
for message in result["messages"]:
    print(f"{message.type}: {message.content}")
```

## Demo

You can run the demo script to see the system in action:

```bash
python examples/customer_service_demo.py
```

## Project Structure

- `src/supervisor_agent/`: The main package for the supervisor agent system.
  - `__init__.py`: Package initialization.
  - `handoff.py`: Tools for transferring control between agents.
  - `supervisor.py`: Implementation of the supervisor agent.
  - `specialized_agents.py`: Implementation of specialized agents.
  - `tools.py`: Tools for the supervisor agent.
  - `orchestrator.py`: Main orchestrator setup.

- `examples/`: Example scripts.
  - `customer_service_demo.py`: Demo script for the customer service system.

## License

This project is licensed under the MIT License - see the LICENSE file for details.