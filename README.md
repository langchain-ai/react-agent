# LangGraph ReAct Agent Template

This LangGraph template implements a simple, extensible ReAct agent.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

## Repo Structure

```txt
├── LICENSE
├── README.md
├── langgraph.json
├── poetry.lock
├── pyproject.toml
├── react_agent
│   ├── __init__.py
│   ├── graph.py
│   └── utils
│       ├── __init__.py
│       ├── configuration.py # Define the configurable variables
│       ├── state.py # Define state variables and how they're updated
│       ├── tools.py # Define the tools your agent can access
│       └── utils.py # Other sundry utilities
└── tests # Add whatever tests you'd like here
    ├── integration_tests
    │   └── __init__.py
    └── unit_tests
        └── __init__.py
```
r