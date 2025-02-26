"""Server for the customer service supervisor agent API.

This module provides a server for the customer service supervisor agent API.
"""

import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from supervisor_agent.api import app
from supervisor_agent.orchestrator import create_orchestrator_system

# Load environment variables
load_dotenv()

# Get port from environment variable or use default
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()

# Initialize the orchestrator system
orchestrator = create_orchestrator_system()


class StateRequest(BaseModel):
    state: Dict[str, Any]


@app.post("/invoke")
async def invoke_orchestrator(state_request: StateRequest):
    try:
        # Call the orchestrator's ainvoke method
        # response = await orchestrator.ainvoke(state_request.state)
        response = orchestrator.invoke(state_request.state)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )
