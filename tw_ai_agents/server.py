"""Server for the customer service supervisor agent API.

This module provides a server for the customer service supervisor agent API.
"""

import os
import uuid
from typing import Dict, Any, List
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tw_ai_agents.agents.tw_supervisor import run_supervisor
from tw_ai_agents.agents.base_agent import State
from tw_ai_agents.pydantic_models.agent_models import (
    AgentResponseRequest,
    AgentResponseModel,
)
from tw_ai_agents.instruction_optimizer import (
    InstructionOptimizationRequest,
    InstructionOptimizationResponse,
    optimize_instruction,
)


# Load environment variables
load_dotenv()

# Get port from environment variable or use default
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()


@app.post(
    "/agent_response",
    response_model=AgentResponseModel,
    description="Processes an agent or user message and returns a response",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "typewise_user_account": {
                            "summary": "User message example",
                            "value": {
                                "message_type": "user",
                                "message_text": "Hello, how are you?\nI'd like to change the shipping address for my ticket 14983 to Heinrichstrasse 237, Zurich, Switzerland. Please make sure to double check that this was actually done!",
                                "discussion_id": "123456",
                                "client": "typewise"
                            }
                        },
                        "vtours_flight_rebooking": {
                            "summary": "Vtours flight rebooking",
                            "value": {
                                "message_type": "user",
                                "message_text": "I need to rebook my flight",
                                "discussion_id": "123456",
                                "client": "vtours"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def process_agent_response(request: AgentResponseRequest) -> AgentResponseModel:
    """Process an agent or user message and generate a response.

    Args:
        request: The AgentResponseRequest containing message details.

    Returns:
        AgentResponseModel: The response with message details and metadata.

    Raises:
        HTTPException: If there's an error processing the message.
    """
    try:
        message = HumanMessage(content=request.message_text)
        state_dict = {
            "messages": [message],
            "next": "tw_supervisor",
            "discussion_id": request.discussion_id,
        }

        response: State = await run_supervisor(state_dict)

        if response and "messages" in response and response["messages"]:
            # Get the last message
            last_message = response["messages"][-1]

            message_id = last_message.id or str(uuid.uuid4())

            # TODO: Get run_supervisor to return metadata (tool calls, etc)

            metadata: Dict[str, Any] = {}

            return AgentResponseModel(
                message_type="agent",
                message_text=last_message.content,
                message_id=message_id,
                metadata=metadata
            )
        else:
            return AgentResponseModel(
                message_type="agent",
                message_text="I'm sorry, I couldn't process your request at this time.",
                message_id=str(uuid.uuid4()),
                metadata={"status": "error",
                          "discussion_id": request.discussion_id}
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing agent response: {str(e)}"
        )


@app.post(
    "/optimize_instruction",
    response_model=InstructionOptimizationResponse,
    description="Optimizes an instruction for AI models",
)
async def api_optimize_instruction(request: InstructionOptimizationRequest) -> InstructionOptimizationResponse:
    """API endpoint to optimize an instruction for better AI responses."""
    return await optimize_instruction(request)


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )
