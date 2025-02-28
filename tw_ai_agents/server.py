import asyncio
import os
import uuid
from typing import Dict, Any, List, cast, Optional
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tw_ai_agents.agents.graph_creator import (
    get_complete_graph,
    get_input_configs,
)
from tw_ai_agents.agents.tw_supervisor import run_supervisor
from tw_ai_agents.agents.message_types.base_message_type import State
from tw_ai_agents.agents.utils import load_chat_model
from tw_ai_agents.config.constants import DB_CHECKPOINT_PATH
from tw_ai_agents.pydantic_models.agent_models import (
    AgentResponseRequest,
    AgentResponseModel,
)
from instruction_optimizer.instruction_optimizer import (
    InstructionOptimizationRequest,
    InstructionOptimizationResponse,
    optimize_instruction,
)

from tw_ai_agents.tools.actions_retriever import (
    get_agent_list,
    ActionListReturnModel,
)

load_dotenv()
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()

model_name: str = "openai/gpt-4o"
model = load_chat_model(model_name)


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
                                "client": "typewise",
                            },
                        },
                        "vtours_flight_rebooking": {
                            "summary": "Vtours flight rebooking",
                            "value": {
                                "message_type": "user",
                                "message_text": "I need to rebook my flight",
                                "discussion_id": "123456",
                                "client": "vtours",
                            },
                        },
                    }
                }
            }
        }
    },
)
def process_agent_response(
    request: AgentResponseRequest,
) -> AgentResponseModel:
    """Process an agent or user message and generate a response.

    Args:
        request: The AgentResponseRequest containing message details.

    Returns:
        AgentResponseModel: The response with message details and metadata.

    Raises:
        HTTPException: If there's an error processing the message.
    """
    try:
        # Create the initial state with message
        message = HumanMessage(content=request.message_text)

        # Initialize a proper State object using dict notation
        initial_state: State = {
            "messages": [message],
            "next": "tw_supervisor",  # Required field in State class
            "metadata": {"discussion_id": request.discussion_id},
            "remaining_steps": 10,
        }

        async def run_supervisor_with_graph():
            async with AsyncSqliteSaver.from_conn_string(
                DB_CHECKPOINT_PATH
            ) as saver:
                graph = get_complete_graph(
                    model, get_input_configs(), memory=saver
                )
                # Your code here
                compiled_supervisor = graph.get_supervisor_compiled_graph()
                config = {"configurable": {"thread_id": request.discussion_id}}

                return await run_supervisor(
                    initial_state, compiled_supervisor, config
                )

        # Run the supervisor with proper State object
        response = asyncio.run(run_supervisor_with_graph())

        if response and "messages" in response and response["messages"]:
            # Get the last message
            messages = cast(List[BaseMessage], response["messages"])
            last_message = messages[-1]

            message_id = getattr(last_message, "id", None) or str(uuid.uuid4())

            metadata: Dict[str, Any] = {}
            if "metadata" in response and response["metadata"]:
                metadata = response["metadata"]

            metadata["discussion_id"] = request.discussion_id

            return AgentResponseModel(
                message_type="agent",
                message_text=str(last_message.content),
                message_id=message_id,
                metadata=metadata,
            )
        else:
            return AgentResponseModel(
                message_type="agent",
                message_text="I'm sorry, I couldn't process your request at this time.",
                message_id=str(uuid.uuid4()),
                metadata={
                    "status": "error",
                    "discussion_id": request.discussion_id,
                },
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing agent response: {str(e)}"
        )


@app.post(
    "/optimize_instruction",
    response_model=InstructionOptimizationResponse,
    description="Optimizes an instruction for AI models",
)
async def api_optimize_instruction(
    request: InstructionOptimizationRequest,
) -> InstructionOptimizationResponse:
    """API endpoint to optimize an instruction for better AI responses."""
    return await optimize_instruction(request)


@app.get(
    "/action_list",
    response_model=ActionListReturnModel,
    description="Returns a list of available actions",
)
async def get_action_list() -> ActionListReturnModel:
    """API endpoint to optimize an instruction for better AI responses."""
    return get_agent_list()


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )
