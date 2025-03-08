import asyncio
import os
import uuid
from typing import Dict, Any, List, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from tw_ai_agents.agents.graph_creator_router_supervisor import (
    get_complete_graph_router_supervisor,
    get_input_configs,
)
from tw_ai_agents.agents.graph_runner import GraphRunner
from tw_ai_agents.agents.llm_models_loader import load_chat_model, get_llm_model
from tw_ai_agents.instruction_optimizer.instruction_optimizer import (
    InstructionOptimizationRequest,
    InstructionOptimizationResponse,
    optimize_instruction,
)
import time
from tw_ai_agents.config_handler.constants import DB_CHECKPOINT_PATH
from tw_ai_agents.config_handler.pydantic_models.agent_models import (
    AgentResponseRequest,
    AgentResponseModel,
)
from tw_ai_agents.agents.tools.actions_retriever import (
    get_agent_list,
    ActionListReturnModel,
)
from tw_ai_agents.metadata_generator.endpoints import metadata_router

load_dotenv()
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()

# Add sub-routers to the app
app.include_router(metadata_router)


model_name: str = "openai/gpt-4o"
model = get_llm_model(model_name)


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
                                "message_text": "Hello, how are you?\nI'd like to change the address for my account to Heinrichstrasse 237, Zurich, Switzerland. Please make sure to double check that this was actually done!",
                                "discussion_id": f"123456_{int(time.time())}",
                                "client": "typewise",
                                "channel_type_id": "67bed9fe3b2f84a3a5e67779",
                            },
                        },
                        "vtours_flight_rebooking": {
                            "summary": "Vtours flight rebooking",
                            "value": {
                                "message_type": "user",
                                "message_text": "I need to rebook my flight",
                                "discussion_id": "123456",
                                "client": "vtours",
                                "channel_type_id": "67bed9fe3b2f84a3a5e67779",
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
    message_type = request.message_type

    input_configs = get_input_configs()

    # async def run_supervisor_with_graph():
    #     async with AsyncSqliteSaver.from_conn_string(
    #         DB_CHECKPOINT_PATH
    #     ) as saver:
    #         supervisor = get_complete_graph_router_supervisor(
    #             model,
    #             input_configs,
    #             memory=saver,
    #             channel_type_id=request.channel_type_id,
    #         )
    #         config = {"configurable": {"thread_id": request.discussion_id}}
    #
    #         if message_type == "agent":
    #             # either from agent or the graph is in an interrupt state
    #             initial_state = Command(
    #                 resume=request.message_text,
    #             )
    #         elif message_type == "user":
    #             message = HumanMessage(content=request.message_text)
    #             initial_state = {
    #                 "messages": [message],
    #                 "metadata": {"discussion_id": request.discussion_id},
    #                 "remaining_steps": 10,
    #             }
    #         else:
    #             raise ValueError(
    #                 "Invalid message type. Please use 'user' or 'agent'."
    #             )
    #
    #         return GraphRunner(supervisor).arun_supervisor(
    #             initial_state, config
    #         )
    #
    # # Run the supervisor with proper State object
    # response = asyncio.run(run_supervisor_with_graph())

    def run_supervisor_with_graph():
        with SqliteSaver.from_conn_string(DB_CHECKPOINT_PATH) as saver:
            start_time = time.time()
            supervisor = get_complete_graph_router_supervisor(
                model,
                input_configs,
                memory=saver,
                channel_type_id=request.channel_type_id,
            )
            print(f"Graph creation time: {time.time() - start_time}")
            config = {"configurable": {"thread_id": request.discussion_id}}

            if message_type == "agent":
                # either from agent or the graph is in an interrupt state
                initial_state = Command(
                    resume=request.message_text,
                )
            elif message_type == "user":
                message = HumanMessage(content=request.message_text)
                initial_state = {
                    "messages": [message],
                    "metadata": {"discussion_id": request.discussion_id},
                    "remaining_steps": 10,
                }
            else:
                raise ValueError(
                    "Invalid message type. Please use 'user' or 'agent'."
                )

            return GraphRunner(supervisor).run_supervisor(initial_state, config)

    # Run the supervisor with proper State object
    response = run_supervisor_with_graph()

    if response and "messages" in response and response["messages"]:
        # Get the last message
        messages = cast(List[BaseMessage], response["messages"])
        last_message = messages[-1]

        message_id = getattr(last_message, "id", None) or str(uuid.uuid4())

        metadata: Dict[str, Any] = {}
        if "metadata" in response and response["metadata"]:
            metadata = response["metadata"]

        metadata["discussion_id"] = request.discussion_id
        target_entity = metadata.get("target_entity", "user")

        return AgentResponseModel(
            message_type=target_entity,
            message_text=str(last_message.content),
            message_id=message_id,
            metadata=metadata,
        )
    else:
        return AgentResponseModel(
            message_type="user",
            message_text="I'm sorry, I couldn't process your request at this time.",
            message_id=str(uuid.uuid4()),
            metadata={
                "status": "error",
                "discussion_id": request.discussion_id,
            },
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
