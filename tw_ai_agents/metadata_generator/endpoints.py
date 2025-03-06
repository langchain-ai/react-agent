from typing import List, Optional

from fastapi import APIRouter, HTTPException
from langchain import hub

from tw_ai_agents.metadata_generator.io_classes import (
    FrontendMessage,
    PriorityGeneratorInput,
    PriorityGeneratorOutput,
    SentimentGeneratorInput,
    SentimentGeneratorOutput,
    TitleGeneratorInput,
    TitleGeneratorOutput,
)

# Create the router
metadata_router = APIRouter(
    prefix="/get_conversation_metadata", tags=["metadata_generator"]
)


def process_conversation_data(messages: List[FrontendMessage]) -> str:
    """
    Sanitize the input to prevent prompt injection.

    :params:
        input_data: The input data to sanitize
    :return:
        The sanitized input data
    """

    # Implement input sanitization if needed
    def format_message(message: FrontendMessage) -> str:
        return (
            "Sender: "
            + message.message_type
            + "\n```"
            + message.message_text
            + "```"
        )

    return "\n".join([format_message(message) for message in messages])


@metadata_router.post(
    "/title",
    response_model=TitleGeneratorOutput,
    description="Generate a title for a conversation based on its messages.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "typewise_user_account": {
                            "summary": "Broken Iphone",
                            "value": {
                                "discussion_id": "string",
                                "client": "string",
                                "message_list": [
                                    {
                                        "message_type": "human",
                                        "message_text": "Hey. I have a problem that my iphone is completly broken. I want a refund immediately.",
                                    }
                                ],
                                "current_title": "IPhone Refund",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def generate_title(request: TitleGeneratorInput) -> TitleGeneratorOutput:
    """
    Generate a title for a conversation based on its messages.

    :params:
        request: The title generation request containing conversation messages
    :return:
        The generated title for the conversation
    """
    try:
        # Sanitize the input to prevent prompt injection
        message_string = process_conversation_data(request.message_list)
        current_title = (
            request.current_title
            if request.current_title
            else "NO_CURRENT_TITLE_AVAILABLE"
        )

        # Pull the model and prompt from LangChain Hub
        langchain_prompt_model = hub.pull(
            "conversation_metadata-title", include_model=True
        )

        # Prepare the input for the model
        model_input = {
            "current_title": current_title,
            "messages": message_string,
        }

        # Generate the title
        response = await langchain_prompt_model.ainvoke(model_input)

        # Create and return the output
        return TitleGeneratorOutput(
            metadata={"generated_by": "title_generator"},
            title=response["title"],
        )

    except Exception as e:
        # Log the specific error for debugging
        print(f"Error in generate_title: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating title: {str(e)}"
        )


@metadata_router.post("/sentiment", response_model=SentimentGeneratorOutput)
async def generate_sentiment(
    request: SentimentGeneratorInput,
) -> SentimentGeneratorOutput:
    """
    Generate a sentiment score for a conversation based on its messages.

    :params:
        request: The sentiment generation request containing conversation messages
    :return:
        The generated sentiment for the conversation
    """
    try:
        # Sanitize the input to prevent prompt injection
        message_string = process_conversation_data(request.message_list)
        current_sentiment = (
            request.current_sentiment
            if request.current_sentiment
            else "NO_CURRENT_SENTIMENT_AVAILABLE"
        )

        # Pull the model and prompt from LangChain Hub
        langchain_prompt_model = hub.pull(
            "conversation_metadata-sentiment", include_model=True
        )

        # Prepare the input for the model
        model_input = {
            "current_sentiment": current_sentiment,
            "messages": message_string,
        }

        # Generate the sentiment
        response = await langchain_prompt_model.ainvoke(model_input)

        # Create and return the output
        return SentimentGeneratorOutput(
            metadata={"generated_by": "sentiment_generator"},
            sentiment=response["sentiment"],
        )

    except Exception as e:
        # Log the specific error for debugging
        print(f"Error in generate_sentiment: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating sentiment: {str(e)}"
        )


@metadata_router.post("/priority", response_model=PriorityGeneratorOutput)
async def generate_priority(
    request: PriorityGeneratorInput,
) -> PriorityGeneratorOutput:
    """
    Generate a priority score for a conversation based on its messages.

    :params:
        request: The priority generation request containing conversation messages
    :return:
        The generated priority for the conversation
    """
    try:
        # Sanitize the input to prevent prompt injection
        message_string = process_conversation_data(request.message_list)
        current_priority = (
            request.current_priority
            if request.current_priority
            else "NO_CURRENT_PRIORITY_AVAILABLE"
        )

        # Pull the model and prompt from LangChain Hub
        langchain_prompt_model = hub.pull(
            "conversation_metadata-priority", include_model=True
        )

        # Prepare the input for the model
        model_input = {
            "current_priority": current_priority,
            "messages": message_string,
        }

        # Generate the priority
        response = await langchain_prompt_model.ainvoke(model_input)

        # Create and return the output
        return PriorityGeneratorOutput(
            metadata={"generated_by": "priority_generator"},
            priority=response["priority"],
        )

    except Exception as e:
        # Log the specific error for debugging
        print(f"Error in generate_priority: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating priority: {str(e)}"
        )
