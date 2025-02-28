"""Instruction optimization module for AI prompts.

This module provides functionality to optimize instructions for AI models.
"""

import os
import asyncio
import json
import re
from typing import List
from pydantic import BaseModel, Field
from fastapi import HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain import hub
# Load environment variables
load_dotenv()

# Configure OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class InstructionOptimizationRequest(BaseModel):
    """Request model for instruction optimization."""
    instruction_text: str = Field(..., min_length=1, max_length=4000)


class InstructionOptimizationResponse(BaseModel):
    """Response model for instruction optimization."""
    corrected_text: str
    suggestions: List[str]


def sanitize_input(text: str) -> str:
    return text


async def optimize_instruction(request: InstructionOptimizationRequest) -> InstructionOptimizationResponse:
    """Optimize an instruction for better AI responses using OpenAI directly.

    Args:
        request: The InstructionOptimizationRequest containing the instruction text.

    Returns:
        InstructionOptimizationResponse: The optimized instruction and suggestions.

    Raises:
        HTTPException: If there's an error processing the instruction.
    """
    try:
        # Sanitize the input to prevent prompt injection
        sanitized_instruction = sanitize_input(request.instruction_text)
        langchainPrompt = hub.pull("agent-optimize_instructions")
        system_message = langchainPrompt.messages[0].prompt.template

        user_prompt = langchainPrompt.messages[1].format(
            instructions=sanitized_instruction).content

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "optimize_instruction",
                    "description": "Optimize an instruction to make it clearer, more specific, and more effective",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "corrected_text": {
                                "type": "string",
                                "description": "The optimized version of the instruction formatted as bullet points"
                            },
                            "suggestions": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of suggestions for improving the instruction"
                            }
                        },
                        "required": ["corrected_text", "suggestions"]
                    }
                }
            }
        ]

        # Create a prompt for OpenAI with clear boundaries
#         prompt = f"""Your task is to optimize the following instruction to make it clearer, more specific, and more effective.

# INSTRUCTION TO OPTIMIZE: {sanitized_instruction}"""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {
                "name": "optimize_instruction"}},
            temperature=0.3
        )

        # Extract the function call result
        function_call = response.choices[0].message.tool_calls[0]
        result = json.loads(function_call.function.arguments)

        return InstructionOptimizationResponse(
            corrected_text=result["corrected_text"],
            suggestions=result["suggestions"]
        )

    except Exception as e:
        # Log the specific error for debugging
        print(f"Error in optimize_instruction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error optimizing instruction: {str(e)}"
        )


async def test_optimizer(instruction_text: str) -> None:
    """Test the instruction optimizer with a given instruction text.

    Args:
        instruction_text: The instruction text to optimize.
    """
    print(f"Testing instruction optimizer with: '{instruction_text}'")

    request = InstructionOptimizationRequest(instruction_text=instruction_text)

    try:
        response = await optimize_instruction(request)

        print("\n=== OPTIMIZATION RESULTS ===")
        print("\nCORRECTED INSTRUCTION:")
        print(response.corrected_text)

        print("\nSUGGESTIONS:")
        for i, suggestion in enumerate(response.suggestions, 1):
            print(f"{i}. {suggestion}")

        # Also output as JSON for programmatic use
        print("\n=== JSON OUTPUT ===")
        print(json.dumps(response.model_dump(), indent=2))

    except Exception as e:
        print(f"Error during optimization: {str(e)}")


if __name__ == "__main__":

    instruction = open("sample-prompts/sample1.md", "r").read()
    # Run the test
    asyncio.run(test_optimizer(instruction))
