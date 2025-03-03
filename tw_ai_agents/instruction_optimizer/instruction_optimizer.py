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


async def optimize_instruction(
    request: InstructionOptimizationRequest,
) -> InstructionOptimizationResponse:
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
        langchain_prompt_model = hub.pull(
            "agent-optimize_instructions", include_model=True
        )
        response = await langchain_prompt_model.ainvoke(
            {"instructions": sanitized_instruction}
        )
        return InstructionOptimizationResponse(**response)

    except Exception as e:
        # Log the specific error for debugging
        print(f"Error in optimize_instruction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error optimizing instruction: {str(e)}"
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
