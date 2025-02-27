"""Instruction optimization module for AI prompts.

This module provides functionality to optimize instructions for AI models.
"""

import os
import asyncio
import json
from typing import List
from pydantic import BaseModel
from fastapi import HTTPException
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class InstructionOptimizationRequest(BaseModel):
    """Request model for instruction optimization."""
    instruction_text: str


class InstructionOptimizationResponse(BaseModel):
    """Response model for instruction optimization."""
    corrected_text: str
    suggestions: List[str]


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
        # Create a prompt for OpenAI
        prompt = f"""You are an expert at optimizing instructions for AI models. 
Please optimize the following instruction to make it clearer, more specific, and more effective:

INSTRUCTION: {request.instruction_text}

Respond in the following format:

CORRECTED INSTRUCTION:
[Your optimized version of the instruction]

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3]
- [Add more suggestions if needed]
"""

        # Call OpenAI API with the new client
        response = await client.chat.completions.create(
            model="gpt-4",  # or another appropriate model
            messages=[
                {"role": "system",
                    "content": "You are an AI instruction optimization assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        # Extract the response content
        content = response.choices[0].message.content

        # Parse the response to extract corrected text and suggestions
        parts = content.split("SUGGESTIONS:")

        if len(parts) < 2 or "CORRECTED INSTRUCTION:" not in parts[0]:
            # Fallback if the format is not as expected
            return InstructionOptimizationResponse(
                corrected_text=request.instruction_text,
                suggestions=[
                    "Could not parse optimization suggestions. Please try again."]
            )

        corrected_text = parts[0].replace("CORRECTED INSTRUCTION:", "").strip()
        suggestions_text = parts[1].strip()
        suggestions = [s.strip().lstrip('- ')
                       for s in suggestions_text.split("\n")
                       if s.strip() and not s.strip().isspace()]

        return InstructionOptimizationResponse(
            corrected_text=corrected_text,
            suggestions=suggestions
        )

    except Exception as e:
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
        print(json.dumps(response.dict(), indent=2))

    except Exception as e:
        print(f"Error during optimization: {str(e)}")


if __name__ == "__main__":
    import sys

    # Get instruction from command line arguments or use a default
    instruction = " ".join(sys.argv[1:]) if len(
        sys.argv) > 1 else "Tell me about the history of artificial intelligence."

    # Run the test
    asyncio.run(test_optimizer(instruction))
