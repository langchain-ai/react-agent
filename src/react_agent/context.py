"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated, get_type_hints

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args, with type conversion."""
        type_hints = get_type_hints(self.__class__)
        for f in fields(self):
            if not f.init:
                continue

            current_value = getattr(self, f.name)
            env_value = os.environ.get(f.name.upper(), None)
            if current_value == f.default and env_value is not None:
                # Convert env_value to the correct type
                target_type = type_hints.get(f.name, str)
                try:
                    if target_type is int:
                        env_value = int(env_value)
                    elif target_type is float:
                        env_value = float(env_value)
                    # Add more types as needed
                except Exception:
                    pass  # fallback to string if conversion fails
                setattr(self, f.name, env_value)
