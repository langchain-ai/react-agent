"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(default=prompts.SYSTEM_PROMPT)
    """The system prompt to use for the agent's interactions.
    
    This prompt sets the context and behavior for the agent.
    """

    model_name: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        "anthropic/claude-3-5-sonnet-20240620"
    )
    """The name of the language model to use for the agent's main interactions.
    
    Should be in the form: provider/model-name.
    """

    scraper_tool_model_name: Annotated[
        str, {"__template_metadata__": {"kind": "llm"}}
    ] = "accounts/fireworks/models/firefunction-v2"
    """The name of the language model to use for the web scraping tool.
    
    This model is specifically used for summarizing and extracting information from web pages.
    """
    max_search_results: int = 10
    """The maximum number of search results to return for each search query."""

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
