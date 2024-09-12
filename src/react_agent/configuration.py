"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = "You are a helpful AI assistant.\nSystem time: {system_time}"
    model_name: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = (
        "anthropic/claude-3-5-sonnet-20240620"
    )
    scraper_tool_model_name: Annotated[
        str, {"__template_metadata__": {"kind": "llm"}}
    ] = "accounts/fireworks/models/firefunction-v2"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
