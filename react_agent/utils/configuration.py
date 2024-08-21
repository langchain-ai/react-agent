"""Define the configurable parameters for the agent."""

from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig


class Configuration(TypedDict):
    """The configuration for the agent."""

    system_prompt: str
    model_name: str
    scraper_tool_model_name: str


def ensure_configurable(config: RunnableConfig) -> Configuration:
    """Ensure the defaults are populated."""
    configurable = config.get("configurable") or {}
    return Configuration(
        system_prompt=configurable.get(
            "system_prompt",
            "You are a helpful AI assistant.\nSystem time: {system_time}",
        ),
        model_name=configurable.get("model_name", "claude-3-5-sonnet-20240620"),
        scraper_tool_model_name=configurable.get(
            "scraper_tool_model_name",
            "accounts/fireworks/models/firefunction-v2",
        ),
    )
