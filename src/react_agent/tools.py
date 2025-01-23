"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools import OpenWeatherMapQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


async def get_weather(location: str, *, config: Annotated[RunnableConfig, InjectedToolArg]) -> Optional[dict[str, Any]]:
    """Get the current weather for a given city or zip code.

    Args:
        location: Can be in formats:
            - City name (e.g., "London")
            - City name with country code (e.g., "London,GB")
            - US ZIP code (e.g., "90210")

    Note: The location must be properly formatted before calling this function.
    Invalid formats may result in no weather data being returned.
    """
    configuration = Configuration.from_runnable_config(config)

    wrapper = OpenWeatherMapAPIWrapper(openweathermap_api_key=configuration.openweather_api_key)
    wrapped = OpenWeatherMapQueryRun(api_wrapper=wrapper)
    result = await wrapped.ainvoke(location)

    return cast(dict[str, Any], result)


TOOLS: List[Callable[..., Any]] = [search, get_weather]
