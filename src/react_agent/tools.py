"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast
import requests
import json

from langchain_community.tools.tavily_search import TavilySearchResults
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

def get_logs(request_id: str) -> str:
    """
    Sends a POST request to the specified API endpoint with the given request ID.
    
    Parameters:
    - request_id (str): The ID to be included in the request body.
    
    Returns:
    - response: The response object from the API.
    """
    url = 'https://gzxkt6chp7.execute-api.us-west-2.amazonaws.com/default/Kenari_Log_Parser'
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "requestId": request_id
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.text

TOOLS: List[Callable[..., Any]] = [search, get_logs]