"""This module provides example tools for web scraping and search functionality.

It includes:
- A web scraper that uses an LLM to summarize content based on instructions
- A basic Tavily search function

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, cast

import httpx
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration
from react_agent.utils import get_message_text


# note that arguments typed as "RunnableConfig" in tools will be excluded from the schema generated
# for the model.
# They are treated as "injected arguments"
async def scrape_webpage(url: str, instructions: str, *, config: RunnableConfig) -> str:
    """Scrape the given webpage and return a summary of text based on the instructions.

    Args:
        url: The URL of the webpage to scrape.
        instructions: The instructions to give to the scraper. An LLM will be used to respond using the
            instructions and the scraped text.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        web_text = response.text

    configuration = Configuration.from_runnable_config(config)
    model = init_chat_model(configuration.model_name)
    response_msg = await model.ainvoke(
        [
            (
                "system",
                "You are a helpful web scraper AI assistant. You are working in extractive Q&A mode, meaning you refrain from making overly abstractive responses."
                "Respond to the user's instructions."
                " Based on the provided webpage. If you are unable to answer the question, let the user know. Do not guess."
                " Provide citations and direct quotes when possible."
                f" \n\n<webpage_text>\n{web_text}\n</webpage_text>"
                f"\n\nSystem time: {datetime.now(tz=timezone.utc)}",
            ),
            ("user", instructions),
        ]
    )
    return get_message_text(response_msg)


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


TOOLS: List[Callable[..., Any]] = [scrape_webpage, search]
