import httpx

from react_agent.utils.configuration import ensure_configurable
from langchain_core.runnables import RunnableConfig

from langchain.chat_models import init_chat_model
from datetime import datetime, timezone

from react_agent.utils.utils import get_message_text


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

    configuration = ensure_configurable(config)
    model = init_chat_model(configuration["model_name"])
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


# Note, in a real use case, you'd want to use a more robust search API.
async def search_duckduckgo(query: str) -> dict:
    """Search DuckDuckGo for the given query and return the JSON response.

    Results are limited, as this is the free public API.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.duckduckgo.com/", params={"q": query, "format": "json"}
        )
        result = response.json()

        result.pop("meta", None)
        return result


async def search_wikipedia(query: str) -> dict:
    """Search Wikipedia for the given query and return the JSON response."""
    url = "https://en.wikipedia.org/w/api.php"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
            },
        )
        return response.json()


TOOLS = [
    scrape_webpage,
    search_duckduckgo,
    search_wikipedia,
]
