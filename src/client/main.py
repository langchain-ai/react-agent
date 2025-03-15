import os, getpass
from langchain_core.messages import HumanMessage
from langgraph_sdk import get_client
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def call_client():
    # Replace this with the URL of your deployed graph
    URL = "https://langchain-academy-8011c561878d50b1883f7ed11b32d720.default.us.langgraph.app"
    client = get_client(url=URL)

    # Search all hosted graphs
    assistants = await client.assistants.search()
    # Select the agent
    agent = assistants[0]

    # We create a thread for tracking the state of our run
    thread = await client.threads.create()

    # Input
    input = {"messages": [HumanMessage(content="Multiply 3 by 2.")]}

    # Stream
    async for chunk in client.runs.stream(
            thread['thread_id'],
            "agent",
            input=input,
            stream_mode="values",
        ):
        if chunk.data and chunk.event != "metadata":
            print(chunk.data['messages'][-1])

asyncio.run(call_client())



