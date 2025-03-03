"""Test script for the orchestrator.

This script tests the orchestrator to ensure it works correctly.
"""

import asyncio
from typing import Dict, Any

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tw_ai_agents.agents.handoff import _normalize_agent_name
from tw_ai_agents.agents.message_types.base_message_type import State
from tw_ai_agents.agents.tw_supervisor import TWSupervisor
from tw_ai_agents.agents.utils import load_chat_model
from tw_ai_agents.agents.tools.actions_retriever import AGENT_LIST
from tw_ai_agents.agents.tools.tools import (
    get_knowledge_info,
    handoff_conversation_to_real_agent,
    real_human_agent_execute_actions,
)

# Load environment variables
load_dotenv()


def get_complete_graph(
    model, configs: dict, memory, channel_type_id: str
) -> TWSupervisor:
    """Test the orchestrator system with a simple query."""

    supervisor_tools = []
    subagents_list = []
    shared_agents = [
        # Now, create a second supervisor that could potentially be called by the main supervisor
        TWSupervisor(
            agents=[],
            model=model,
            tools=[get_knowledge_info],
            prompt="You are an agent specialized in knowledge information lookup.",
            state_schema=State,
            supervisor_name="knowledge_handler",
            description="Agent able to lookup knowledge information.",
            memory=memory,
        ),
    ]
    shared_tools = [
        handoff_conversation_to_real_agent,
        real_human_agent_execute_actions,
    ]

    for config in configs["caseCategories"]:
        description = config["description"]
        name = _normalize_agent_name(config["name"])
        instructions_with_tools = config["instructions"]
        instructions = instructions_with_tools["text"]
        action_list = instructions_with_tools["actions"]
        agent_list_as_tools = []
        for action in action_list:
            agent = AGENT_LIST[action["id"]]
            new_agent = agent()
            agent_list_as_tools.append(
                TWSupervisor(
                    agents=[],
                    model=model,
                    tools=new_agent.get_tools(),
                    prompt=new_agent.system_prompt,
                    state_schema=State,
                    supervisor_name=f"{name}_{new_agent.node_name}",
                    description=new_agent.description,
                )
            )

        handoff_conditions = config["handoffConditions"]["text"]
        agent_prompt = hub.pull("case_agent_initial_prompt").format(
            instructions=instructions, handoff_conditions=handoff_conditions
        )
        subagents_list.append(
            TWSupervisor(
                agents=agent_list_as_tools + shared_agents,
                tools=shared_tools,
                model=model,
                prompt=agent_prompt,
                state_schema=State,
                supervisor_name=name,
                description=description,
                # handoff_conditions=handoff_conditions,
                memory=memory,
            )
        )

    # Main Supervisor
    correct_channel = next(
        (
            channel
            for channel in configs["channels"]
            if channel["channelTypeId"] == channel_type_id
        ),
        None,
    )
    if correct_channel is None:
        raise ValueError("Channel type not found in the configuration data.")
    channel_type = correct_channel["channelType"]["name"]
    channel_rules = correct_channel["instructions"]["text"]

    starting_supervisor_prompt = hub.pull("tw-supervisor-system-prompt")
    final_supervisor_prompt = starting_supervisor_prompt.format(
        agents="- "
        + "\n - ".join(
            [subagent.get_pretty_description() for subagent in subagents_list]
        ),
        tools="- "
        + "\n - ".join(
            [tool.get_pretty_description() for tool in supervisor_tools]
        ),
        channel_type=channel_type,
        channel_rules=channel_rules,
    )

    supervisor_system = TWSupervisor(
        agents=subagents_list,
        model=model,
        prompt=final_supervisor_prompt,
        state_schema=State,
        supervisor_name="tw_supervisor",
        description="Agent able to handle the flow of the conversation.",
        memory=memory,
        tools=[handoff_conversation_to_real_agent],
    )

    return supervisor_system


def get_input_configs() -> Dict[str, Any]:
    """
    Fetch configuration data from the API endpoint.

    Returns:
        Dict[str, Any]: The configuration data including company info, channels, and case categories.

    Raises:
        Exception: If there's an error fetching or processing the configuration data.
    """
    response = requests.get(
        "https://gqyrvmzupb.eu-central-1.awsapprunner.com/api/config/all"
    )
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()


if __name__ == "__main__":
    # Run the test
    model_name: str = "openai/gpt-4o"
    model = load_chat_model(model_name)

    messages = [
        HumanMessage(
            # content="Hello, how are you?\nI'd like to cancel the order I just did!"
            # content="My order number is 423423, and my email f.roberts@gmail.com."
            # content="It is because the desk arrived completly broken. I want to return it."
            content="Hello, how are you?\nI'd like to change the shipping address for my ticket 14983 to Heinrichstrasse 237, Zurich, Switzerland. Please make sure to double check that this was actually done!"
        ),
    ]

    async def main():
        async with AsyncSqliteSaver.from_conn_string(
            "checkpoints.sqlite"
        ) as saver:
            graph = get_complete_graph(model, get_input_configs(), memory=saver)
            # Your code here
            compiled_supervisor = graph.get_supervisor_compiled_graph()
            config = {"configurable": {"thread_id": "thread-5e54334"}}
            # async for event in graph.astream_events(..., config, version="v1"):
            # print(event)

            result = await compiled_supervisor.ainvoke(
                input=State(messages=messages),
                config=config,
            )

            # result_2 = await compiled_supervisor.ainvoke(
            #     input=Command(
            #         resume="Stop. We can't accept the return. We are not doing return for this things."
            #     ),
            #     config=config,
            # )

            a = 1

        last_message = result["messages"][-1]
        print(last_message)

    # Run the async function
    asyncio.run(main())

    a = 1
