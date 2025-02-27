"""Test script for the orchestrator.

This script tests the orchestrator to ensure it works correctly.
"""

import asyncio

from dotenv import load_dotenv
from langchain import hub
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from tw_ai_agents.agents.handoff import _normalize_agent_name
from tw_ai_agents.agents.message_types.base_message_type import State
from tw_ai_agents.agents.tw_supervisor import TWSupervisor
from tw_ai_agents.agents.utils import load_chat_model
from tw_ai_agents.tools.tools import (
    get_knowledge_info,
    real_human_agent_execute_actions,
)

# Load environment variables
load_dotenv()


def get_complete_graph(model, configs: dict, memory) -> TWSupervisor:
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
        )
    ]
    shared_tools = [real_human_agent_execute_actions]

    for config in configs["caseCategories"]:
        description = config["description"]
        instructions = config["instructions"]
        name = _normalize_agent_name(config["name"])
        handoff_conditions = config["handoffConditions"]
        subagents_list.append(
            TWSupervisor(
                agents=[] + shared_agents,
                tools=shared_tools,
                model=model,
                prompt=instructions,
                state_schema=State,
                supervisor_name=name,
                description=description,
                # handoff_conditions=handoff_conditions,
                memory=memory,
            )
        )

    starting_supervisor_prompt = hub.pull("tw-supervisor-system-prompt")
    final_supervisor_prompt = starting_supervisor_prompt.format(
        agents="\n - ".join(
            [subagent.get_pretty_description() for subagent in subagents_list]
        ),
        tools="\n - ".join(
            [tool.get_pretty_description() for tool in supervisor_tools]
        ),
    )

    supervisor_system = TWSupervisor(
        agents=subagents_list,
        model=model,
        prompt=final_supervisor_prompt,
        state_schema=State,
        supervisor_name="tw_supervisor",
        description="Agent able to handle the flow of the conversation.",
        memory=memory,
    )

    return supervisor_system


INPUT_CONFIGS = {
    "company": {
        "_id": "67bedb178cd252e1eeb3c007",
        "name": "typewise",
        "description": "this is the typewise client that handles customer support queries for typewise! asdasdasd",
        "instructions": "",
        "handoffConditions": "",
    },
    "channels": [
        {
            "_id": "67c030ff6b309468e3b17701",
            "channelTypeId": "67bed9fe3b2f84a3a5e67779",
            "clientId": "67bedb178cd252e1eeb3c007",
            "instructions": "test",
            "handoffConditions": "awesome",
            "channelType": {
                "_id": "67bed9fe3b2f84a3a5e67779",
                "name": "Chat",
                "description": "Live chat channel",
                "key": "CHAT",
            },
        },
    ],
    "caseCategories": [
        {
            "_id": "67beddaf61774cd352f423e5",
            "name": "Refunds and cancellations",
            "description": "Handling customer requests for refunds and cancellations of orders or services",
            "instructions": """
:params
- order_id: The ID of the order to be refunded or cancelled
- customer_email: Email address of the customer requesting the refund or cancellation
- reason: Reason for the refund or cancellation request

:return
- Confirmation of refund or cancellation process

For Refund Requests:
1. Verify the order ID and customer email
2. Check if the order is eligible for a refund
3. Process the refund through the payment gateway
4. Send confirmation email to the customer

For Cancellation Requests:
1. Verify the order ID and customer email
2. Check if the order is eligible for cancellation
3. Cancel the order in the system
4. Send confirmation email to the customer

Once you know all the information, you must call the real_human_agent_execute_actions tool to make sure that the real agent does the return.
""",
            "clientId": "67bedb178cd252e1eeb3c007",
            "isActive": True,
            "handoffConditions": "Escalate if the refund or cancellation cannot be processed automatically",
        },
        {
            "_id": "67beddaf61774cd352f423e6",
            "name": "Account Issues",
            "description": "Problems related to user accounts including login, registration, and account management",
            "instructions": """
:params
- account_email: User's email address
- issue_type: Login, Registration, Password Reset, or Other
- error_message: Any error messages the user is seeing (if applicable)

:return
- Specific instructions based on the issue type

For Login Issues:
1. Verify the user's email is registered in our system
2. Check if account is locked due to too many failed attempts
3. Guide user to reset password if needed
4. If persistent, check for browser cache/cookie issues

For Registration Issues:
1. Confirm email is not already registered
2. Check for valid email format
3. Verify all required fields were completed
4. If error persists, suggest using a different browser

For Password Reset:
1. Send password reset link to registered email
2. Guide user to check spam folder if not received
3. Verify the reset link is used within 24 hours
4. If link expired, instruct to request a new one
""",
            "clientId": "67bedb178cd252e1eeb3c007",
            "isActive": True,
            "handoffConditions": "Escalate if account appears compromised or after 3 failed resolution attempts",
        },
        {
            "_id": "67beddaf61774cd352f423e7",
            "name": "Technical Support",
            "description": "Technical issues with the application including bugs, crashes, and performance problems",
            "instructions": """
:params
- app_version: Current version of the application
- device_model: User's device model
- os_version: Operating system version
- issue_description: Detailed description of the technical issue

:return
- Troubleshooting steps based on the issue type

For App Crashes:
1. Instruct user to update to the latest app version
2. Clear app cache and data
3. Restart device
4. If on iOS, check for iOS updates
5. If on Android, check for sufficient storage space

For Performance Issues:
1. Check if device meets minimum requirements
2. Close other running applications
3. Update app to latest version
4. Check for poor network connectivity
5. Reinstall application as last resort

For Feature Not Working:
1. Verify feature is available in user's subscription plan
2. Check if feature requires specific permissions
3. Update to latest app version
4. Clear app cache
5. Provide alternative workflow if feature temporarily unavailable
""",
            "clientId": "67bedb178cd252e1eeb3c007",
            "isActive": True,
            "handoffConditions": "Escalate if issue persists after all troubleshooting steps or if it's a confirmed bug",
        },
    ],
}

if __name__ == "__main__":
    # Run the test
    model_name: str = "openai/gpt-4o"
    model = load_chat_model(model_name)

    messages = [
        HumanMessage(
            # content="Hello, how are you?\nI'd like to cancel the order I just did!"
            # content="My order number is 423423, and my email f.roberts@gmail.com."
            # content="It is because the desk arrived completly broken. I want to return it."
            content="Hello, how are you?\nI'd like to cancel the order I just did!\n\nMy order number is 423423, and my email f.roberts@gmail.com. I want to return it because the desk arrived completly broken."
        ),
    ]

    async def main():
        async with AsyncSqliteSaver.from_conn_string(
            "checkpoints.sqlite"
        ) as saver:
            graph = get_complete_graph(model, INPUT_CONFIGS, memory=saver)
            # Your code here
            compiled_supervisor = graph.get_supervisor_compiled_graph()
            config = {"configurable": {"thread_id": "thread-5e54334"}}
            # async for event in graph.astream_events(..., config, version="v1"):
            # print(event)

            result = await compiled_supervisor.ainvoke(
                input=State(messages=messages),
                config=config,
            )

            result_2 = await compiled_supervisor.ainvoke(
                input=Command(
                    resume="Stop. We can't accept the return. We are not doing return for this things."
                ),
                config=config,
            )

            a = 1

        last_message = result["messages"][-1]
        print(last_message)

    # Run the async function
    asyncio.run(main())

    a = 1
