"""Define a custom graph that implements a simple Reasoning and Action agent pattern. 

Works with a chat model that utilizes tool calling."""

from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from react_agent.utils.configuration import Configuration, ensure_configurable
from react_agent.utils.state import InputState, State
from react_agent.utils.tools import TOOLS

# Define the function that calls the model


async def call_model(state: State, config: RunnableConfig):
    """Call the LLM powering our "agent"."""
    configuration = ensure_configurable(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [("system", configuration["system_prompt"]), ("placeholder", "{messages}")]
    )
    model = init_chat_model(configuration["model_name"]).bind_tools(TOOLS)

    message_value = await prompt.ainvoke(
        {**state, "system_time": datetime.now(tz=timezone.utc).isoformat()}, config
    )
    response: AIMessage = await model.ainvoke(message_value, config)
    if state["is_last_step"] and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph

workflow = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
workflow.add_node(call_model)
workflow.add_node("tools", ToolNode(TOOLS))


# Set the entrypoint as `call_model`
# This means that this node is the first one called
workflow.add_edge("__start__", "call_model")


# Define the function that determines whether to continue or not
def route_model_output(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise if there are tools called, we continue
    else:
        return "tools"


# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the edges' source node. We use `call_model`.
    # This means these are the edges taken after the `call_model` node is called.
    "call_model",
    # Next, we pass in the function that will determine the sink node(s), which
    # will be called after the source node is called.
    route_model_output,
)

# We now add a normal edge from `tools` to `call_model`.
# This means that after `tools` is called, `call_model` node is called next.
workflow.add_edge("tools", "call_model")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = workflow.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
