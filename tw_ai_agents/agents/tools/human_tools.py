import re
from typing import Any, Union
from typing import Callable, Dict, List

from langchain import hub
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from langgraph.types import interrupt
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel
from typing_extensions import Annotated

from tw_ai_agents.agents.handoff import _normalize_agent_name
from tw_ai_agents.agents.message_types.base_message_type import (
    InterruptBaseModel,
)
from tw_ai_agents.agents.message_types.base_message_type import (
    State,
)

WHITESPACE_RE = re.compile(r"\s+")
SUBAGENT_TOOL_NAME_PREFIX = f"transfer_to_"
SUBAGENT_TOOL_NAME_SUFFIX = "_agent"


COMPLETE_HANDOFF_STRING = "Handoff the full conversation to a real agent."


@tool("real_human_agent_execute_actions")
def real_human_agent_execute_actions(query: str) -> str:
    """
    Make a real agent execute actions.
    This tool makes a real human agent execute actions based on the provided query.

    Args:
        query: The query to make the real human agent execute.

    Returns:
        A string containing the information from the real human agent. It can be a confirmation or a negative answer.
    """
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        query,
    )
    print(f"> Received an input from the interrupt: {answer}")
    return answer


@tool("handoff_conversation_to_real_agent")
def handoff_conversation_to_real_agent() -> str:
    """
    Handoff the full conversation to a real agent.
    """
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        COMPLETE_HANDOFF_STRING,
    )
    return answer


class AskUserTool:
    # Class-level constants
    TOOL_NAME = "ask_user"
    TOOL_DESCRIPTION = "Tool write directly to the user for an additional information or final resolution."
    HUB_MODEL_NAME = "agent-writing_instructions"
    LOG_PREFIX = "> Received an input from the interrupt from user: "

    def __init__(self, prompt_input: dict) -> None:
        """
        Initialize the AskUserTool with prompt input.

        :params:
            prompt_input: Dictionary containing prompt parameters
        """
        self.prompt_input = prompt_input
        self.name = self.TOOL_NAME
        self.description = self.TOOL_DESCRIPTION
        self.langchain_prompt_model = None

    def _get_prompt_model(self):
        """
        Fetch and cache the LangChain prompt model.

        :return: The LangChain prompt model
        """
        if not self.langchain_prompt_model:
            self.langchain_prompt_model = hub.pull(
                self.HUB_MODEL_NAME, include_model=True
            )
        return self.langchain_prompt_model

    def _prepare_input_dict(self, query, messages_to_from_user):
        """
        Prepare the input dictionary for the model.

        :params:
            query: The query to process
            messages_to_from_user: List of previous messages
        :return: Input dictionary for the model
        """
        previous_messages_string = self.format_previous_user_messages(
            messages_to_from_user
        )
        return {
            **self.prompt_input,
            "query": query,
            "previous_messages": previous_messages_string,
        }

    def _process_response(self, user_message, tools_called):
        """
        Process the response from the user.

        :params:
            user_message: The message to show to the user
            tools_called: List of tools that have been called
        :return: Answer from the user and formatted response
        """
        answer = interrupt(
            InterruptBaseModel(
                user_message=user_message,
                tools_called=tools_called,
                destination="user",
            ).dict(),
        )
        print(f"{self.LOG_PREFIX}{answer}")
        return {
            "messages": [AIMessage(content=answer)],
            "messages_to_from_user": [
                AIMessage(content=user_message),
                HumanMessage(content=answer),
            ],
        }

    def invoke(self, state_dict: State) -> Any:
        """
        Function to improve writing of the user-facing message using channel-specific instructions.

        :params:
            state_dict: Current state dictionary
        :return: Updated state dictionary
        """
        prompt_model = self._get_prompt_model()
        query = state_dict.message_from_supervisor[-1]
        input_dict = self._prepare_input_dict(
            query, state_dict.messages_to_from_user
        )

        response = prompt_model.invoke(input=input_dict)
        user_message = response.content

        result_dict = self._process_response(
            user_message, state_dict.tools_called
        )

        return result_dict

    async def ainvoke(self, state_dict: State) -> Any:
        """
        Asynchronous version of invoke.

        :params:
            state_dict: Current state dictionary
        :return: Updated state dictionary
        """
        prompt_model = self._get_prompt_model()
        query = state_dict["query"]
        input_dict = self._prepare_input_dict(
            query, state_dict["messages_to_from_user"]
        )

        response = await prompt_model.ainvoke(input=input_dict)
        user_message = response.content

        result_dict = self._process_response(
            user_message, state_dict["tools_called"]
        )

        return result_dict

    def format_previous_user_messages(
        self, messages_to_from_user: List[Union[HumanMessage, AIMessage]]
    ) -> str:
        """
        Function to format the previous messages from the user into a string.

        :params:
            messages_to_from_user: List of previous messages
        :return: Formatted string of messages
        """

        def format_single_message(
            message: Union[HumanMessage, AIMessage],
        ) -> str:
            return f"{message.type}:\n{message.content}"

        return "\n\n".join(
            [
                format_single_message(message)
                for message in messages_to_from_user
            ]
        )

    def make_call_dependant_agent(
        self,
    ) -> Callable[[Dict], Dict]:
        """
        Create a function that calls an agent and processes its output.
        This function is what is actually executed when the handoff to a sub-agent happens.
        Here is were we process messages going to and coming from sub-agents.

        Returns:
            A callable that invokes the agent and processes its output.
        """

        def _process_output(output: Dict, input_state: State) -> Dict:
            # Remove last two messages which were the AIMessage to call the Tool, plus the ToolMessage created by create_handoff_tool
            delete_messages = [
                RemoveMessage(id=m.id) for m in input_state["messages"][-2:]
            ]

            # Add the two messages back and forth from the user, as a normal conversation
            to_add_messages = output["messages_to_from_user"]

            updated_messages = delete_messages + to_add_messages

            # remove last message_from_supervisor
            return {
                **output,
                "messages": updated_messages,
                "message_from_supervisor": [None],
            }

        def _process_input(
            input: State,
        ) -> State:
            return input

        def call_agent(input_state: State) -> Dict:
            state = _process_input(input_state)
            output = self.invoke(state)
            return _process_output(output, input_state)

        async def acall_agent(input_state: State) -> Dict:
            state = _process_input(input_state)
            output = await self.ainvoke(state)
            return _process_output(output, input_state)

        return RunnableCallable(call_agent, acall_agent)

    def create_handoff_tool(self) -> BaseTool:
        """Create a tool that can handoff control to the requested agent.

        Args:
            agent_name: The name of the agent to handoff control to, i.e.
                the name of the agent node in the multi-agent graph.
                Agent names should be simple, clear and unique, preferably in snake_case,
                although you are only limited to the names accepted by LangGraph
                nodes as well as the tool names accepted by LLM providers
                (the tool name will look like this: `transfer_to_<agent_name>`).

        Returns:
            A tool that can be used to transfer control to another agent.
        """
        tool_name = "ask_user"

        class BaseArgsSchema(BaseModel):
            tool_call_id: Annotated[str, InjectedToolCallId]
            message_for_user: str

        @tool(
            tool_name, description=self.description, args_schema=BaseArgsSchema
        )
        def handoff_to_agent(
            tool_call_id: Annotated[str, InjectedToolCallId],
            message_for_user: str,
        ) -> Command:
            """Ask another agent for help."""
            tool_message = ToolMessage(
                content=f"Successfully transferred to {self.name}\n\n"
                f"## Message from the user\n"
                f"{message_for_user}",
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            return Command(
                goto=self.name,
                graph=Command.PARENT,
                update={
                    "messages": [tool_message],
                    "message_from_supervisor": [message_for_user],
                },
            )

        return handoff_to_agent
