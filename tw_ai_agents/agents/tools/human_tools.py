from typing import Any, List, Union

from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from tw_ai_agents.agents.message_types.base_message_type import (
    InterruptBaseModel,
    State,
)

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
    TOOL_DESCRIPTION = "Tool to ask the final user for an additional input."
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
