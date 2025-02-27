from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tw_ai_agents.agents.base_agent import BaseAgent

llm = ChatOpenAI(model="gpt-4o")


class ZendeskAgentWithTools(BaseAgent):
    def __init__(self, system_prompt: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_name = "zendesk_info_searcher"
        base_system_prompt = """You are a helpful assistant that can use tools to answer questions about Zendesk tickets.
        Your goal is to provide accurate information by using the available tools to search for and retrieve ticket information.
        """
        self.system_prompt = system_prompt or base_system_prompt
        self.description = "You are a helpful assistant that can use tools to answer questions about Zendesk tickets."

    @tool
    @staticmethod
    def get_ticket_info(ticket_id: str):
        """
        Tool to get the ticket info from a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get information for
        :return: The ticket information
        """
        return f"This is the ticket {ticket_id}. Infos: very nice ticket"

    @tool
    @staticmethod
    def get_comments(ticket_id: str):
        """
        Tool to get the comments from a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get comments for
        :return: The comments from the ticket
        """
        return f"This is are the comments from the ticket {ticket_id}: [comment1, comment2, comment3]"

    @tool
    @staticmethod
    def get_ticket_address(ticket_id: str):
        """
        Tool to get the comments from a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get comments for
        :return: The comments from the ticket
        """
        return f"Ticket address for ticket_id {ticket_id} is Heinrichstrasse 267, 8005 Zurich, Switzerland"

    def get_tools(self):
        return [
            self.get_comments,
            self.get_ticket_info,
            self.get_ticket_address,
        ]

    def get_agent(self):

        tool_list = [self.get_comments, self.get_ticket_info]
        agent = create_react_agent(
            llm, tools=tool_list, prompt=self.system_prompt, name=self.node_name
        )
        return agent


if __name__ == "__main__":

    system_prompt = """You are a helpful assistant that can use tools to answer questions about Zendesk tickets.
Your goal is to provide accurate information by using the available tools to search for and retrieve ticket information.
"""

    zendesk_agent = ZendeskAgentWithTools(system_prompt=system_prompt)
    graph = zendesk_agent.get_agent()

    for s in graph.stream(
        {
            "messages": [
                (
                    "user",
                    "Can you give me the messages from the ticket 1234567890",
                )
            ]
        },
        {"recursion_limit": 100},
    ):
        print(s)
        print("---")
