from langchain import hub
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Optional, List

from tw_ai_agents.tools.base_agent_tools import BaseAgentWithTools
from mock.mock_database_service import MockDatabaseService

llm = ChatOpenAI(model="gpt-4o")


class ZendeskAgentWithTools(BaseAgentWithTools):
    def __init__(self, db_service=None):
        node_name = "zendesk_info_searcher"
        base_system_prompt = """You are a helpful assistant that can use tools to answer questions about Zendesk tickets.
        Your goal is to provide accurate information by using the available tools to search for and retrieve ticket information.
        """
        description = "You are a helpful assistant that can use tools to answer questions about Zendesk tickets."
        # base_system_prompt = hub.pull(
        #     "zendesk_info_searcher-system_prompt"
        # ).content
        # description = hub.pull("zendesk_info_searcher-description").content

        # Use the provided database service or default to MockDatabaseService
        self.db_service = db_service if db_service is not None else MockDatabaseService()

        super().__init__(
            system_prompt=base_system_prompt,
            node_name=node_name,
            description=description,
        )

    @tool
    def get_ticket_info(self, ticket_id: str):
        """
        Tool to get the ticket info from a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get information for
        :return: The ticket information
        """
        ticket_info = self.db_service.get_ticket(ticket_id)
        if ticket_info:
            return ticket_info
        return f"No information found for ticket {ticket_id}"

    @tool
    def get_comments(self, ticket_id: str):
        """
        Tool to get the comments from a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get comments for
        :return: The comments from the ticket
        """
        comments = self.db_service.get_ticket_comments(ticket_id)
        if comments:
            return f"Comments for ticket {ticket_id}: {comments}"
        return f"No comments found for ticket {ticket_id}"

    @tool
    def get_ticket_address(self, ticket_id: str):
        """
        Tool to get the address associated with a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get the address for
        :return: The address associated with the ticket
        """
        address = self.db_service.get_ticket_address(ticket_id)
        if address:
            return f"Ticket address for ticket_id {ticket_id} is {address}"
        return f"No address found for ticket {ticket_id}"

    def get_tools(self):
        return [
            self.get_comments,
            self.get_ticket_info,
            self.get_ticket_address,
        ]
