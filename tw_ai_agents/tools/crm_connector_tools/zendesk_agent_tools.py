import copy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from tw_ai_agents.tools.base_agent_tools import BaseAgentWithTools

llm = ChatOpenAI(model="gpt-4o")


class ZendeskAgentWithTools(BaseAgentWithTools):
    def __init__(self):
        node_name = "zendesk_info_searcher"
        base_system_prompt = """You are a helpful assistant that can use tools to answer questions about Zendesk tickets.
        Your goal is to provide accurate information by using the available tools to search for and retrieve ticket information.
        """
        description = "You are a helpful assistant that can use tools to answer questions about Zendesk tickets."
        # base_system_prompt = hub.pull(
        #     "zendesk_info_searcher-system_prompt"
        # ).content
        # description = hub.pull("zendesk_info_searcher-description").content

        super().__init__(
            system_prompt=base_system_prompt,
            node_name=node_name,
            description=description,
        )
        self.state_database = copy.deepcopy(self.BASE_DATABASE)

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
