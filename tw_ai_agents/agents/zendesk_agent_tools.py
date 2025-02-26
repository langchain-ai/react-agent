from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from tw_ai_agents.agents.base_agent import BaseAgent, make_supervisor_node
from tw_ai_agents.react_agent.state import State

llm = ChatOpenAI(model="gpt-4o")


class ZendeskAgentWithTools(BaseAgent):
    def __init__(self, system_prompt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_name = "ZendeskSearcher"
        self.system_prompt = system_prompt

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

    def get_graph(self):

        tool_list = [self.get_comments, self.get_ticket_info]
        agent = create_react_agent(
            llm, tools=tool_list, prompt=self.system_prompt
        )
        return agent



if __name__ == "__main__":

    system_prompt = """You are a helpful assistant that can use tools to answer questions about Zendesk tickets.
Your goal is to provide accurate information by using the available tools to search for and retrieve ticket information.
"""

    zendesk_agent = ZendeskAgentWithTools(system_prompt=system_prompt)
    graph = zendesk_agent.get_graph()

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
