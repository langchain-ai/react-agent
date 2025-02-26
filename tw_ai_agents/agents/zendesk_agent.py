from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import StateGraph

from tw_ai_agents.agents.base_agent import BaseAgent, make_supervisor_node
from tw_ai_agents.react_agent.state import State

llm = ChatOpenAI(model="gpt-4o")


class ZendeskAgentWithTools(BaseAgent):
    def __init__(self, system_prompt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_name = "ZendeskSearcher"
        self.system_prompt = system_prompt
        self.system_prompt = "You are a supervisor tasked with managing a conversation between the"
        f" following workers: Booh. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."

    @tool
    def get_ticket_info(self, ticket_id: str):
        """
        Tool to get the ticket info from a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get information for
        :return: The ticket information
        """
        return f"This is the ticket {ticket_id}. Infos: very nice ticket"

    @tool
    def get_comments(self, ticket_id: str):
        """
        Tool to get the comments from a zendesk ticket
        :params:
            ticket_id: The ID of the ticket to get comments for
        :return: The comments from the ticket
        """
        return f"This is are the comments from the ticket {ticket_id}: [comment1, comment2, comment3]"

    def get_compiled_graph(self):
        research_supervisor_node = make_supervisor_node(
            llm,
            members=["get_comments", "get_ticket"],
            system_prompt=self.system_prompt,
        )

        research_builder = StateGraph(State)

        research_builder.add_node(self.node_name, research_supervisor_node)
        research_builder.add_node(
            "get_comments",
            self.get_node(
                llm,
                tool_list=[self.get_comments],
                name="get_comments",
                target_node_name=self.node_name,
            ),
        )
        research_builder.add_node(
            "get_ticket",
            self.get_node(
                llm,
                tool_list=[self.get_ticket_info],
                name="get_ticket",
                target_node_name=self.node_name,
            ),
        )

        research_builder.add_edge(START, self.node_name)
        research_graph = research_builder.compile()
        return research_graph


if __name__ == "__main__":

    zendesk_agent = ZendeskAgentWithTools(system_prompt="ciao")
    graph = zendesk_agent.get_compiled_graph()

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
