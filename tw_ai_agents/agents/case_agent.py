from typing import List, Any, TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from tw_ai_agents.agents.base_agent import BaseAgent, make_supervisor_node
from tw_ai_agents.agents.base_agent import State

llm = ChatOpenAI(model="gpt-4o")


class BaseCaseSpecificAgent(BaseAgent):
    def __init__(
        self,
        node_name: str,
        system_prompt: str,
        sub_agents: List[Any],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.node_name = node_name
        self.system_prompt = system_prompt
        self.sub_agents = sub_agents

    def make_supervisor_node(self, members: list[str], system_prompt: str):
        options = [END] + members

        class Router(TypedDict):
            """Worker to route to next. If no workers needed, route to FINISH."""

            next: str  # Will be one of the options

        def supervisor_node(state: State) -> Command[str]:
            """An LLM-based router."""
            messages = [
                {"role": "system", "content": system_prompt},
            ] + state["messages"]
            response = llm.with_structured_output(Router).invoke(messages)
            goto = response["next"]
            if goto == "FINISH":
                goto = END

            return Command(goto=goto, update={"next": goto})

        return supervisor_node

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
        research_graph = research_builder.compile(debug=True)
        return research_graph


if __name__ == "__main__":

    system_prompt = """
    You are a supervisor agent which should solve a ticket by using some tool.
    You can use the following workers: get_comments, get_ticket.
    Given the following user request, respond with the worker to act next.
    Each worker will perform a task and respond with their results and status.
    When finished, respond with FINISH.
    """

    zendesk_agent = ZendeskAgentWithTools(system_prompt=system_prompt)
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
