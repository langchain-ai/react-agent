from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from pydantic import BaseModel
from agents.market_research_bot import market_research_chain
from agents.domain_generator import generate_domain_suggestions
from agents.domain_name_scoring_bot import evaluate_domain_set


# Define our state
class State(TypedDict):
    initialized: bool
    market_trends: dict
    generated_domains: list
    scored_domains: list
    domains: list
    available_domains: list
    iteration: int

# Define the nodes of our graph
def initialize(state: State = None) -> State:
    return State(
        initialized=True,
        market_trends={},
        generated_domains=[],
        scored_domains=[],
        domains=[],
        available_domains=[],
        iteration=0
    )

def market_trends_bot(state: State):
    print('generating market trends..')
    state['market_trends'] = market_research_chain.invoke({})
    print(state['market_trends'])
    return state

def domain_name_generator_bot(state: State):
    print("cooking up some delicious domains...")
    state['generated_domains'] = generate_domain_suggestions(state['market_trends'])
    print(state['generated_domains'])
    state['iteration'] += 1
    return state

def name_scoring_bot(state: State):
    print("scoring the domains...")
    state['scored_domains'] = evaluate_domain_set(state['generated_domains'])
    print(state['scored_domains'])
    state['domains'] = [eval.domain for eval in state['scored_domains'].evaluations]
    print(state['domains'])
    return state

def check_availability(state: State):
    # TODO Will call domain name availability API
    print("Checking domain availability...")
    state['available_domains'] = state['domains']  # Assuming all domains are available for now
    return state

def route(state: State) -> str:
    if state['available_domains']:
        return "process_available_domains"
    elif state['iteration'] < 3:  # Limit to 3 attempts
        return "domain_name_generator_bot"
    else:
        return "end_process"

def process_available_domains(state: State) -> dict:
    print("Processing available domains...")
    return state

def end_process(state: State) -> dict:
    print("Ending process...")
    return {"final_message": "No available domains found after multiple attempts."}

# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("initialize", initialize)
workflow.add_node("market_trends_bot", market_trends_bot)
workflow.add_node("domain_name_generator_bot", domain_name_generator_bot)
workflow.add_node("name_scoring_bot", name_scoring_bot)
workflow.add_node("check_availability", check_availability)
workflow.add_node("process_available_domains", process_available_domains)
workflow.add_node("end_process", end_process)

# Add edges
workflow.add_edge("initialize", "market_trends_bot")
workflow.add_edge("market_trends_bot", "domain_name_generator_bot")
workflow.add_edge("domain_name_generator_bot", "name_scoring_bot")
workflow.add_edge("name_scoring_bot", "check_availability")
workflow.add_conditional_edges(
    'check_availability',
    route,
    {
        "process_available_domains": 'process_available_domains',
        "domain_name_generator_bot": 'domain_name_generator_bot',
        "end_process": 'end_process'
    }
)
workflow.add_edge("process_available_domains", END)
workflow.add_edge("end_process", END)

# Set the entrypoint
workflow.set_entry_point("initialize")

config = {"configurable": {"thread_id": "1"}}

# Compile the graph
graph = workflow.compile()

# def run_domain_research():
#     initial_state = initialize()
#     for output in graph.stream(initial_state):
#         if "intermediate_steps" in output:
#             print(f"Step: {output['intermediate_steps'][-1][0]}")
#             print(f"Output: {output['intermediate_steps'][-1][1]}")
#         else:
#             print("Final output:", output)
    
#     return output

# # Add this new code block at the end of the file
# try:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     pass
