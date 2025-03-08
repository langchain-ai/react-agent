from tw_ai_agents.config_handler.constants import TW_SUPERVISOR_NAME
from tw_ai_agents.agents.message_types.base_message_type import State

ROUTER_NODE_NAME = "router_node"


def initial_router_node(state: State) -> State:
    """
    Initial entry point for the graph.
    Just a placeholder to start the graph and to be starting point of the sorting node.
    """
    return state


def initial_router_sorting_condition(state: State) -> str:
    """
    Sorting node for the router.
    Decides if to go to the supervisor or the a specific sub-agent based on the metadata content.
    Default is to go to the supervisor, so that the first time the supervisor is called.
    """
    next_node = state.metadata.get("next_node", TW_SUPERVISOR_NAME)
    return next_node
