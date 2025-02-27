from dataclasses import dataclass
from typing import List

from pydantic import BaseModel

from tw_ai_agents.tools.crm_connector_tools.zendesk_agent_tools import (
    ZendeskAgentWithTools,
)

AGENT_LIST = {
    "ID_sdkjcnsdjhcnd": ZendeskAgentWithTools,
}


class AgentListElement(BaseModel):
    id: str
    title: str
    description: str


class ActionListReturnModel(BaseModel):
    agents: List[AgentListElement]


def get_agent_list():
    all_agents_list = []

    for key, value in AGENT_LIST.items():
        all_agents_list.append(
            AgentListElement(
                id=key, title=value().node_name, description=value().description
            )
        )

    return ActionListReturnModel(agents=all_agents_list)
