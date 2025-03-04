from abc import abstractmethod

from tw_ai_agents.agents.llm_models_loader import get_llm_model

llm = get_llm_model()


class BaseAgentWithTools:
    def __init__(self, system_prompt: str, node_name: str, description: str):
        self.node_name = node_name
        self.system_prompt = system_prompt
        self.description = description

    @abstractmethod
    def get_tools(self):
        raise NotImplementedError(
            "This method should be implemented by the subclass"
        )
