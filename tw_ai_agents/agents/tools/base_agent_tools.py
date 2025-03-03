from abc import abstractmethod

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")


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
