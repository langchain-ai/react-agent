from abc import abstractmethod
from typing import Callable, List, Optional
from typing import Union, Any

from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt.chat_agent_executor import StateSchemaType

from tw_ai_agents.agents.case_agent import CaseAgent
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


class ToolAgent(CaseAgent):
    def __init__(
        self,
        agents: List[Union[CompiledStateGraph, "TWSupervisor"]],
        tools: List[Union[Callable, BaseTool]],
        model: LanguageModelLike,
        prompt: Optional[str],
        state_schema: StateSchemaType,
        name: str,
        description: str,
        memory=None,
        dependant_agents: Optional[List[Any]] = None,
    ):
        """
        Initialize a tool-agent which handles a specific case.
        The only difference with the CaseAgent is that it cannot return to the supervisor,
            but can only go to END and thus return to the parent CaseAgent.

        :params
            agents: List of agents or supervisors to manage
            tools: Tools to use for the agent
            model: Language model to use for the agent
            prompt: Prompt to use for the agent
            state_schema: Schema for the agent state
            name: Name of this agent for identification
            description: Description of this agent's capabilities
            memory: Memory store for the agent
            dependant_agents: List of AskUserTool instances this agent depends on
        :return
            None
        """
        super().__init__(
            agents,
            tools,
            model,
            prompt,
            state_schema,
            name,
            description,
            memory,
            dependant_agents,
        )
        self.is_case_agent = False
