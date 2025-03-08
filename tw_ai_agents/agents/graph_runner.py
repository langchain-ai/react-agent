from typing import Dict, List

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from tw_ai_agents.agents.case_agent import SUBAGENT_TOOL_NAME_PREFIX
from tw_ai_agents.agents.message_types.base_message_type import (
    InterruptBaseModel,
    State,
    ToolMessageInfo,
)
from tw_ai_agents.agents.tools.human_tools import COMPLETE_HANDOFF_STRING


class GraphRunner:
    def __init__(self, graph: CompiledStateGraph):
        self.graph = graph

    def run_supervisor(self, state: State, config) -> Dict:
        """Run the supervisor agent system with metadata tracking synchronously.

        :params
            state: The initial state containing messages and other context
            config: Configuration for the graph execution
        :return
            The updated state with results and metadata about the execution
        """

        result = None
        last_message = None
        while last_message is None or isinstance(last_message, ToolMessage):
            # While loop because when the supervisor decides to call a sub-agent, it will return a tool message.
            # We need to call the graph again and the router will sort the case to the correct sub-agent.

            # We only want the latest result
            for chunk in self.graph.stream(state, config=config):
                # We only want the latest result
                result = {"messages": [], "metadata": {}, "tools_called": []}
                result = self._process_output_chunk(chunk, result)

                last_message = result["messages"][-1]

        if result is None:
            raise ValueError("No output from the graph execution")

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["tool_calls"] = self._extract_tool_calls(
            result["tools_called"]
        )
        return result

    async def arun_supervisor(self, state: State, config) -> Dict:
        """Run the supervisor agent system with metadata tracking asynchronously.

        :params
            state: The initial state containing messages and other context
            config: Configuration for the graph execution
        :return
            The updated state with results and metadata about the execution
        """

        result = None
        last_message = None
        while last_message is None or isinstance(last_message, ToolMessage):
            # While loop because when the supervisor decides to call a sub-agent, it will return a tool message.
            # We need to call the graph again and the router will sort the case to the correct sub-agent.

            async for chunk in self.graph.astream(state, config=config):
                # We only want the latest result
                result = {"messages": [], "metadata": {}, "tools_called": []}
                result = self._process_output_chunk(chunk, result)

                last_message = result["messages"][-1]

        if result is None:
            raise ValueError("No output from the graph execution")

        if "metadata" not in result:
            result["metadata"] = {}

        result["metadata"]["tool_calls"] = self._extract_tool_calls(
            result["tools_called"]
        )
        return result

    def _process_output_chunk(self, chunk, result):
        """Process a chunk of output from the graph execution and update the result dictionary.

        :params
            chunk: Output chunk from graph execution
            result: Dictionary containing messages and metadata to update
        :return
            Updated result dictionary
        """
        if "__interrupt__" in chunk:
            # human interruption
            interrupt_content = InterruptBaseModel.parse_obj(
                chunk["__interrupt__"][0].value
            )

            new_message = AIMessage(content=interrupt_content.user_message)

            metadata = {
                "ns": chunk["__interrupt__"][0].ns,
                "target_entity": interrupt_content.destination,
                "complete_handoff": COMPLETE_HANDOFF_STRING
                == new_message.content, # Legacy, will be removed
                "agent_message_mode": interrupt_content.agent_message_mode.value,
            }
            result["messages"].append(new_message)
            result["metadata"].update(metadata)
            result["tools_called"].extend(interrupt_content.tools_called)
        else:
            for key, values in chunk.items():
                # we always want the last
                if isinstance(values, list):
                    for value in values:
                        result["messages"].extend(value.get("messages", []))
                        result["tools_called"].extend(
                            value.get("tools_called", [])
                        )
                        result["metadata"].update(value.get("metadata", {}))
                else:
                    result["messages"].extend(values.get("messages", []))
                    result["tools_called"].extend(
                        values.get("tools_called", [])
                    )
                    result["metadata"].update(values.get("metadata", {}))
        return result

    def _extract_tool_calls(
        self, tool_message_infos: List[ToolMessageInfo]
    ) -> List[Dict]:
        """Extract tool calls from a list of ToolMessageInfo objects.

        :params tool_message_infos: List of ToolMessageInfo objects
        :return List of dictionaries containing tool call information
        """
        tool_calls = []
        for tool_message_info in tool_message_infos:
            if not isinstance(tool_message_info, ToolMessageInfo):
                raise ValueError(
                    f"Expected ToolMessageInfo object, got {type(tool_message_info)}"
                )
            a = 1
            # Filter out our sub-agents from the tool call list.
            # The ones whose name starts with SUBAGENT_TOOL_NAME_PREFIX
            if (
                tool_message_info.name is not None
                and not tool_message_info.name.startswith(
                    SUBAGENT_TOOL_NAME_PREFIX
                )
            ):
                a = 1
                tool_calls.append(
                    {
                        "tool_name": tool_message_info.name,
                        "tool_input": tool_message_info.parameters,
                        "tool_id": tool_message_info.tool_call_id,
                    }
                )
        return tool_calls
