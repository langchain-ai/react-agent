from langchain_core.tools import tool

from tw_ai_agents.agents.llm_models_loader import get_llm_model
from tw_ai_agents.agents.tools.base_agent_tools import BaseAgentWithTools
from tw_ai_agents.mock.mock_database_service import MockDatabaseService

llm = get_llm_model()


class ReadERPInfoTool(BaseAgentWithTools):
    def __init__(self, db_service=None):
        node_name = "read_erp_info"
        base_system_prompt = """You are a helpful assistant that can use tools to read information from the ERP system.
        Your goal is retrieve information from the ERP system and update it with new information.
        If you can't find the requested information using the tools, ask to the Human Agent through the real_human_agent_execute_actions tool.
        """
        description = "You are a helpful assistant that can use tools to read information from the ERP system."
        # base_system_prompt = hub.pull(
        #     "zendesk_info_searcher-system_prompt"
        # ).content
        # description = hub.pull("zendesk_info_searcher-description").content

        # Use the provided database service or default to MockDatabaseService
        self.db_service = (
            db_service if db_service is not None else MockDatabaseService()
        )

        super().__init__(
            system_prompt=base_system_prompt,
            node_name=node_name,
            description=description,
        )

    @tool(parse_docstring=True)
    @staticmethod
    def read_address(email_id: str):
        """Tool to read the address of a customer, given their email ID.

        Args:
            email_id: The email ID of the customer to read the address for.

        Returns:
            The address of the customer.
        """
        db_service = MockDatabaseService()
        address = db_service.get_customer_address(email_id)
        if address:
            return address
        return f"No address found for customer {email_id}"

    @tool(parse_docstring=True)
    @staticmethod
    def read_document_id(email_id: str):
        """Tool to read the document ID of a customer, given their email ID.

        Args:
            email_id: The email ID of the customer to read the document ID for.

        Returns:
            The document ID of the customer.
        """
        db_service = MockDatabaseService()
        document_id = db_service.get_customer_document_id(email_id)
        if document_id:
            return f"Document ID {document_id} found for customer {email_id}"
        return f"No document ID found for customer {email_id}"

    def get_tools(self):
        return [
            self.read_address,
            self.read_document_id,
        ]
