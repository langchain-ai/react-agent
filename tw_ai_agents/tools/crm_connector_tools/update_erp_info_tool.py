import copy
from typing import List, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from mock.mock_database_service import MockDatabaseService
from tw_ai_agents.tools.base_agent_tools import BaseAgentWithTools

llm = ChatOpenAI(model="gpt-4o")


class UpdateERPInfoTool(BaseAgentWithTools):
    def __init__(self, db_service=None):
        node_name = "update_erp_info"
        base_system_prompt = """You are a helpful assistant that can use tools to update information in the ERP system.
        Your goal is retrieve information from the ERP system and update it with new information.
        """
        description = "You are a helpful assistant that can use tools to update information in the ERP system."
        # base_system_prompt = hub.pull(
        #     "zendesk_info_searcher-system_prompt"
        # ).content
        # description = hub.pull("zendesk_info_searcher-description").content


        super().__init__(
            system_prompt=base_system_prompt,
            node_name=node_name,
            description=description,
        )

    @tool
    @staticmethod
    def update_address(email_id: str, new_address: str):
        """
        Tool to update the address of a customer, given their email ID and the new address.
        :params:
            email_id: The email ID of the customer to update the address for
            new_address: The new address to update the customer with
        :return: The updated address
        """
        db_service = MockDatabaseService()
        updated_address = db_service.update_customer_address(
            email_id, new_address
        )
        if updated_address:
            return updated_address
        return f"Address {new_address} updated for customer {email_id}"

    @tool
    @staticmethod
    def update_document_id(email_id: str, document_id: str):
        """
        Tool to update the document ID of a customer, given their email ID and the new document ID.
        :params:
            email_id: The email ID of the customer to update the document ID for
            document_id: The new document ID to update the customer with
        :return: The updated document ID
        """
        db_service = MockDatabaseService()
        updated_document_id = db_service.update_customer_document_id(
            email_id, document_id
        )
        if updated_document_id:
            return f"Document ID {document_id} updated for customer {email_id}"
        return f"No document ID found for customer {email_id}"

    def get_tools(self):
        return [
            self.update_address,
            self.update_document_id,
        ]
