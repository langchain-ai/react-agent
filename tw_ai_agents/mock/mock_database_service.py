import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from tw_ai_agents.config_handler.absolute_dirs import get_mock_db_folder


class MockDatabaseService:
    def __init__(self, db_path: str = get_mock_db_folder() / "erp_info.db"):
        self.db_path = db_path

    def _get_connection(self):
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def get_customer_address(self, email_id: str) -> Optional[str]:
        """
        Get the address of a customer by their email ID.

        Args:
            email_id: The email ID of the customer

        Returns:
            The customer's address or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT address FROM customers WHERE email_id = ?", (email_id,)
        )
        result = cursor.fetchone()

        conn.close()
        return result[0] if result else None

    def get_customer_document_id(self, email_id: str) -> Optional[str]:
        """
        Get the document ID of a customer by their email ID.

        Args:
            email_id: The email ID of the customer

        Returns:
            The customer's document ID or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT document_id FROM customers WHERE email_id = ?", (email_id,)
        )
        result = cursor.fetchone()

        conn.close()
        return result[0] if result else None

    def update_customer_address(self, email_id: str, new_address: str) -> bool:
        """
        Update the address of a customer by their email ID.

        Args:
            email_id: The email ID of the customer
            new_address: The new address to set

        Returns:
            True if the update was successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE customers SET address = ? WHERE email_id = ?",
            (new_address, email_id),
        )
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return success

    def update_customer_document_id(
        self, email_id: str, document_id: str
    ) -> bool:
        """
        Update the document ID of a customer by their email ID.

        Args:
            email_id: The email ID of the customer
            document_id: The new document ID to set

        Returns:
            True if the update was successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE customers SET document_id = ? WHERE email_id = ?",
            (document_id, email_id),
        )
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return success
