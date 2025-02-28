import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from mock.sample_tickets import (
    SAMPLE_ADDRESSES,
    SAMPLE_COMMENTS,
    SAMPLE_TICKETS,
)


class MockDatabaseService:
    """Service to interact with the mock ticket database."""

    def __init__(self, db_path: str = "mock/data/tickets.db"):
        """Initialize the database service.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the database.

        Returns:
            A SQLite connection object
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def update_customer_address(
        self, email_id: str, new_address: str
    ) -> Optional[str]:
        """Update the address of a customer, given their email ID and the new address.

        Args:
            email_id: The email ID of the customer to update the address for
            new_address: The new address to update the customer with
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE customers SET address = ? WHERE email = ?",
            (new_address, email_id),
        )
        conn.commit()
        conn.close()

    def update_customer_document_id(
        self, email_id: str, document_id: str
    ) -> Optional[str]:
        """Update the document ID of a customer, given their email ID and the new document ID.

        Args:
            email_id: The email ID of the customer to update the document ID for
            document_id: The new document ID to update the customer with
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE customers SET documentId = ? WHERE email = ?",
            (document_id, email_id),
        )
        conn.commit()
        conn.close()

    def get_ticket(self, ticket_id: str) -> Optional[str]:
        """Get ticket contents by ticket ID.

        Args:
            ticket_id: The ID of the ticket to retrieve

        Returns:
            The ticket contents or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT ticketContents FROM tickets WHERE ticketId = ?",
            (ticket_id,),
        )
        result = cursor.fetchone()

        conn.close()
        return result[0] if result else None

    def get_ticket_comments(self, ticket_id: str) -> List[str]:
        """Get comments for a specific ticket.

        Args:
            ticket_id: The ID of the ticket to retrieve comments for

        Returns:
            List of comments for the ticket
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT commentText FROM comments WHERE ticketId = ? ORDER BY commentId",
            (ticket_id,),
        )
        results = cursor.fetchall()

        conn.close()
        return [comment[0] for comment in results] if results else []

    def get_ticket_address(self, ticket_id: str) -> Optional[str]:
        """Get the address associated with a ticket.

        Args:
            ticket_id: The ID of the ticket to retrieve the address for

        Returns:
            The address or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT address FROM addresses WHERE ticketId = ?", (ticket_id,)
        )
        result = cursor.fetchone()

        conn.close()
        return result[0] if result else None

    def get_all_tickets(self) -> List[Tuple[str, str]]:
        """Get all tickets in the database.

        Returns:
            List of tuples containing (ticketId, ticketContents)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT ticketId, ticketContents FROM tickets")
        results = cursor.fetchall()

        conn.close()
        return results

    def update_ticket(self, ticket_id: str, contents: str) -> bool:
        """Update the contents of a ticket.

        Args:
            ticket_id: The ID of the ticket to update
            contents: The new contents for the ticket

        Returns:
            True if the update was successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE tickets SET ticketContents = ? WHERE ticketId = ?",
            (contents, ticket_id),
        )
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return success

    def add_ticket(self, ticket_id: str, contents: str) -> bool:
        """Add a new ticket to the database.

        Args:
            ticket_id: The ID for the new ticket
            contents: The contents for the new ticket

        Returns:
            True if the insertion was successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO tickets (ticketId, ticketContents) VALUES (?, ?)",
                (ticket_id, contents),
            )
            success = True
        except sqlite3.IntegrityError:
            # Ticket ID already exists
            success = False

        conn.commit()
        conn.close()
        return success

    def delete_ticket(self, ticket_id: str) -> bool:
        """Delete a ticket from the database.

        Args:
            ticket_id: The ID of the ticket to delete

        Returns:
            True if the deletion was successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM tickets WHERE ticketId = ?", (ticket_id,))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return success

    def reseed(self) -> None:
        """Reset the database with the original sample data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Drop the existing tables if they exist
        cursor.execute("DROP TABLE IF EXISTS tickets")
        cursor.execute("DROP TABLE IF EXISTS comments")
        cursor.execute("DROP TABLE IF EXISTS addresses")

        # Create the tickets table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS tickets (
            ticketId TEXT PRIMARY KEY,
            ticketContents TEXT
        )
        """
        )

        # Create the comments table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS comments (
            commentId INTEGER PRIMARY KEY AUTOINCREMENT,
            ticketId TEXT,
            commentText TEXT,
            FOREIGN KEY (ticketId) REFERENCES tickets (ticketId)
        )
        """
        )

        # Create the addresses table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS addresses (
            addressId INTEGER PRIMARY KEY AUTOINCREMENT,
            ticketId TEXT UNIQUE,
            address TEXT,
            FOREIGN KEY (ticketId) REFERENCES tickets (ticketId)
        )
        """
        )

        # Insert sample data
        cursor.executemany(
            "INSERT OR REPLACE INTO tickets (ticketId, ticketContents) VALUES (?, ?)",
            SAMPLE_TICKETS,
        )

        # Insert sample comments
        for ticket_id, comment_text in SAMPLE_COMMENTS:
            cursor.execute(
                "INSERT INTO comments (ticketId, commentText) VALUES (?, ?)",
                (ticket_id, comment_text),
            )

        # Insert sample addresses
        cursor.executemany(
            "INSERT OR REPLACE INTO addresses (ticketId, address) VALUES (?, ?)",
            SAMPLE_ADDRESSES,
        )

        # Commit changes and close connection
        conn.commit()
        conn.close()
