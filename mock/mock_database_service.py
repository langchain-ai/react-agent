import sqlite3
import os
from typing import List, Tuple, Optional
from mock.sample_tickets import SAMPLE_TICKETS


class MockDatabaseService:
    """Service to interact with the mock ticket database."""

    def __init__(self, db_path: str = 'mock/data/tickets.db'):
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
            'SELECT ticketContents FROM tickets WHERE ticketId = ?', (ticket_id,))
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

        cursor.execute('SELECT ticketId, ticketContents FROM tickets')
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

        cursor.execute('UPDATE tickets SET ticketContents = ? WHERE ticketId = ?',
                       (contents, ticket_id))
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
            cursor.execute('INSERT INTO tickets (ticketId, ticketContents) VALUES (?, ?)',
                           (ticket_id, contents))
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

        cursor.execute('DELETE FROM tickets WHERE ticketId = ?', (ticket_id,))
        success = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return success

    def reseed(self) -> None:
        """Reset the database with the original sample data."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Drop the existing table if it exists
        cursor.execute('DROP TABLE IF EXISTS tickets')

        # Create the tickets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            ticketId TEXT PRIMARY KEY,
            ticketContents TEXT
        )
        ''')

        # Insert sample data
        cursor.executemany(
            'INSERT OR REPLACE INTO tickets (ticketId, ticketContents) VALUES (?, ?)', SAMPLE_TICKETS)

        # Commit changes and close connection
        conn.commit()
        conn.close()
