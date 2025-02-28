import sqlite3
import os
from sample_tickets import SAMPLE_TICKETS

# Create the database directory if it doesn't exist
os.makedirs('mock/data', exist_ok=True)

# Connect to the SQLite database (will create it if it doesn't exist)
conn = sqlite3.connect('mock/data/tickets.db')

cursor = conn.cursor()

cursor.execute('DROP TABLE IF EXISTS tickets')

cursor.execute('''
CREATE TABLE IF NOT EXISTS tickets (
    ticketId TEXT PRIMARY KEY,
    ticketContents TEXT
)
''')

cursor.executemany(
    'INSERT OR REPLACE INTO tickets (ticketId, ticketContents) VALUES (?, ?)', SAMPLE_TICKETS)

conn.commit()
conn.close()

print("Database seeded successfully with ticket data.")
