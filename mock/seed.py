import sqlite3
import os
from sample_tickets import SAMPLE_TICKETS, SAMPLE_COMMENTS, SAMPLE_ADDRESSES

# Create the database directory if it doesn't exist
os.makedirs('mock/data', exist_ok=True)

# Connect to the SQLite database (will create it if it doesn't exist)
conn = sqlite3.connect('mock/data/tickets.db')

cursor = conn.cursor()

# Drop existing tables
cursor.execute('DROP TABLE IF EXISTS tickets')
cursor.execute('DROP TABLE IF EXISTS comments')
cursor.execute('DROP TABLE IF EXISTS addresses')

# Create tickets table
cursor.execute('''
CREATE TABLE IF NOT EXISTS tickets (
    ticketId TEXT PRIMARY KEY,
    ticketContents TEXT
)
''')

# Create comments table
cursor.execute('''
CREATE TABLE IF NOT EXISTS comments (
    commentId INTEGER PRIMARY KEY AUTOINCREMENT,
    ticketId TEXT,
    commentText TEXT,
    FOREIGN KEY (ticketId) REFERENCES tickets (ticketId)
)
''')

# Create addresses table
cursor.execute('''
CREATE TABLE IF NOT EXISTS addresses (
    addressId INTEGER PRIMARY KEY AUTOINCREMENT,
    ticketId TEXT UNIQUE,
    address TEXT,
    FOREIGN KEY (ticketId) REFERENCES tickets (ticketId)
)
''')

# Insert sample tickets
cursor.executemany(
    'INSERT OR REPLACE INTO tickets (ticketId, ticketContents) VALUES (?, ?)', SAMPLE_TICKETS)

# Insert sample comments
for ticket_id, comment_text in SAMPLE_COMMENTS:
    cursor.execute(
        'INSERT INTO comments (ticketId, commentText) VALUES (?, ?)',
        (ticket_id, comment_text)
    )

# Insert sample addresses
cursor.executemany(
    'INSERT OR REPLACE INTO addresses (ticketId, address) VALUES (?, ?)', SAMPLE_ADDRESSES)

conn.commit()
conn.close()

print("Database seeded successfully with ticket data, comments, and addresses.")
