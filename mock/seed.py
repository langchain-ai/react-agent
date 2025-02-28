import sqlite3
import os
from mock.sample_data import SAMPLE_ORDERS, SAMPLE_CUSTOMERS

os.makedirs('mock/data', exist_ok=True)

conn = sqlite3.connect('mock/data/erp_info.db')
cursor = conn.cursor()

cursor.execute('DROP TABLE IF EXISTS orders')
cursor.execute('DROP TABLE IF EXISTS customers')

# Create customers table for ERP info (email, address, document_id)
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    email_id TEXT PRIMARY KEY,
    address TEXT,
    document_id TEXT
)
''')

# Create orders table with a foreign key to customers
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    order_contents TEXT,
    status TEXT,
    price REAL,
    customer_email TEXT,
    FOREIGN KEY (customer_email) REFERENCES customers (email_id)
)
''')

# Insert customer data
for customer in SAMPLE_CUSTOMERS:
    cursor.execute(
        "INSERT INTO customers (email_id, address, document_id) VALUES (?, ?, ?)",
        customer
    )

# Insert order data
for order in SAMPLE_ORDERS:
    cursor.execute(
        "INSERT INTO orders (order_id, order_contents, status, price, customer_email) VALUES (?, ?, ?, ?, ?)",
        order
    )

conn.commit()
conn.close()

print("Database seeded successfully with order data and customer ERP information.")
