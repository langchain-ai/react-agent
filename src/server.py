"""Server for the customer service supervisor agent API.

This module provides a server for the customer service supervisor agent API.
"""

import os
import uvicorn
from dotenv import load_dotenv

from supervisor_agent.api import app

# Load environment variables
load_dotenv()

# Get port from environment variable or use default
PORT = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    ) 