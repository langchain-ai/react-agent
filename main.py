import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents.domain_research_graph import run_domain_research
import openai

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Set the API key for the OpenAI client
openai.api_key = api_key

# Initialize the language model
llm = ChatOpenAI(model='gpt-4', temperature=0.7, openai_api_key=api_key)

# Example usage
if __name__ == "__main__":
    result = run_domain_research()
    print("Final result:", result)
