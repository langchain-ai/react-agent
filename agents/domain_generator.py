import datetime
import os
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

class DomainSuggestion(BaseModel):
    domain: str = Field(description="The suggested domain name")
    explanation: str = Field(description="A brief explanation of why this domain is suitable")

class DomainSuggestionSet(BaseModel):
    suggestions: List[DomainSuggestion] = Field(description="A list of domain suggestions")

domain_parser = PydanticOutputParser(pydantic_object=DomainSuggestionSet)

domain_template = """
You are an expert domain name investment expert with 30 years of experience and expertise. 
Your task is to generate high-quality, brandable domain names based on the information provided
about existing successful companies.

A good domain name has the following qualities:
- Short (preferably less than 10 characters, but up to 15 is acceptable)
- Brandable and unique
- Pronounceable
- Memorable
- Relevant to the business or industry

Here are some examples of good domain names: laminar.ai, Rippling.com, Amplitude.com, Zenefits.com

Now, based on the following list of successful companies and their domains:

{company_list}

Generate 10 high-quality domain names for new AI SaaS B2B Enterprise startups.
Your suggestions should be inspired by the trends and patterns you see in the provided company list,
but should be entirely new and unique names, not variations of existing ones.

Only generate .com and .ai TLDs.

For each suggested domain, provide a brief explanation of why it's suitable and how it
relates to the AI SaaS B2B Enterprise space.

System time: {system_time}

{format_instructions}
"""

domain_prompt = PromptTemplate(
    template=domain_template,
    input_variables=["company_list", "system_time"],
    partial_variables={"format_instructions": domain_parser.get_format_instructions()}
)

llm = ChatOpenAI(model='gpt-4', temperature=0.7, openai_api_key=api_key)

def format_company_list(companies):
    return "\n".join([f"- {company.name} ({company.domain}): {company.keyword}" for company in companies])

def generate_domain_suggestions(company_set):
    company_list = format_company_list(company_set.companies)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    domain_generator_chain = domain_prompt | llm | domain_parser
    result = domain_generator_chain.invoke({"company_list": company_list, "system_time": current_time})

    return result

def extract_domain_names(domain_evaluations):
    return [eval.domain for eval in domain_evaluations.evaluations]

@tool
def get_domain_suggestions(company_set):
    """Generate domain suggestions based on a list of companies."""
    return generate_domain_suggestions(company_set)
