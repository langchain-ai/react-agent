import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from datetime import datetime 

# Load environment variables
load_dotenv()

class Company(BaseModel):
    name: str = Field(description="The name of the company")
    # TODO category should be a list of categories
    # TODO should this be a list of categories?
    category: str = Field(description="The category the company belongs to (SaaS, B2B, enterprise)")
    description: str = Field(description="The description of what the company does")
    # TODO keyword should be a list of keywords
    keyword: str = Field(description="The main keywords associated with the company")
    domain: str = Field(description="The company's domain name Example akinator.com")

class CompanySet(BaseModel):
    companies: List[Company] = Field(description="A list of all companies in the set")

parser = PydanticOutputParser(pydantic_object=CompanySet)

# TODO instead of this prompt, we have to crawl the pages or use some other method to get the information
template = """
You are a knowledgeable assistant with extensive information about startups and tech companies.
Your task is to generate a list of at least 20 notable companies that have been featured on 
Y-Combinator, Crunchbase, or Product Hunt. For each company, provide the following details:

1. Name: The official name of the company.
2. Category: Classify the company using one of the following tags:

        - Analytics, B2B
        - Engineering, Product and Design, B2B
        - Finance and Accounting, B2B
        - Human Resources, B2B
        - Infrastructure, B2B
        - Legal, B2B
        - Marketing, B2B
        - Office Management, B2B
        - Operations, B2B
        - Productivity, B2B
        - Recruiting and Talent, B2B
        - Retail, B2B
        - Sales, B2B
        - Security, B2B
        - Supply Chain and Logistics, B2B
        - Education
        - Asset Management, Fintech
        - Banking and Exchange, Fintech
        - Consumer Finance, Fintech
        - Credit and Lending, Fintech
        - Insurance, Fintech
        - Payments, Fintech
        - Apparel and Cosmetics, Consumer
        - Consumer Electronics, Consumer
        - Content, Consumer
        - Food and Beverage, Consumer
        - Gaming, Consumer
        - Home and Personal, Consumer
        - Job and Career Services, Consumer
        - Social, Consumer
        - Transportation Services, Consumer
        - Travel, Leisure and Tourism, Consumer
        - Virtual and Augmented Reality, Consumer
        - Consumer Health and Wellness, Healthcare
        - Diagnostics, Healthcare
        - Drug Discovery and Delivery, Healthcare
        - Healthcare IT, Healthcare
        - Healthcare Services, Healthcare
        - Industrial Bio, Healthcare
        - Medical Devices, Healthcare
        - Therapeutics, Healthcare
        - Construction, Real Estate and Construction
        - Real Estate, Real Estate and Construction
        - Agriculture, Industrials
        - Automotive, Industrials
        - Aviation and Space, Industrials
        - Climate, Industrials
        - Drones, Industrials
        - Energy, Industrials
        - Engineering and Robotics, Industrials
        - Government, Industrials
        - Unspecified, Industrials

3. Description: A brief (1-2 sentences) description of what the startup does or the problem it solves.
4. Keyword: A single word or short phrase that best represents the company's main focus or industry.
5. Domain: The company's website domain name.

Ensure that your list includes a diverse range of companies, including some well-known ones and some that 
might be less familiar but innovative. Try to have a mix of companies from different sectors and with 
different target markets.

Remember, you should rely on your existing knowledge and not perform any real-time web searches.
If you're unsure about any specific details, you can provide plausible information based on your
understanding of similar companies in the same sector.

{format_instructions}

Please provide a well-formatted list of 10 companies with the requested information.
"""

prompt = PromptTemplate(
    template=template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Get the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize the language model with the API key
llm = ChatOpenAI(model='gpt-4o', temperature=0.7, openai_api_key=api_key)

market_research_chain = prompt | llm | StrOutputParser() | parser

def format_company_list(companies):
    return "\n".join([f"- {company.name} ({company.domain}): {company.keyword}" for company in companies])

def generate_domain_suggestions(company_set):
    company_list = format_company_list(company_set.companies)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    domain_generator_chain = domain_prompt | llm | domain_parser
    result = domain_generator_chain.invoke({"company_list": company_list, "system_time": current_time})
    return result