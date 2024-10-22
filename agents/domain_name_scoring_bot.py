from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import Tool
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

class DomainScore(BaseModel):
    memorability: int = Field(description="Score for memorability (1-10)", ge=1, le=10)
    pronounceability: int = Field(description="Score for pronounceability (1-10)", ge=1, le=10)
    length: int = Field(description="Score for length (1-10)", ge=1, le=10)
    brandability: int = Field(description="Score for brandability (1-10)", ge=1, le=10)
    total_score: int = Field(description="Total score (sum of all metrics)")
    explanation: str = Field(description="Detailed explanation for the scores")

class DomainEvaluation(BaseModel):
    domain: str = Field(description="The evaluated domain name")
    scores: DomainScore = Field(description="Scores for various metrics")

class DomainEvaluationSet(BaseModel):
    evaluations: List[DomainEvaluation] = Field(description="List of domain evaluations")

domain_eval_parser = PydanticOutputParser(pydantic_object=DomainEvaluationSet)

eval_template = """
You are an expert in domain name evaluation with extensive experience in branding and marketing. Your task is to evaluate the given domain name based on four key metrics: memorability, pronounceability, length, and brandability. Each metric should be scored on a scale of 1-10, where 1 is poor and 10 is excellent.

Domain to evaluate: {domain}
Description: {description}

Please think through each metric carefully and provide a detailed explanation for your scoring. Use the following guidelines:

1. Memorability (1-10):
   - Consider how easily the domain can be remembered after hearing it once.
   - Think about any unique elements or associations that make it stick in memory.
   - Evaluate if it's catchy or has a rhythm that aids recall.

2. Pronounceability (1-10):
   - Assess how easily the domain can be pronounced by native and non-native English speakers.
   - Consider if there are any ambiguous pronunciations or difficult sound combinations.
   - Evaluate if it can be easily spelled after hearing it spoken.

3. Length (1-10):
   - Consider the number of characters in the domain.
   - Shorter domains (around 6-8 characters) generally score higher.
   - Evaluate if the length is appropriate for the brand and easy to type.

4. Brandability (1-10):
   - Assess how well the domain represents a unique brand identity.
   - Consider if it's relevant to the AI SaaS B2B Enterprise space.
   - Evaluate its potential for building a strong brand around it.

For each metric, think step-by-step:
1. What are the positive aspects?
2. What are the potential drawbacks?
3. How does it compare to ideal domains in this space?
4. What's the final score based on these considerations?

After evaluating each metric, sum up the scores to get a total score out of 40.

Provide a detailed explanation for your scoring, highlighting key points that influenced your decision.

{format_instructions}
"""

eval_prompt = PromptTemplate(
    template=eval_template,
    input_variables=["domain", "description"],
    partial_variables={"format_instructions": domain_eval_parser.get_format_instructions()}
)

llm = ChatOpenAI(model='gpt-4', temperature=0, openai_api_key=api_key)

def evaluate_domain(domain: str, description: str):
    evaluate_chain = eval_prompt | llm | domain_eval_parser
    result = evaluate_chain.invoke({"domain": domain, "description": description})
    return result.evaluations[0]

def evaluate_domain_set(domain_set):
    evaluations = []
    for suggestion in domain_set.suggestions:
        evaluation = evaluate_domain(suggestion.domain, suggestion.explanation)
        evaluations.append(evaluation)
    return DomainEvaluationSet(evaluations=evaluations)

@tool
def get_domain_evaluations(domain_set):
    """Evaluate a set of domain suggestions based on memorability, pronounceability, length, and brandability."""
    return evaluate_domain_set(domain_set)

domain_evaluation_tool = Tool(
    name="Domain Evaluation",
    func=get_domain_evaluations,
    description="Evaluate domain suggestions based on memorability, pronounceability, length, and brandability."
)
