# Domain Research and Generation System

This project implements a domain research and generation system using a graph-based workflow. It utilizes various agents to perform market research, generate domain suggestions, and evaluate domain names.

## Codebase Logic

The system is built using LangGraph, a library for creating multi-agent workflows. The main logic is defined in `agents/domain_research_graph.py`, which orchestrates the following agents:

1. **Market Trends Bot** (`market_research_bot.py`):
   - Generates a list of notable companies in various tech sectors.
   - Provides company details including name, category, description, keyword, and domain.

2. **Domain Name Generator Bot** (`domain_generator.py`):
   - Uses the market trends data to generate domain suggestions for new AI SaaS B2B Enterprise startups.
   - Creates unique and brandable domain names based on current trends.

3. **Domain Name Scoring Bot** (`domain_name_scoring_bot.py`):
   - Evaluates generated domain names based on memorability, pronounceability, length, and brandability.
   - Provides detailed scoring and explanations for each domain.

4. **Availability Checker** (placeholder in `domain_research_graph.py`):
   - Currently a placeholder for future implementation of domain availability checking.

The workflow is defined as a graph with conditional edges, allowing for iteration and decision-making based on the results of each step.

## Running Instructions

To run the domain research system:

1. Ensure you have Python 3.7+ installed.

2. Clone the repository:   ```
   git clone <repository-url>
   cd <repository-directory>   ```

3. Install the required dependencies:   ```
   pip install -r requirements.txt   ```

4. Set up your OpenAI API key:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key to the file:     ```
     OPENAI_API_KEY=your_api_key_here     ```

5. Run the main script:   ```
   python main.py   ```

The script will execute the domain research workflow, printing progress and results to the console. The final output will include the generated and evaluated domain suggestions.

## Customization

You can customize the behavior of the system by modifying the following files:

- `agents/market_research_bot.py`: Adjust the prompt or categories for market research.
- `agents/domain_generator.py`: Modify the domain generation criteria or prompt.
- `agents/domain_name_scoring_bot.py`: Change the scoring metrics or evaluation process.
- `agents/domain_research_graph.py`: Alter the workflow structure or add new nodes to the graph.

## Dependencies

The main dependencies for this project are:
- langchain
- langgraph
- openai
- python-dotenv

For a complete list of dependencies, refer to the `requirements.txt` file.

## Note

This system uses the OpenAI GPT-4 model, which may incur costs. Make sure you understand the pricing and usage limits of the OpenAI API before running the script.
