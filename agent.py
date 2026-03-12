from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType

from config import GROQ_API_KEY
from tools import (
    search_hcp,
    log_interaction,
    edit_interaction,
    interaction_history,
    suggest_followup
)

# Initialize LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)

# Register tools
tools = [
    search_hcp,
    log_interaction,
    edit_interaction,
    interaction_history,
    suggest_followup
]

# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)


def run_agent(message: str):

    try:

        # Provide system guidance for the agent
        prompt = f"""
You are an AI CRM assistant for pharmaceutical field representatives.

Your responsibilities:
- Log interactions with Healthcare Professionals (HCPs)
- Search HCP details
- Edit interaction records
- Retrieve interaction history
- Suggest follow-up actions

Use the available tools whenever required.

User message:
{message}
"""

        result = agent.invoke({"input": prompt})

        # If the agent returned a proper output
        if isinstance(result, dict) and "output" in result and result["output"]:
            return result["output"]

        # If output missing, return raw result for debugging
        return str(result)

    except Exception as e:
        return f"Agent error: {str(e)}"