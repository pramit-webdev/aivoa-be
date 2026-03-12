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
    model="llama3-70b-8192"
)

# Register tools
tools = [
    search_hcp,
    log_interaction,
    edit_interaction,
    interaction_history,
    suggest_followup
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)


def run_agent(message: str):

    try:
        result = agent.invoke({"input": message})

        # Ensure response is serializable
        return result["output"]

    except Exception as e:
        return f"Agent error: {str(e)}"