from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

from config import OPENAI_API_KEY
from tools import (
    search_hcp,
    log_interaction,
    edit_interaction,
    interaction_history,
    suggest_followup
)

# Initialize GPT model
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
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

        prompt = f"""
You are an AI CRM assistant for pharmaceutical field representatives.

Your job is to record interactions with healthcare professionals.

If the user describes a meeting with a doctor, you MUST log the interaction using the log_interaction tool.

Extract these fields from the message if possible:
- hcp_name
- interaction_type
- product
- notes
- date
- follow_up

If some fields are missing, make reasonable assumptions.

User message:
{message}
"""

        result = agent.invoke({"input": prompt})

        if isinstance(result, dict):

            if result.get("output"):
                return result["output"]

            return str(result)

        return str(result)

    except Exception as e:
        return f"Agent error: {str(e)}"