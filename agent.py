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

llm = ChatGroq(

    groq_api_key=GROQ_API_KEY,

    model="gemma2-9b-it"

)

tools = [

    search_hcp,

    log_interaction,

    edit_interaction,

    interaction_history,

    suggest_followup

]

agent = initialize_agent(

    tools,

    llm,

    agent=AgentType.OPENAI_FUNCTIONS,

    verbose=True

)


def run_agent(message: str):

    return agent.run(message)