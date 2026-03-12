from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import OPENAI_API_KEY
from tools import (
    search_hcp,
    log_interaction,
    edit_interaction,
    interaction_history,
    suggest_followup
)

# Initialize GPT
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0
)

tools = [
    search_hcp,
    log_interaction,
    edit_interaction,
    interaction_history,
    suggest_followup
]

tool_node = ToolNode(tools)


class AgentState(dict):
    pass


def call_llm(state):

    message = state["message"]

    prompt = f"""
You are a CRM AI assistant for pharmaceutical representatives.

User message:
{message}

Decide whether to call a tool or answer directly.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"response": response.content}


workflow = StateGraph(AgentState)

workflow.add_node("llm", call_llm)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("llm")

workflow.add_edge("llm", END)

graph = workflow.compile()


def run_agent(message: str):

    result = graph.invoke({"message": message})

    return result.get("response", "")