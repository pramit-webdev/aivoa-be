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

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


class AgentState(dict):
    pass


def call_llm(state):

    message = state["message"]

    prompt = f"""
You are a CRM AI assistant for pharmaceutical representatives.

If the user describes an interaction with a doctor,
use the appropriate tool to record or retrieve information.

User message:
{message}
"""

    response = llm_with_tools.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}


def should_use_tool(state):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


workflow = StateGraph(AgentState)

workflow.add_node("llm", call_llm)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("llm")

workflow.add_conditional_edges(
    "llm",
    should_use_tool,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "llm")

graph = workflow.compile()


def run_agent(message: str):

    result = graph.invoke({"message": message})

    last = result["messages"][-1]

    return getattr(last, "content", "")