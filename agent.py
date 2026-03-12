from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from config import OPENAI_API_KEY
from tools import (
    search_hcp,
    log_interaction,
    edit_interaction,
    interaction_history,
    suggest_followup
)

# LLM
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

llm = llm.bind_tools(tools)

tool_node = ToolNode(tools)


# ---------------------
# STATE
# ---------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]


# ---------------------
# LLM NODE
# ---------------------

def call_model(state: AgentState):

    response = llm.invoke(state["messages"])

    return {"messages": [response]}


# ---------------------
# GRAPH
# ---------------------

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

graph = workflow.compile()


# ---------------------
# ENTRY FUNCTION
# ---------------------

def run_agent(message: str):

    system_prompt = f"""
You are an AI CRM assistant for pharmaceutical representatives.

Responsibilities:
- Log interactions with healthcare professionals
- Search HCP records
- Edit interactions
- Retrieve interaction history
- Suggest follow-ups

IMPORTANT:
If the user describes meeting or interacting with a doctor,
you MUST call the log_interaction tool.

User message:
{message}
"""

    result = graph.invoke({
        "messages": [
            HumanMessage(content=system_prompt)
        ]
    })

    return result["messages"][-1].content