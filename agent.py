from typing import TypedDict, List, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
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

# -----------------------------
# LLM SETUP
# -----------------------------

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

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Tool execution node
tool_node = ToolNode(tools)


# -----------------------------
# STATE DEFINITION
# Use Annotated with operator.add so LangGraph appends messages
# instead of replacing the entire list on each node return.
# This is what keeps tool messages properly paired with their
# preceding tool_calls messages — preventing the 400 error.
# -----------------------------

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# -----------------------------
# LLM NODE
# -----------------------------

def call_llm(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    # Return only the new message; operator.add will append it
    return {"messages": [response]}


# -----------------------------
# ROUTING LOGIC
# -----------------------------

def should_use_tools(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


# -----------------------------
# GRAPH WORKFLOW
# -----------------------------

workflow = StateGraph(AgentState)

workflow.add_node("llm", call_llm)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("llm")

workflow.add_conditional_edges(
    "llm",
    should_use_tools,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "llm")

graph = workflow.compile()


# -----------------------------
# AGENT ENTRY FUNCTION
# -----------------------------

SYSTEM_PROMPT = """You are an AI CRM assistant for pharmaceutical field representatives.

Responsibilities:
- Log interactions with healthcare professionals
- Search HCP records
- Edit interactions
- Retrieve interaction history
- Suggest follow-up actions

IMPORTANT RULES:

If the user describes meeting or interacting with a doctor,
you MUST call the log_interaction tool.

Extract these fields if possible:
- hcp_name
- interaction_type
- product
- notes
- date
- follow_up

If information is missing:
- interaction_type = "meeting"
- date = "today"
- notes = summarize user message
"""

def run_agent(message: str):
    result = graph.invoke({
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=message)
        ]
    })

    last_message = result["messages"][-1]

    return getattr(last_message, "content", "")