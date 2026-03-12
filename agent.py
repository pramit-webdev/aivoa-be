from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
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
# -----------------------------

class AgentState(TypedDict):
    messages: List[BaseMessage]


# -----------------------------
# LLM NODE
# -----------------------------

def call_llm(state: AgentState):

    messages = state.get("messages", [])

    response = llm_with_tools.invoke(messages)

    return {
        "messages": messages + [response]
    }


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

def run_agent(message: str):

    system_prompt = f"""
You are an AI CRM assistant for pharmaceutical field representatives.

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

User message:
{message}
"""

    result = graph.invoke({
        "messages": [
            HumanMessage(content=system_prompt)
        ]
    })

    last_message = result["messages"][-1]

    return getattr(last_message, "content", "")