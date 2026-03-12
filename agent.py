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

# Bind tools so LLM can call them
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


class AgentState(dict):
    pass


def call_llm(state):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": messages + [response]}


def should_use_tools(state):

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


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


def run_agent(message: str):

    system_prompt = f"""
You are an AI CRM assistant for pharmaceutical field representatives.

Responsibilities:
- Log interactions with healthcare professionals
- Search HCP records
- Edit interactions
- Retrieve interaction history
- Suggest follow-up actions

IMPORTANT:
If the user describes meeting or interacting with a doctor,
you MUST call the log_interaction tool.

User message:
{message}
"""

    result = graph.invoke({
        "messages": [HumanMessage(content=system_prompt)]
    })

    last = result["messages"][-1]

    return getattr(last, "content", "")