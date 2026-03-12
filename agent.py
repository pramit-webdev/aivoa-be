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

# Bind tools to LLM so it can call them
llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)


class AgentState(dict):
    pass


def call_llm(state):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": messages + [response]}


def should_use_tools(state):

    last = state["messages"][-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
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

    result = graph.invoke({
        "messages": [
            HumanMessage(
                content=f"""
You are a CRM assistant for pharmaceutical representatives.

If the user describes a doctor interaction, use the log_interaction tool.

User message:
{message}
"""
            )
        ]
    })

    last = result["messages"][-1]

    return getattr(last, "content", "")