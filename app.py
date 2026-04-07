import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Imports
# ==============================
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from typing import Annotated

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==============================
# Initialize Tools
# ==============================

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=1200
    )
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=800
    )
)

tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily]

# ==============================
# LLM (ReAct-enabled)
# ==============================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# 🔥 IMPORTANT: Enable tool calling
llm_with_tools = llm.bind_tools(tools)

# ==============================
# LangGraph State
# ==============================

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ==============================
# Chatbot Node (Reasoning Step)
# ==============================

def chatbot(state: State):
    messages = [
        SystemMessage(content="""
You are ScholarGraph AI, a research assistant.

You must follow ReAct pattern:
- Think step-by-step
- Use tools when needed
- Use arXiv for research papers
- Use Wikipedia for concepts
- Use Tavily for latest info
- Continue reasoning until final answer
""")
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# ==============================
# Build ReAct Graph
# ==============================

builder = StateGraph(State)

# Nodes
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

# Flow
builder.add_edge(START, "chatbot")

# 🔥 THIS ENABLES REACT LOOP
builder.add_conditional_edges(
    "chatbot",
    tools_condition   # decides: call tool OR finish
)

builder.add_edge("tools", "chatbot")

# Compile graph
graph = builder.compile()

# ==============================
# Memory Helper (Last 5 Messages)
# ==============================

def get_last_messages(chat_history, k=5):
    messages = []
    for msg in chat_history[-k:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

# ==============================
# CLI Chat Loop (Test)
# ==============================

if __name__ == "__main__":
    chat_history = []

    print("ScholarGraph ReAct Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Store user message
        chat_history.append({"role": "user", "content": user_input})

        # Get last 5 messages
        messages = get_last_messages(chat_history, 5)

        # Add current input
        messages.append(HumanMessage(content=user_input))

        # Run ReAct agent
        response = graph.invoke({
            "messages": messages
        })

        final_output = response["messages"][-1].content

        print("\nAI:", final_output, "\n")

        # Store AI response
        chat_history.append({"role": "assistant", "content": final_output})
