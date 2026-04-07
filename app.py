import streamlit as st
import os
from dotenv import load_dotenv

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# LangChain + Tools Imports
# ==============================
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from typing import Annotated

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

# LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==============================
# LangSmith (Tracing Optional)
# ==============================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "ScholarGraph"

# ==============================
# Initialize Tools (Improved)
# ==============================

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=5,
        doc_content_chars_max=1500,
    )
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=1000,
    )
)

tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily]

# ==============================
# Initialize LLM (Upgraded)
# ==============================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Strong tool calling
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)

# ==============================
# State Schema
# ==============================

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ==============================
# Intent Detection (IMPORTANT)
# ==============================

def detect_intent(user_input: str):
    text = user_input.lower()

    if any(word in text for word in ["research paper", "paper", "arxiv", "journal"]):
        return "arxiv"
    elif any(word in text for word in ["what is", "define", "meaning"]):
        return "wiki"
    elif any(word in text for word in ["latest", "news", "recent", "update"]):
        return "tavily"
    else:
        return "general"

# ==============================
# Chatbot Node (Improved)
# ==============================

def chatbot(state: State):
    user_message = state["messages"][-1].content
    intent = detect_intent(user_message)

    # 🔥 ROUTING LOGIC (Key Fix)
    if intent == "arxiv":
        result = arxiv.run(user_message)
        return {"messages": [HumanMessage(content=result)]}

    elif intent == "wiki":
        result = wiki.run(user_message)
        return {"messages": [HumanMessage(content=result)]}

    elif intent == "tavily":
        result = tavily.run(user_message)
        return {"messages": [HumanMessage(content=result)]}

    # 🔥 Default → LLM with tool support
    messages = [
        SystemMessage(
            content="""
You are ScholarGraph AI, a research assistant.

Guidelines:
- Use arXiv tool for research papers
- Use Wikipedia for explanations
- Use Tavily for current info
- Be clear, structured, and helpful
"""
        )
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# ==============================
# Build Graph
# ==============================

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_edge("tools", "chatbot")

graph = builder.compile()

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="ScholarGraph AI", layout="wide")
st.title("📚 ScholarGraph – Multi-Tool Research Assistant")

# Session Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Ask anything (research, papers, live search)...")

if user_input:

    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # Invoke Graph
    response = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # Debug (Optional)
    # for msg in response["messages"]:
    #     print(msg)

    final_message = response["messages"][-1].content

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(final_message)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_message}
    )
