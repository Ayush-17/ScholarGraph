import streamlit as st
import os
from dotenv import load_dotenv

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# LangChain + Tools Imports (v0.4.x compatible)
# ==============================
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
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

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "ScholarGraph"

# ==============================
# Initialize Tools
# ==============================

arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=500,
    )
)

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=500,
    )
)

tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily]

# ==============================
# Initialize LLM
# ==============================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)

# ==============================
# LangGraph State Schema
# ==============================

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ==============================
# Node Definition
# ==============================

def chatbot(state: State):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
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

    # Invoke graph
    response = graph.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # Extract final AI message
    final_message = response["messages"][-1].content

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(final_message)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_message}
    )
