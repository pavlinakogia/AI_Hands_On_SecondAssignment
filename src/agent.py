import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from src.tools import retrieval_tool, prediction_tool, dataset_stats_tool
from dotenv import load_dotenv

load_dotenv()

_agent = None


def get_agent():
    global _agent
    if _agent is not None:
        return _agent

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    tools = [retrieval_tool, prediction_tool, dataset_stats_tool]

    _agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    return _agent


def run_agent(message: str, history: list = None) -> str:
    agent = get_agent()

    messages = []
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=message))

    result = agent.invoke({"messages": messages})

    # Αποθηκεύουμε όλα τα messages στο history (όχι μόνο το τελευταίο)
    if history is not None:
        new_messages = result["messages"][len(messages):]
        history.extend(new_messages)

    content = result["messages"][-1].content
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content
