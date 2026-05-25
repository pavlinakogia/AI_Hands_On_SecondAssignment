from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import run_agent, get_agent as get_agent_instance

app = FastAPI(title="Weather Prediction Agent")

session_store: dict = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id

    if session_id not in session_store:
        session_store[session_id] = []

    history = session_store[session_id]
    history.append(HumanMessage(content=request.message))
    response = run_agent(request.message, history)

    return ChatResponse(response=response)


async def stream_agent_response(message: str, history: list):
    agent = get_agent_instance()

    messages = []
    if history:
        messages.extend(history)
    messages.append(HumanMessage(content=message))

    full_response = []

    async for chunk in agent.astream({"messages": messages}):
        if "agent" in chunk:
            for msg in chunk["agent"]["messages"]:
                content = msg.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block["text"]
                            full_response.append(text)
                            yield f"data: {text}\n\n"
                elif isinstance(content, str) and content:
                    full_response.append(content)
                    yield f"data: {content}\n\n"

    yield "data: [DONE]\n\n"

    # Ενημέρωση history μετά το streaming
    full_text = " ".join(full_response)
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=full_text))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id

    if session_id not in session_store:
        session_store[session_id] = []

    history = session_store[session_id]

    return StreamingResponse(
        stream_agent_response(request.message, history),
        media_type="text/event-stream"
    )