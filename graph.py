from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from rag import answer_question, direct_answer


class State(TypedDict):
    question: str
    intent: str
    answer: str


def classify_intent(state: State) -> dict:
    """Route questions into 'policy_question' or 'general_question'.

    Beginner note: keyword matching is fragile — any future improvement
    should swap this for an LLM-based classifier or a regex with word
    boundaries to avoid partial matches like 'cancel' inside 'canceling'.
    """
    question = state["question"].lower()

    policy_keywords = [
        "delivery", "refund", "return", "membership",
        "track", "tracking", "cancel", "cancellation",
        "damaged", "missing", "gift card",
    ]

    if any(word in question for word in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {"intent": intent}


def retrieve_and_answer(state: State) -> dict:
    """RAG path — fetch from vector store, then LLM."""
    return {"answer": answer_question(state["question"])}


def direct_answer_node(state: State) -> dict:
    """Direct LLM path — no retrieval (for general questions)."""
    return {"answer": direct_answer(state["question"])}


def route_question(state: State) -> str:
    """Conditional edge: decide which node to call next."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


# Build the LangGraph pipeline
graph_builder = StateGraph(State)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer_node)

graph_builder.add_edge(START, "classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)

graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)

graph = graph_builder.compile()