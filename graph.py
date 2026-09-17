from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from rag import answer_question, direct_answer


# State
class State(TypedDict):
    question: str
    intent: str
    answer: str


# 1. Classify question
def classify_intent(state: State):

    question = state["question"].lower()

    policy_keywords = [
        "delivery",
        "refund",
        "return",
        "membership",
        "track",
        "tracking",
        "cancel",
        "cancellation",
        "damaged",
        "missing",
        "gift card"
    ]

    if any(word in question for word in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# 2. Policy question
def retrieve_and_answer(state: State):

    question = state["question"]

    answer = answer_question(question)

    return {
        "answer": answer
    }


# 3. General question
def direct_answer_node(state: State):

    question = state["question"]

    answer = direct_answer(question)

    return {
        "answer": answer
    }


# Decide which node to go to
def route_question(state: State):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# Create graph
graph_builder = StateGraph(State)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer_node)


# Starting point
graph_builder.add_edge(START, "classify_intent")


# Conditional routing
graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)


# Ending points
graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)


# Compile
graph = graph_builder.compile()