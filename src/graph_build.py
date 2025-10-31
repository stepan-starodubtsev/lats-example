from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.tree_state import TreeState
from src.start_node import generate_initial_response
from src.expand import expand


def should_loop(state: TreeState) -> Literal["expand", "__end__"]:
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 5:
        return END
    return "expand"


def build_graph():
    builder = StateGraph(TreeState)
    builder.add_node("start", generate_initial_response)
    builder.add_node("expand", expand)
    builder.add_edge(START, "start")

    builder.add_conditional_edges(
        "start",
        should_loop,
        ["expand", END],
    )
    builder.add_conditional_edges(
        "expand",
        should_loop,
        ["expand", END],
    )

    return builder.compile()


