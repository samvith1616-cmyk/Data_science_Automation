from langgraph.graph import StateGraph, END

from src.state import PipelineState
from src.agents.agent1_preprocessing import agent1_preprocessing
from src.agents.agent2_eda import agent2_eda
from src.agents.agent3_code_generation import agent3_model_selector
from src.agents.package_approval import package_approval
from src.agents.agent4_execution import agent4_execute
from src.agents.human_feedback import human_feedback
from src.agents.agent5_report import agent5_report


def route_after_execution(state: PipelineState) -> str:
    if state.get("execution_success"):
        return "success"

    retries = state.get("agent3_retry_count", 0)
    max_retries = state.get("max_agent3_retries", 3)

    if retries < max_retries:
        return "retry"
    return "give_up"


def route_after_feedback(state: PipelineState) -> str:
    if state.get("user_satisfied"):
        return "satisfied"
    return "retry"


def mark_failed(state: PipelineState) -> dict:
    return {"status": "failed_max_retries"}


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("agent1", agent1_preprocessing)
    graph.add_node("agent2", agent2_eda)
    graph.add_node("agent3", agent3_model_selector)
    graph.add_node("package_approval", package_approval)
    graph.add_node("agent4_execute", agent4_execute)
    graph.add_node("human_feedback", human_feedback)
    graph.add_node("agent5_report", agent5_report)
    graph.add_node("mark_failed", mark_failed)

    graph.set_entry_point("agent1")

    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "agent3")
    graph.add_edge("agent3", "package_approval")
    graph.add_edge("package_approval", "agent4_execute")

    graph.add_conditional_edges(
        "agent4_execute",
        route_after_execution,
        {
            "success": "human_feedback",
            "retry": "agent3",
            "give_up": "mark_failed",
        },
    )

    graph.add_conditional_edges(
        "human_feedback",
        route_after_feedback,
        {
            "satisfied": "agent5_report",
            "retry": "agent3",
        },
    )

    graph.add_edge("agent5_report", END)
    graph.add_edge("mark_failed", END)

    return graph.compile()