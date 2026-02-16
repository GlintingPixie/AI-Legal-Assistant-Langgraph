from langgraph.graph import StateGraph, START, END
from config.azure_llm import get_llm

from agents.case_intake_agent import case_intake_agent
from agents.ipc_section_agent import ipc_section_agent
from agents.legal_precedent_agent import legal_precedent_agent
from agents.legal_drafter_agent import legal_drafter_agent
from typing import TypedDict


class LegalState(TypedDict, total=False):
    query: str
    case_summary: str

    ipc_sections: str
    ipc_sources: list
    ipc_confidence: float

    precedents: str
    precedent_sources: list
    precedent_confidence: float

    final_opinion: str
    opinion_confidence: float
    overall_confidence: float



def build_graph():
    llm = get_llm()
    graph = StateGraph(LegalState)

    graph.add_node("case_intake", lambda s: case_intake_agent(s, llm))
    graph.add_node("ipc_section", lambda s: ipc_section_agent(s, llm))
    graph.add_node("precedent", lambda s: legal_precedent_agent(s, llm))
    graph.add_node("drafter", lambda s: legal_drafter_agent(s, llm))

    graph.add_edge(START, "case_intake")
    graph.add_edge("case_intake", "ipc_section")
    graph.add_edge("ipc_section", "precedent")
    graph.add_edge("precedent", "drafter")
    graph.add_edge("drafter", END)

    return graph.compile()
