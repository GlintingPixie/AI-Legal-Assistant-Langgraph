from langchain_core.messages import HumanMessage
from tools.legal_precedent_search_tool import legal_precedent_search_tool
import re


def _build_short_search_query(ipc_text: str) -> str:
    sections = re.findall(r"\b\d{2,3}\b", ipc_text)
    sections = list(set(sections))

    if not sections:
        return "Indian forgery and cheating landmark judgments"

    query = (
        "Indian Supreme Court High Court judgments "
        "related to IPC sections " + ", ".join(sections)
    )

    return query[:380]

def _domain_score(url):
    if "indiankanoon" in url:
        return 1.0
    if "scconline" in url:
        return 0.9
    if "supremecourt" in url:
        return 0.9
    return 0.6

def legal_precedent_agent(state, llm):
    ipc_text = state["ipc_sections"]

    search_query = _build_short_search_query(ipc_text)
    search_results = legal_precedent_search_tool(search_query)

    if search_results:
        reliability = sum(_domain_score(r["url"]) for r in search_results)
        reliability /= len(search_results)
        state["precedent_confidence"] = round(reliability, 2)
    else:
        state["precedent_confidence"] = 0.4

    if not search_results:
        state["precedents"] = (
            "Relevant precedents could not be fetched due to search constraints. "
            "Legal principles are derived from general judicial interpretations "
            "of forgery and cheating under IPC."
        )
        state["precedent_sources"] = []
        return state

    context = "\n\n".join(
        f"Title: {r['title']}\nSummary: {r['content']}\nSource: {r['url']}"
        for r in search_results
    )

    prompt = f"""
    Based on the following web search results,
    summarize relevant Indian legal precedents.

    {context}

    Provide:
    - Case name
    - Year
    - Legal principle
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    state["precedents"] = response.content.strip()
    state["precedent_sources"] = search_results
    return state
