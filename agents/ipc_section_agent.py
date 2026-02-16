from langchain_core.messages import HumanMessage
from ipc_vectordb_builder import load_ipc_vectordb


def ipc_section_agent(state, llm):
    """
    Retrieves relevant IPC sections using vector similarity search,
    computes retrieval confidence, and asks the LLM to identify
    applicable sections with justification.
    """

    case_summary = state.get("case_summary", "")

    if not case_summary:
        state["ipc_sections"] = "No case summary provided."
        state["ipc_sources"] = []
        state["ipc_confidence"] = 0.0
        return state

    # Load vector DB
    vectordb = load_ipc_vectordb()

    # Retrieve documents WITH similarity scores
    docs_scores = vectordb.similarity_search_with_score(case_summary, k=3)

    if not docs_scores:
        state["ipc_sections"] = "No relevant IPC sections found."
        state["ipc_sources"] = []
        state["ipc_confidence"] = 0.0
        return state

    # Separate documents and scores
    docs = [doc for doc, score in docs_scores]
    scores = [score for doc, score in docs_scores]

    # Convert distance → confidence
    # (Chroma returns distance: lower = better match)
    avg_distance = sum(scores) / len(scores)
    confidence = round(1 - avg_distance, 2)

    # Build IPC context
    retrieved_ipc = "\n\n".join(
        f"Section {doc.metadata.get('section')}: {doc.metadata.get('section_title')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

    # Ask LLM to determine applicable sections
    prompt = f"""
    Based on the IPC sections retrieved from the legal database,
    identify the most applicable sections for the case.

    Case Summary:
    {case_summary}

    Retrieved IPC Sections:
    {retrieved_ipc}

    Respond with:
    - IPC Section number
    - Section title
    - Legal justification
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    # Update state
    state["ipc_sections"] = response.content.strip()
    state["ipc_sources"] = docs
    state["ipc_confidence"] = confidence

    return state
