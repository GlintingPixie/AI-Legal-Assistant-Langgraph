from langchain_core.messages import HumanMessage
from ipc_vectordb_builder import load_ipc_retriever

retriever = load_ipc_retriever(k=3)

def ipc_section_agent(state, llm):
    case_summary = state["case_summary"]

    # ✅ UPDATED retrieval call
    docs = retriever.invoke(case_summary)

    retrieved_ipc = "\n\n".join(
        f"{doc.metadata.get('section')}: {doc.page_content}"
        for doc in docs
    )

    prompt = f"""
    Based on the following IPC sections retrieved from
    the IPC vector database, identify the most applicable
    sections for the given case.

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

    state["ipc_sections"] = response.content.strip()
    state["ipc_sources"] = docs
    return state
