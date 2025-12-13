from langchain_core.messages import HumanMessage

def legal_drafter_agent(state, llm):
    prompt = f"""
    Draft a legal opinion based on:

    Case Summary:
    {state["case_summary"]}

    IPC Sections:
    {state["ipc_sections"]}

    Legal Precedents:
    {state["precedents"]}

    Include:
    - Legal analysis
    - Possible liabilities
    - Disclaimer (not legal advice)
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    state["final_opinion"] = response.content.strip()
    return state
