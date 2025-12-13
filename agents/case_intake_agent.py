from langchain_core.messages import HumanMessage

def case_intake_agent(state, llm):
    query = state["query"]

    prompt = f"""
    Extract key legal facts from the following case:

    {query}

    Return:
    - Parties
    - Facts
    - Alleged offence
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    # ✅ UPDATE state, do not replace it
    state["case_summary"] = response.content.strip()
    return state
