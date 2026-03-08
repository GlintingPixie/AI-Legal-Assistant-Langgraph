from langchain_core.messages import HumanMessage

def case_intake_agent(state, llm):
    query = state["query"]

    prompt = f"""
    You are a legal research assistant. Provide neutral, educational legal analysis only. Do not provide actionable wrongdoing guidance.
    Extract key legal facts from the following case:

    {query}

    Return:
    - Parties
    - Facts
    - Alleged offence
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    state["case_summary"] = response.content.strip()
    return state
