from langchain_core.messages import HumanMessage


def legal_drafter_agent(state, llm):
    """
    Generates final legal opinion and computes opinion confidence.
    """

    case_summary = state.get("case_summary", "")
    ipc_sections = state.get("ipc_sections", "")
    precedents = state.get("precedents", "")

    # Step 1: Generate legal opinion
    prompt = f"""
    Draft a legal opinion based on:

    Case Summary:
    {case_summary}

    IPC Sections:
    {ipc_sections}

    Legal Precedents:
    {precedents}

    Include:
    - Legal analysis
    - Possible liabilities
    - Disclaimer (not legal advice)
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    opinion_text = response.content.strip()

    # Save opinion FIRST
    state["final_opinion"] = opinion_text

    # Step 2: Ask LLM to rate confidence
    confidence_prompt = f"""
    Rate your confidence in the legal opinion below on a scale from 0 to 1.

    Opinion:
    {opinion_text}

    Return only a number.
    """

    score_response = llm.invoke([HumanMessage(content=confidence_prompt)])

    try:
        opinion_conf = float(score_response.content.strip())
    except:
        opinion_conf = 0.7  # fallback default

    state["opinion_confidence"] = round(opinion_conf, 2)

    # Step 3: Compute overall confidence
    ipc_conf = state.get("ipc_confidence", 0.5)
    prec_conf = state.get("precedent_confidence", 0.5)

    overall = 0.5 * ipc_conf + 0.3 * prec_conf + 0.2 * opinion_conf

    state["overall_confidence"] = round(overall, 2)

    return state
