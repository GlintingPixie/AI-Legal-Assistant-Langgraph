# app.py

import streamlit as st
from dotenv import load_dotenv

from graph import build_graph

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Title
st.title("⚖️ AI Legal Assistant (IPC + Precedents)")
st.markdown(
    """
Enter a legal problem in plain English.  
This assistant will help you:
- 📌 Understand the legal issue
- 📘 Identify applicable IPC sections
- ⚖️ Retrieve relevant legal precedents
- 🧾 Generate a structured legal opinion
"""
)

# Input form
with st.form("legal_form"):
    user_input = st.text_area(
        "📝 Describe your legal issue:",
        height=250,
        placeholder="Example: A person forged land documents and cheated another by selling the land..."
    )
    submitted = st.form_submit_button("🔍 Run Legal Assistant")

# Run pipeline
if submitted:
    if not user_input.strip():
        st.warning("Please enter a legal issue to analyze.")
    else:
        with st.spinner("🔎 Analyzing your case using multi-agent reasoning..."):
            try:
                app = build_graph()
                result = app.invoke({"query": user_input})
            except Exception as e:
                st.error("❌ An error occurred while processing the case.")
                st.exception(e)
                st.stop()

        st.success("✅ Legal Assistant completed the analysis!")

        # ============================
        # Display Outputs (Agent-wise)
        # ============================

        st.subheader("📌 Case Summary")
        st.markdown(result.get("case_summary", "Not available"))

        st.subheader("📘 Applicable IPC Sections")
        st.markdown(result.get("ipc_sections", "Not available"))

        with st.expander("🔍 IPC Retrieval Sources"):
            for doc in result.get("ipc_sources", []):
                st.markdown(
                    f"**Section {doc.metadata.get('section')}** – {doc.metadata.get('section_title')}"
                )

        st.subheader("⚖️ Relevant Legal Precedents")
        st.markdown(result.get("precedents", "Not available"))

        with st.expander("🌐 Precedent Sources"):
            for src in result.get("precedent_sources", []):
                st.markdown(f"- [{src['title']}]({src['url']})")

        st.subheader("🧾 Final Legal Opinion")
        st.markdown(result.get("final_opinion", "Not available"))

        st.caption(
            "⚠️ Disclaimer: This AI-generated legal analysis is for educational purposes only "
            "and does not constitute legal advice."
        )
