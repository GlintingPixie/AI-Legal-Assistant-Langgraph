import streamlit as st
from graph import build_graph

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Agentic Legal AI Assistant")
st.write("Analyze legal cases using IPC retrieval, precedents, and AI reasoning.")

# -----------------------------
# Build LangGraph App
# -----------------------------
@st.cache_resource
def load_app():
    return build_graph()

app = load_app()

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area(
    "Enter case description:",
    placeholder="Example: A person cheated another by forging documents to sell land."
)

run_button = st.button("Analyze Case")

# -----------------------------
# Run Analysis
# -----------------------------
if run_button and user_input:

    with st.spinner("Analyzing case..."):

        result = app.invoke({"query": user_input})

    st.success("Analysis Complete")

    # -----------------------------
    # Legal Opinion Output
    # -----------------------------
    st.subheader("⚖️ Legal Opinion")
    st.write(result.get("final_opinion", "No opinion generated."))

    # -----------------------------
    # Confidence Scores
    # -----------------------------
    st.subheader("📊 Confidence & Explainability")

    overall_conf = result.get("overall_confidence", 0)

    # Progress bar for overall confidence
    st.progress(overall_conf)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="IPC Retrieval Confidence",
            value=result.get("ipc_confidence", "N/A")
        )

        st.metric(
            label="Precedent Reliability",
            value=result.get("precedent_confidence", "N/A")
        )

    with col2:
        st.metric(
            label="Opinion Confidence",
            value=result.get("opinion_confidence", "N/A")
        )

        st.metric(
            label="Overall Confidence",
            value=overall_conf
        )

    # -----------------------------
    # Sources (Explainability)
    # -----------------------------
    with st.expander("🔍 Retrieved IPC Sources"):
        for doc in result.get("ipc_sources", []):
            st.write(f"**Section {doc.metadata.get('section')}**")
            st.write(doc.page_content)
            st.divider()

    with st.expander("📚 Precedent Sources"):
        for src in result.get("precedent_sources", []):
            st.write(f"**{src.get('title')}**")
            st.write(src.get("url"))
            st.divider()

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("⚠️ This tool provides educational legal insights and is not legal advice.")
