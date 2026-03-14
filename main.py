import streamlit as st
import rag

# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="Forex AI Research Assistant",
    page_icon="📊",
    layout="wide"
)

# ---------- CUSTOM CSS ----------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

section[data-testid="stSidebar"] {
    background-color: #1A1D24;
}

.main-title{
    font-size:40px;
    font-weight:700;
    color:white;
}

.subtitle{
    font-size:18px;
    color:#B0B3B8;
}

.user-msg{
    background:#2563EB;
    color:white;
    padding:14px;
    border-radius:12px;
    margin:8px;
    max-width:70%;
}

.bot-msg{
    background:#2D2F36;
    color:white;
    padding:14px;
    border-radius:12px;
    margin:8px;
    max-width:70%;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------

st.markdown('<div class="main-title">📊 Forex AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">RAG powered Forex market analysis chatbot</div>', unsafe_allow_html=True)

st.divider()

# ---------- INITIALIZE BACKEND ----------

rag.initialize_components()
qa_chain = rag.get_qa_chain()

# ---------- SIDEBAR ----------

with st.sidebar:

    st.header("⚙️ Settings")

    urls_text = st.text_area(
        "Add Forex article URLs",
        height=200,
        placeholder="Paste URLs here (one per line)"
    )

    urls = [u.strip() for u in urls_text.split("\n") if u.strip()]

    if st.button("📥 Load Articles"):

        if urls:

            with st.spinner("Processing articles..."):
                rag.process_urls(urls)

            st.success("Articles successfully indexed!")

        else:
            st.warning("Please enter at least one URL.")

    st.divider()

    if st.button("🗑 Clear Database"):

        rag.vector_store.reset_collection()
        st.success("Vector database cleared!")

    st.divider()

    st.subheader("Example Questions")

    st.write("• Why is USD rising?")
    st.write("• What affects EUR/USD?")
    st.write("• Why is Japanese Yen weak?")
    st.write("• What moves GBP/USD?")

# ---------- CHAT MEMORY ----------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- DISPLAY CHAT ----------

for role, message in st.session_state.chat_history:

    if role == "user":

        st.markdown(
            f'<div style="display:flex; justify-content:flex-end">'
            f'<div class="user-msg">{message}</div></div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f'<div style="display:flex; justify-content:flex-start">'
            f'<div class="bot-msg">{message}</div></div>',
            unsafe_allow_html=True
        )

# ---------- CHAT INPUT ----------

user_input = st.chat_input("Ask about Forex markets...")

if user_input:

    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Analyzing market data..."):

        result = qa_chain.invoke({"question": user_input})

        answer = result["answer"]
        sources = result.get("sources", "")

        final_answer = answer

        if sources:
            final_answer += f"\n\nSources:\n{sources}"

    st.session_state.chat_history.append(("bot", final_answer))

    st.rerun()