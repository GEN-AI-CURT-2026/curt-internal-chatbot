"""
app.py  (Memory-Enhanced Streamlit UI)
=======================================
Adds a live Memory Inspector panel in the sidebar so you can watch
the buffer + summary evolve in real-time while you chat.

Changes vs original
-------------------
- Imports MemoryManager and passes it into RAGPipeline.
- Sidebar shows: buffer contents, rolling summary, and stats.
- "Clear Conversation" also resets memory.
- Response header shows which memory context was active.
"""

import streamlit as st
import os
from rag_pipeline import RAGPipeline
from memory_manager import MemoryManager
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Khaled's CV Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size:2.2rem; color:#1f77b4; text-align:center; margin-bottom:1.5rem; }
    .response-box { background:#f0f2f6; padding:1.2rem; border-radius:10px;
                    border-left:5px solid #1f77b4; margin:0.8rem 0; }
    .project-tag  { background:#ff9900; color:white; padding:0.2rem 0.5rem;
                    border-radius:5px; font-size:0.75rem; font-weight:bold;
                    margin-left:8px; display:inline-block; }
    .memory-box   { background:#eaf4e8; padding:0.8rem; border-radius:8px;
                    border-left:4px solid #2ca02c; font-size:0.82rem;
                    font-family:monospace; white-space:pre-wrap; }
    .summary-box  { background:#fff3cd; padding:0.8rem; border-radius:8px;
                    border-left:4px solid #fd7e14; font-size:0.82rem; }
    .stat-chip    { background:#e8f4fd; padding:0.15rem 0.5rem; border-radius:12px;
                    font-size:0.78rem; margin:2px; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_project_query(question: str) -> bool:
    keywords = ['project', 'work', 'built', 'designed', 'robotics', 'car',
                'calculator', 'app', 'system', 'watch', 'sumo',
                'worked on', 'done', 'created', 'developed']
    q = question.lower()
    return any(k in q for k in keywords)


# ── Main App Class ─────────────────────────────────────────────────────────────

class StreamlitApp:

    def __init__(self):
        self.rag_pipeline: RAGPipeline | None = None
        self._init_pipeline()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_pipeline(self):
        try:
            with st.spinner("🔄 Loading CV Assistant…"):
                # Shared MemoryManager (kept in session state so it survives reruns)
                if "memory_manager" not in st.session_state:
                    st.session_state.memory_manager = MemoryManager(
                        buffer_size=3,
                        persistence_path="memory_state.json",
                    )

                pipeline = RAGPipeline(memory=st.session_state.memory_manager)

                docs = pipeline.load_and_prepare_documents()
                if not docs:
                    st.error("❌ Failed to load documents")
                    return

                if not pipeline.create_vector_store(docs):
                    st.error("❌ Failed to create vector store")
                    return

                if not pipeline.setup_qa_chain():
                    st.error("❌ Failed to setup QA chain")
                    return

                self.rag_pipeline = pipeline
        except Exception as e:
            st.error(f"❌ Initialisation failed: {e}")

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _sidebar(self):
        st.sidebar.title("ℹ️ About")
        st.sidebar.markdown("""
        AI assistant for **Khaled Ashraf's CV**:
        - 🎓 Education & Background
        - 💻 Skills & Technologies
        - 🚀 Projects & Experience
        - 📜 Certificates
        - 🏆 Achievements
        """)

        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Sample Questions")
        samples = [
            "What projects has Khaled worked on?",
            "Tell me about the Autonomous RC car project.",
            "What programming languages does Khaled know?",
            "What is his education background?",
            "What certificates does Khaled have?",
        ]
        for q in samples:
            if st.sidebar.button(q, key=f"btn_{q}"):
                st.session_state.user_question = q

        # ── Memory Inspector ──────────────────────────────────────────────────
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 Memory Inspector")

        mem: MemoryManager | None = st.session_state.get("memory_manager")
        if mem is None:
            st.sidebar.info("Memory not initialised yet.")
            return

        stats = mem.stats()

        # Stat chips
        st.sidebar.markdown(
            f'<span class="stat-chip">Total turns: {stats["total_exchanges"]}</span>'
            f'<span class="stat-chip">Buffer: {stats["buffer_length"]}/{stats["buffer_size_limit"]}</span>'
            f'<span class="stat-chip">Summaries: {stats["summarisation_calls"]}</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.markdown("")

        # Rolling summary
        summary = mem.get_summary()
        if summary:
            st.sidebar.markdown("**📜 Rolling Summary (long-term)**")
            st.sidebar.markdown(
                f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True
            )
        else:
            st.sidebar.caption("No summary yet (buffer hasn't overflowed).")

        # Buffer contents
        buffer = mem.get_buffer()
        if buffer:
            st.sidebar.markdown(f"**📌 Buffer (last {stats['buffer_size_limit']} turns)**")
            for i, ex in enumerate(buffer, 1):
                with st.sidebar.expander(f"Turn {i} — {ex.timestamp}"):
                    st.markdown(f"**Q:** {ex.question}")
                    st.markdown(f"**A:** {ex.answer[:200]}{'…' if len(ex.answer)>200 else ''}")
        else:
            st.sidebar.caption("Buffer is empty.")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**🔧 Stack:** Streamlit · LangChain · Gemini · ChromaDB · HybridMemory")

    # ── Main Content ──────────────────────────────────────────────────────────

    def _main(self):
        st.markdown('<div class="main-header">📄 Khaled\'s CV AI Assistant</div>', unsafe_allow_html=True)

        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        if "user_question" not in st.session_state:
            st.session_state.user_question = ""

        # Chat history
        for q, a, sources, is_proj in st.session_state.conversation_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                if is_proj:
                    st.markdown('**Assistant** <span class="project-tag">PROJECT FOCUS</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown("**Assistant**")
                st.markdown(a)
                if sources:
                    with st.expander("📚 Sources"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(f"**{i}.** {s}")

        # Input row
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input(
                "Ask about Khaled's CV:",
                value=st.session_state.user_question,
                placeholder="e.g., What programming languages does Khaled know?",
                key="question_input",
            )
        with col2:
            ask = st.button("Ask", type="primary")

        if st.button("🗑️ Clear Conversation & Memory"):
            st.session_state.conversation_history = []
            st.session_state.user_question = ""
            if "memory_manager" in st.session_state:
                st.session_state.memory_manager.clear()
            st.rerun()

        if ask and question:
            st.session_state.user_question = question
            self._process(question)

    def _process(self, question: str):
        if not self.rag_pipeline:
            st.error("❌ Pipeline not ready")
            return

        is_proj = _is_project_query(question)

        with st.spinner("🔍 Searching CV…"):
            try:
                result = (
                    self.rag_pipeline.query_projects(question)
                    if is_proj
                    else self.rag_pipeline.query_standard(question)
                )

                if result:
                    answer = result.get("result", "No answer found.")
                    sources = []
                    for doc in result.get("source_documents", []):
                        sec = doc.metadata.get("section", "Unknown")
                        preview = doc.page_content[:100].replace("\n", " ")
                        if len(doc.page_content) > 100:
                            preview += "…"
                        sources.append(f"[{sec}] {preview}")

                    st.session_state.conversation_history.append(
                        (question, answer, sources, is_proj)
                    )
                    st.rerun()
                else:
                    st.error("❌ No response from assistant")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # ── Quick Facts ───────────────────────────────────────────────────────────

    def _quick_facts(self):
        st.markdown("---")
        st.subheader("📊 Quick CV Facts")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Programming Languages", "4+")
            st.metric("Projects Completed", "7")
        with c2:
            st.metric("Certificates", "8+")
            st.metric("Technologies", "15+")
        with c3:
            st.metric("Education", "B.Eng Mechatronics")
            st.metric("Experience", "2+ years")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self._sidebar()
        self._main()
        self._quick_facts()


# ── Run ───────────────────────────────────────────────────────────────────────

def main():
    StreamlitApp().run()


if __name__ == "__main__":
    main()
