"""
rag_pipeline.py  (Memory-Enhanced)
====================================
Original RAG pipeline + MemoryManager integration.

Changes vs original
-------------------
- Accepts an optional MemoryManager instance.
- Both query_standard() and query_projects() inject the memory context
  into the prompt and record the answer back into memory after the call.
- The prompt templates have a new {memory_context} variable.

Author : Khaled Ashraf  |  CURT Gen-AI Team — Memory & R&D
"""

import os
import json
import shutil
import glob
import tempfile
import re
from typing import List, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Memory module (place memory_manager.py in the same directory)
from memory_manager import MemoryManager

load_dotenv()

# --- Configuration ---
VECTOR_DB_DIR = os.path.join(tempfile.gettempdir(), f"cv_vector_db_{os.getpid()}")
CHUNKS_FILE = "cv_chunks_optimized.json"


class RAGPipeline:
    """
    RAG pipeline with hybrid buffer+summary memory.

    Parameters
    ----------
    memory : MemoryManager | None
        Pass a pre-built MemoryManager or leave None to create a default one.
    """

    def __init__(self, memory: Optional[MemoryManager] = None):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")

        # Clean up old temp vector stores
        for old in glob.glob(os.path.join(tempfile.gettempdir(), "cv_vector_db_*")):
            shutil.rmtree(old, ignore_errors=True)

        # Embeddings & LLM
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
        )
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            temperature=0.1,
            google_api_key=self.api_key,
        )

        self.vectorstore = None
        self.qa_chain_standard = None

        # --- Memory ---
        self.memory: MemoryManager = memory or MemoryManager(
            llm_model=self.llm_model,
            api_key=self.api_key,
            buffer_size=3,                    # N = 3 per CURT spec
            persistence_path="memory_state.json",
        )

        print("✅ RAGPipeline (memory-enhanced) initialised.")

    # ------------------------------------------------------------------
    # Setup helpers (unchanged from original)
    # ------------------------------------------------------------------

    def load_and_prepare_documents(self) -> List[Document]:
        if not os.path.exists(CHUNKS_FILE):
            print(f"❌ {CHUNKS_FILE} not found. Run the data-processing step first.")
            return []
        print(f"📄 Loading documents from {CHUNKS_FILE}…")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=item["content"], metadata=item["metadata"]) for item in data]
        print(f"   Loaded {len(docs)} chunks.")
        return docs

    def create_vector_store(self, documents: List[Document]) -> bool:
        print(f"💾 Creating vector store at: {VECTOR_DB_DIR}")
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=VECTOR_DB_DIR,
            )
            print("   Vector store ready.")
            return True
        except Exception as e:
            print(f"❌ Vector store error: {e}")
            return False

    def setup_qa_chain(self) -> bool:
        if not self.vectorstore:
            print("❌ Vector store not initialised.")
            return False

        # NOTE: {memory_context} is the new variable injected by MemoryManager
        standard_template = """
        You are an AI assistant specialised in answering questions about Khaled Ashraf's CV.
        Maintain a professional, concise, and helpful tone.

        If the user asks something unrelated to Khaled's CV, politely redirect them.

        If the user asks about Khaled's focus field, synthesise from his Education 
        (B.Eng Mechatronics), Skills (Embedded Systems, PLCs, C/C++, CAD), Projects 
        (Robotics, Clinic Management System), and Extracurriculars (ASME Robotics, CURT Gen AI).
        The evidence points strongly towards Mechatronics, Embedded Systems, and Robotics.

        --- CONVERSATION MEMORY ---
        {memory_context}
        --------------------------

        Context from CV:
        {context}

        Question: {question}
        Answer:"""

        STANDARD_PROMPT = PromptTemplate(
            template=standard_template,
            input_variables=["memory_context", "context", "question"],
        )

        self.qa_chain_standard = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 6}),
            chain_type_kwargs={"prompt": STANDARD_PROMPT},
            return_source_documents=True,
        )
        print("🔗 Standard QA Chain ready.")
        return True

    # ------------------------------------------------------------------
    # Query methods  (memory-aware)
    # ------------------------------------------------------------------

    def query_standard(self, question: str) -> dict:
        """Standard RAG query with memory context injected."""
        if not self.qa_chain_standard:
            print("❌ Standard QA chain not initialised.")
            return {}

        memory_ctx = self.memory.get_context()
        print(f"🧠 Memory context injected ({len(memory_ctx)} chars)")
        print(f"🔍 Standard Query: {question}")

        result = self.qa_chain_standard.invoke({
            "query": question,
            "memory_context": memory_ctx,   # ← injected into prompt
        })

        # Record this exchange into memory
        if result:
            self.memory.add_exchange(
                question=question,
                answer=result.get("result", ""),
                metadata={"query_type": "standard"},
            )

        return result

    def query_projects(self, question: str) -> dict:
        """Project-focused RAG query with memory context injected."""
        if not self.vectorstore:
            print("❌ Vector store not initialised.")
            return {}

        project_template = """
        You are a precise AI analyst. Summarise the requested project(s) from the context,
        including technologies, timeline, and main outcomes.
        Base your answer STRICTLY on the context provided.

        --- CONVERSATION MEMORY ---
        {memory_context}
        --------------------------

        Context (Projects):
        {context}

        Question: {question}
        Answer:"""

        PROJECT_PROMPT = PromptTemplate(
            template=project_template,
            input_variables=["memory_context", "context", "question"],
        )

        project_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "filter": {"section": "Projects"}},
        )
        project_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=project_retriever,
            chain_type_kwargs={"prompt": PROJECT_PROMPT},
            return_source_documents=True,
        )

        memory_ctx = self.memory.get_context()
        print(f"🧠 Memory context injected ({len(memory_ctx)} chars)")
        print(f"🔍 Project Query: {question}")

        result = project_chain.invoke({
            "query": question,
            "memory_context": memory_ctx,
        })

        if result:
            self.memory.add_exchange(
                question=question,
                answer=result.get("result", ""),
                metadata={"query_type": "projects"},
            )

        return result

    # ------------------------------------------------------------------
    # Memory convenience pass-throughs
    # ------------------------------------------------------------------

    def clear_memory(self) -> None:
        """Wipe conversation memory."""
        self.memory.clear()

    def memory_stats(self) -> str:
        """Pretty-print memory statistics."""
        return self.memory.pretty_stats()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("🚀 RAGPipeline (Memory-Enhanced) — Test Run")
    print("=" * 60)

    try:
        pipeline = RAGPipeline()
        docs = pipeline.load_and_prepare_documents()
        if not docs:
            exit()
        if not pipeline.create_vector_store(docs):
            exit()
        if not pipeline.setup_qa_chain():
            exit()

        print("\n🎯 Pipeline ready. Running multi-turn test…\n")

        turns = [
            ("What projects has Khaled worked on?", False),
            ("Which of those involved hardware?", False),           # ← uses memory
            ("Tell me about the Autonomous RC car.", True),
            ("What controller did he use for that?", True),         # ← uses memory
            ("What is his education background?", False),
        ]

        for question, is_project in turns:
            print(f"\n{'='*55}")
            print(f"❓ Q: {question}")
            result = pipeline.query_projects(question) if is_project else pipeline.query_standard(question)
            print(f"💬 A: {result.get('result', 'No answer')[:300]}")

        print(f"\n{'='*55}")
        print(pipeline.memory_stats())
        print("\n📋 Full Memory Context:")
        print(pipeline.memory.get_context())

    except Exception as e:
        import traceback
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
