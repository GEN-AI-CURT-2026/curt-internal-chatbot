import os, pickle, re
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import build_chroma as config
import prompts

load_dotenv()

class CURTRagPipeline:
    def __init__(self, embedding_model=config.EMBEDDING_MODEL, chroma_dir=config.CHROMA_DIR,
                 bm25_path=config.BM25_PATH, collection_name=config.COLLECTION_NAME,
                 retrieval_mode="hybrid"):
        self.retrieval_mode, self.bm25_path = retrieval_mode, Path(bm25_path)
        self.llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-5.4-mini"), temperature=0)
        embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings,
                                collection_name=collection_name)
        self.validation_chain = prompts.validation_prompt_template | self.llm.with_structured_output(prompts.ValidationDecision)
        self.bm25_data = self._load_bm25()

    def _load_bm25(self):
        try:
            with open(self.bm25_path, "rb") as source: return pickle.load(source)
        except FileNotFoundError: return None

    def _retrieve(self, query: str, limit=6) -> List[Document]:
        dense = [doc for doc, _ in self.vector_db.similarity_search_with_score(query, k=20)]
        if self.retrieval_mode == "dense" or not self.bm25_data: return dense[:limit]
        scores = self.bm25_data["bm25"].get_scores(re.findall(r"\w+", query.lower()))
        sparse_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
        dense_by_id = {doc.metadata.get("chunk_id"): doc for doc in dense}
        ranked, seen = [], set()
        for rank, chunk_id in enumerate([d.metadata.get("chunk_id") for d in dense] + sparse_ids):
            if chunk_id in seen: continue
            seen.add(chunk_id)
            if chunk_id in dense_by_id: doc = dense_by_id[chunk_id]
            else:
                doc = Document(self.bm25_data["corpus_texts"][chunk_id], self.bm25_data["metadatas"][chunk_id])
            ranked.append(doc)
            if len(ranked) == limit: break
        return ranked

    @staticmethod
    def _context(docs):
        return "\n\n---\n\n".join(f"[section={d.metadata.get('section')}; page={d.metadata.get('page')}]\n{d.page_content}" for d in docs)

    def run(self, query: str, chat_history: List[Dict] | None = None) -> Dict[str, Any]:
        if prompts.is_greeting(query): return {"answer": prompts.GREETING_RESPONSE, "sources": [], "status": "greeting"}
        if prompts.is_vague_query(query): return {"answer": prompts.NEEDS_CLARIFICATION_RESPONSE, "sources": [], "status": "needs_clarification"}
        docs = self._retrieve(query)
        if not docs: return {"answer": prompts.NO_CONTEXT_RESPONSE, "sources": [], "status": "no_docs"}
        decision = self.validation_chain.invoke({"context": self._context(docs), "question": query,
                                                 "chat_history": prompts.format_chat_history(chat_history or [])})
        cited = [d for d in docs if (d.metadata.get("section") or "").upper() == (decision.cited_section or "").upper()]
        if decision.verdict != "insufficient_evidence" and not cited:
            decision = prompts.ValidationDecision(verdict="insufficient_evidence", justification="No retrieved section supports the proposed citation.")
        elif cited:
            decision.cited_section, decision.cited_page = cited[0].metadata.get("section"), cited[0].metadata.get("page")
        if decision.verdict == "insufficient_evidence":
            answer = "Insufficient evidence in the retrieved rulebook sections to validate this update."
            cited = []
        else:
            answer = f"This is a {decision.verdict} car update because {decision.justification}. Source: section {decision.cited_section} (page {decision.cited_page})"
        return {"answer": answer, "raw_answer": decision.model_dump_json(), "sources": cited, "status": "success"}
