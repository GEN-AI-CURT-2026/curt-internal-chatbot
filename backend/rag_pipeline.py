import os
import pickle
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import prompts
import build_chroma as chroma_config 

try:
    import cohere
except ImportError:
    cohere = None

# Load environment variables
load_dotenv()

class CURTRagPipeline:
    def __init__(self):
        """
        Initialize the RAG pipeline using configuration from build_chroma.py
        """
        self.llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
        #query expansioin using prompts.py
        self.expansion_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3)
        #for reranking
        self.cohere_client = None
        if cohere and os.getenv("COHERE_API_KEY"):
            self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

        #Load Vector Store and connect to chromadb
        db_path = str(chroma_config.CHROMA_DIR)
        model_name = chroma_config.EMBEDDING_MODEL
        collection_name = chroma_config.COLLECTION_NAME

        # print(f"Loading Vector Store from: {db_path}")
        # print(f"Using Embedding Model: {model_name}")
        
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 6})  #top 6 most relevant chunks
        self.bm25_data = self._load_bm25_index()

        self._init_chains()

    def _init_chains(self):
        """Initialize all LangChain runnables"""
        
        self.expansion_chain = (
            prompts.query_expansion_template  #from prompts.py
            | self.expansion_llm 
            | StrOutputParser()
        )

        self.compression_chain = (
            prompts.compression_template 
            | self.llm 
            | StrOutputParser()
        )

        self.rag_chain = (
            prompts.rag_prompt_template 
            | self.llm 
            | StrOutputParser()
        )

        self.hallucination_chain = (
            prompts.hallucination_check_template 
            | self.llm 
            | StrOutputParser()
        )

    def _load_bm25_index(self):
        """Load the BM25 sparse index created by build_chroma.py."""
        bm25_path = chroma_config.BM25_PATH
        if not bm25_path.exists():
            print(f"BM25 index not found at {bm25_path}; hybrid retrieval will fall back to dense only.")
            return None

        try:
            with open(bm25_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            return None

    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize a query for BM25 scoring."""
        return re.findall(r"\w+", query.lower())

    def hybrid_search(self, query: str, n_results: int = 5, fetch_k: int = 20) -> List[Document]:
        """
        Hybrid retrieval: dense Chroma + sparse BM25 fused with RRF.

        fetch_k = how many candidates each retriever fetches before fusion
        n_results = how many final chunks to return to the LLM
        """
        dense_results = self.vector_db.similarity_search_with_score(query, k=fetch_k)
        dense_docs = [doc for doc, _ in dense_results]
        dense_ids = [doc.metadata.get("chunk_id", i) for i, doc in enumerate(dense_docs)]

        if not self.bm25_data:
            return dense_docs[:n_results]

        bm25 = self.bm25_data["bm25"]
        corpus_texts = self.bm25_data["corpus_texts"]
        metadatas = self.bm25_data["metadatas"]

        tokenized_query = self._tokenize_query(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        sparse_ids_sorted = sorted(
            range(len(corpus_texts)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:fetch_k]

        fused_ids = rrf_fusion(dense_ids, sparse_ids_sorted)[:n_results]

        dense_by_id = {
            doc.metadata.get("chunk_id", i): doc
            for i, doc in enumerate(dense_docs)
        }

        final_chunks = []
        seen_texts = set()

        for fid in fused_ids:
            if fid in dense_by_id:
                doc = dense_by_id[fid]
            else:
                if fid >= len(corpus_texts):
                    continue
                doc = Document(
                    page_content=corpus_texts[fid],
                    metadata=metadatas[fid],
                )

            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                final_chunks.append(doc)

        return final_chunks or dense_docs[:n_results]

    def _rerank_with_cohere(self, query: str, documents: List, top_n: int = 5) -> List:
        """Rerank documents using Cohere's reranking API."""
        if not documents:
            return []

        if not self.cohere_client:
            return documents[:top_n]
        
        doc_texts = [doc.page_content for doc in documents]
        
        try:
            rerank_response = self.cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=doc_texts,
                top_n=top_n,
                return_documents=True
            )
            
            reranked_docs = []
            for result in rerank_response.results:
                original_doc = documents[result.index]
                original_doc.metadata['rerank_score'] = result.relevance_score
                reranked_docs.append(original_doc)
            
            print(f"Cohere Reranking: {len(documents)} → {len(reranked_docs)} docs")
            
            return reranked_docs
            
        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            return documents[:top_n]

    def _compress_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Compress retrieved chunks down to only the sentences relevant to the query."""
        if not documents:
            return []

        compressed_docs: List[Document] = []

        for doc in documents:
            compressed_text = self.compression_chain.invoke({
                "query": query,
                "chunk_text": doc.page_content,
            }).strip()

            if not compressed_text or compressed_text.upper() == "NO_RELEVANT_CONTENT":
                continue

            compressed_docs.append(
                Document(
                    page_content=compressed_text,
                    metadata=dict(doc.metadata),
                )
            )

        return compressed_docs

    def _sanitize_answer(self, answer: str) -> str:
        """Convert model output into a single plain-text validation sentence."""
        text = (answer or "").strip()

        # Remove common markdown and noise the model may emit.
        text = re.sub(r"`+", "", text)
        text = re.sub(r"[*_]+", "", text)

        unwanted_phrases = [
            "I don't have that information in my knowledge base. You can contact CURT directly for more details.",
            "I don't have that information in my knowledge base.",
            "You can contact CURT directly for more details.",
            "Please verify with official team documents.",
            "*(Note:",
            "(Note:",
        ]
        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")

        # Drop any explicit source list appended by the model.
        source_markers = ["\nSources:", "\n\nSources:", "\nSource:", "\n\nSource:"]
        cut_points = [text.find(marker) for marker in source_markers if marker in text]
        if cut_points:
            text = text[: min(cut_points)].strip()

        # Prefer the exact requested output if the model followed it.
        pattern = re.search(
            r"(This is an (?:valid|invalid) car update because ./n*?Source:\s*section\s*[^\n.]+(?:\.)? Citation: )",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if pattern:
            text = pattern.group(1).strip()

        # Normalize whitespace and remove spaces before punctuation.
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([.,:;!?])", r"\1", text)

        return text
        
    def run(self, query: str, chat_history: List[Dict] = []) -> Dict[str, Any]:
        
        if prompts.is_greeting(query):
            return {"answer": prompts.GREETING_RESPONSE, "sources": [], "status": "greeting"}
        
        if prompts.is_off_topic(query):
            return {"answer": prompts.OFF_TOPIC_RESPONSE, "sources": [], "status": "off_topic"}

        if prompts.is_vague_query(query):
            return {
                "answer": prompts.NEEDS_CLARIFICATION_RESPONSE,
                "sources": [],
                "status": "needs_clarification",
            }

        expanded_query = self.expansion_chain.invoke({"query": query})
        print(f"Expanded Query: '{expanded_query}'")

        # Retrieval 
        raw_docs = self.hybrid_search(expanded_query, n_results=10, fetch_k=20)
        #print(f"Retrieved {len(raw_docs)} raw documents")
        
        if not raw_docs:
            return {"answer": prompts.NO_CONTEXT_RESPONSE, "sources": [], "status": "no_docs"}

        #Reranking using cohere
        valid_docs = self._rerank_with_cohere(query, raw_docs, top_n=5)
        compressed_docs = self._compress_documents(query, valid_docs)

        if not compressed_docs:
            print("Compression removed all retrieved content; falling back to reranked chunks.")
            compressed_docs = valid_docs

        compressed_context = "\n\n".join([doc.page_content for doc in compressed_docs])
            
        # Generation
        formatted_history = prompts.format_chat_history(chat_history)
        
        answer = self.rag_chain.invoke({
            "context": compressed_context,
            "question": query, 
            "chat_history": formatted_history
        })

        #Hallucination Detection
        check_result = self.hallucination_chain.invoke({
            "context": compressed_context,
            "answer": answer
        })
        
        print(f"Verification: {check_result}")

        if check_result.strip().upper().startswith("HALLUCINATION"):
            print("Warning: generated answer may contain unsupported details.")

        final_response = self._sanitize_answer(answer)
        return {
            "answer": final_response,
            "raw_answer": answer,
            "sources": compressed_docs,
            "expanded_query": expanded_query,
            "status": "success"
        }

def take_input(input):
    """Function to take input """
    return input


def rrf_fusion(dense_ids, sparse_ids, k=60):
    """Reciprocal Rank Fusion merges ranked lists by rank position."""
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


if __name__ == "__main__":
    try:
        pipeline = CURTRagPipeline()
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure you have run 'python build_chroma.py' first!")
