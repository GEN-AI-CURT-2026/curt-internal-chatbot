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

RULEBOOK_TEST_CASES = [
    {
        "section": "A1.1.2",
        "expected": "valid",
        "query": "Validate this update: Formula Student has three entry classes: FS Class, FS-AI Class, and Concept Class.",
    },
    {
        "section": "A1.2.2",
        "expected": "valid",
        "query": "Validate this update: Every vehicle must meet the requirements in Section T and its powertrain-specific section, such as CV, EV, or AFV.",
    },
    {
        "section": "A2.2.1",
        "expected": "invalid",
        "query": "Validate this update: A vehicle may compete in Formula Student class in more than one competition year if it passes inspection.",
    },
    {
        "section": "A2.2.3",
        "expected": "valid",
        "query": "Validate this update: A new vehicle must have a newly manufactured chassis with significant changes in the Primary Structure compared to its predecessor.",
    },
    {
        "section": "A3.5.3",
        "expected": "valid",
        "query": "Validate this update: If there is a discrepancy between the rulebook and another official document, the rulebook takes priority.",
    },
    {
        "section": "IN1.2.1",
        "expected": "valid",
        "query": "Validate this update: A vehicle must pass all parts of technical inspection before it can enter any dynamic event.",
    },
    {
        "section": "IN1.4.1",
        "expected": "valid",
        "query": "Validate this update: For EV electrical inspection and accumulator inspection, the inspection responsible person must be an ESO.",
    },
    {
        "section": "IN1.5.1",
        "expected": "valid",
        "query": "Validate this update: After technical inspection, the team may adjust tyre pressure, brake bias, winglet angles, and software calibration, but not move the complete aerodynamic device.",
    },
    {
        "section": "IN7.1.3",
        "expected": "valid",
        "query": "Validate this update: The tilt test uses a 60 degree angle, maximum fluid levels, and the wheels must remain in contact with the surface.",
    },
    {
        "section": "IN8.1.2",
        "expected": "valid",
        "query": "Validate this update: During vehicle weighing, oil and coolant circuits must be at maximum fill level, and the fuel tank must be empty for CV vehicles.",
    },
    {
        "section": "IN11.1.2",
        "expected": "valid",
        "query": "Validate this update: In the EV brake test, the driver must switch off the tractive system and then brake using only the mechanical brakes.",
    },
    {
        "section": "IN13.2.7",
        "expected": "valid",
        "query": "Validate this update: All drivers must be able to exit the vehicle in no more than 5 seconds during the Driver Egress Test.",
    },
    {
        "section": "S2.3.4",
        "expected": "invalid",
        "query": "Validate this update: A BPP presentation can run for 12 minutes before any penalty is applied.",
    },
    {
        "section": "S2.3.10",
        "expected": "valid",
        "query": "Validate this update: Teams will not be supplied with an internet connection during BPP judging.",
    },
    {
        "section": "D7.5.4",
        "expected": "valid",
        "query": "Validate this update: During endurance driver change, the team gets three minutes to change the driver and one minute to restart the car.",
    },
    {
        "section": "D7.6.9",
        "expected": "valid",
        "query": "Validate this update: If a vehicle cannot maintain lap times within 145 percent of the fastest lap time for the course, it must exit immediately.",
    },
    {
        "section": "D7.7.5",
        "expected": "valid",
        "query": "Validate this update: If a vehicle has a restart problem after a red flag or at driver change, it has two minutes to restart the engine, enable the tractive system, or enter R2D.",
    },
    {
        "section": "D9.1.5",
        "expected": "valid",
        "query": "Validate this update: Off-course means all four wheels are outside the track boundary or a required slalom gate is missed.",
    },
    {
        "section": "D9.1.14",
        "expected": "valid",
        "query": "Validate this update: Vehicle-to-vehicle contact can lead to a time penalty or disqualification depending on the incident.",
    },
]

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
        
    def run(self, query: str, chat_history: List[Dict] = []) -> Dict[str, Any]:
        
        if prompts.is_greeting(query):
            return {"answer": prompts.GREETING_RESPONSE, "sources": [], "status": "greeting"}
        
        if prompts.is_off_topic(query):
            return {"answer": prompts.OFF_TOPIC_RESPONSE, "sources": [], "status": "off_topic"}

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
            answer += "\n\n*(Note: I verified this answer against my database and found some parts might not be explicitly supported. Please verify with official team documents.)*"

        final_response = prompts.enhance_response_with_sources(answer, compressed_docs)
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


def get_rulebook_test_inputs() -> List[str]:
    """Return ready-made user inputs derived from the rulebook."""
    return [case["query"] for case in RULEBOOK_TEST_CASES]


def rrf_fusion(dense_ids, sparse_ids, k=60):
    """Reciprocal Rank Fusion merges ranked lists by rank position."""
    scores = {}
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def run_rulebook_test_suite(pipeline: CURTRagPipeline) -> None:
    """Run the pipeline against the built-in rulebook test prompts."""
    for idx, case in enumerate(RULEBOOK_TEST_CASES, 1):
        print("\n" + "=" * 80)
        print(f"TEST {idx}: {case['section']} | expected: {case['expected']}")
        print(f"USER INPUT: {case['query']}")
        result = pipeline.run(case["query"])
        print("ANSWER:")
        print(result["answer"])
        print(f"STATUS: {result['status']}")

if __name__ == "__main__":
    try:
        pipeline = CURTRagPipeline()
        run_rulebook_test_suite(pipeline)
    except Exception as e:
        print(f"\nError: {e}")
        print("Ensure you have run 'python build_chroma.py' first!")
