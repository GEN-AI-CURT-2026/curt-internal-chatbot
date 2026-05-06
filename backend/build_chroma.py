import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = Path("backend/chroma")
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 700 
CHUNK_OVERLAP = 100 
COLLECTION_NAME = "curt_knowledge"


class ChromaDBBuilder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        
    def load_documents(self):
        """Load all documents from data directory using LangChain.
        
        Returns:
            List of LangChain Document objects
        """
        print("Step 1: Loading documents with LangChain...")
        
        all_documents = []
        
        try:
            md_loader = DirectoryLoader(
                str(DATA_DIR),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            md_docs = md_loader.load()
            all_documents.extend(md_docs)
            print(f"Loaded {len(md_docs)} markdown files")
        except Exception as e:
            print(f"Note: {e}")
        
        try:
            txt_loader = DirectoryLoader(
                str(DATA_DIR),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            txt_docs = txt_loader.load()
            all_documents.extend(txt_docs)
            print(f"Loaded {len(txt_docs)} text files")
        except Exception as e:
            print(f"Note: {e}")
            
        try:
            pdf_loader = DirectoryLoader(
                str(DATA_DIR),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
            )
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            print(f"Loaded {len(pdf_docs)} pdf files")
        except Exception as e:
            print(f"Note: {e}")
        
        if not all_documents:
            print(" No documents found!")
            return []
        
        total_chars = sum(len(doc.page_content) for doc in all_documents)
        print(f"\nTotal documents loaded: {len(all_documents)}")
        print(f"Total characters: {total_chars:,}\n")
        
        return all_documents
    
    def chunk_documents(self, documents):
        """Split documents into chunks using LangChain.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        print("Step 2: Chunking documents with LangChain...")
        
        chunks = self.text_splitter.split_documents(documents)

        print(f" Created {len(chunks)} chunks")
 
        source_counts = {}
        for chunk in chunks:
            source = Path(chunk.metadata.get("source", "unknown")).name
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("\n  Chunks per file:")
        for source, count in sorted(source_counts.items()):
            print(f"    {source}: {count} chunks")
        
        print(f"\n Total chunks created: {len(chunks)}\n")
        
        return chunks
    
    def build_database(self):
        print("\n" + "="*60)
        print("CURT CHATBOT - BUILDING CHROMADB WITH LANGCHAIN")
        print("="*60 + "\n")
        
        documents = self.load_documents()
        
        if not documents:
            print("No data files found! Please add .md, .txt, or .pdf files to /data/")
            return
        
        chunks = self.chunk_documents(documents)
        
        print("Step 3-4: Creating ChromaDB with embeddings...")
        print(f"  Embedding model: {EMBEDDING_MODEL}")
        print(f"  Generating embeddings and storing in ChromaDB...")
        
        if CHROMA_DIR.exists():
            import shutil
            try:
                shutil.rmtree(CHROMA_DIR)
                print(f"  Deleted existing database at {CHROMA_DIR}")
            except Exception as e:
                print(f"  Note: {e}")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
            collection_metadata={"description": "CURT Racing Team knowledge base"}
        )
        
        print(f"Created collection: {COLLECTION_NAME}")
        print(f"Generated and stored {len(chunks)} embeddings\n")

        print("Step 5: Verifying database...")
        
        # print("\n  Testing retrieval with similarity search...")
        # test_query = "What is CURT?"
        # test_results = vectorstore.similarity_search(test_query, k=3)
        
        # print(f"Retrieved {len(test_results)} relevant documents\n")
        
        # if test_results:
        #     print("  Sample result preview:")
        #     sample = test_results[0].page_content[:200]
        #     source = Path(test_results[0].metadata.get("source", "unknown")).name
        #     print(f"    Source: {source}")
        #     print(f"    Content: '{sample}...'\n")
        
        print("="*60)
        print("✓ CHROMADB BUILD COMPLETE!")
        print("="*60)
        print(f"Collection name: {COLLECTION_NAME}")
        print(f"Total chunks stored: {len(chunks)}")
        print(f"Embedding model: {EMBEDDING_MODEL}")
        test_embedding = self.embeddings.embed_query("test")
        print(f"Embedding dimension: {len(test_embedding)}")
        print("\n")
        
        return vectorstore

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    builder = ChromaDBBuilder()
    builder.build_database()

if __name__ == "__main__":
    main()
