"""Build a section-aware Chroma + BM25 index from the Formula Student rulebook."""
import argparse
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = BASE_DIR / "chroma"
BM25_PATH = BASE_DIR / "bm25_index.pkl"
EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION_NAME = "curt_knowledge"
SECTION_PATTERN = re.compile(r"^SECTION\s+(?P<id>[A-Z]+)\s*[–-]\s*(?P<title>.+)$", re.I)
RULE_PATTERN = re.compile(r"^(?P<id>[A-Z]{1,4}\d+(?:\.\d+)*)\s+(?P<title>.+)$")


class ChromaDBBuilder:
    def __init__(self, embedding_model=EMBEDDING_MODEL, chroma_dir=CHROMA_DIR,
                 bm25_path=BM25_PATH, collection_name=COLLECTION_NAME):
        self.embedding_model, self.chroma_dir = embedding_model, Path(chroma_dir)
        self.bm25_path, self.collection_name = Path(bm25_path), collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200,
                                                         separators=["\n\n", "\n", ". ", " ", ""])

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []
        for pdf_path in DATA_DIR.glob("**/*.pdf"):
            reader = PdfReader(str(pdf_path))
            parent_id = parent_title = None
            rule: Optional[Dict] = None

            def flush():
                nonlocal rule
                if not rule:
                    return
                body = "\n".join(rule["lines"]).strip()
                # Atomic rules often have their complete text on the heading line.
                if body or rule["id"].count(".") >= 2:
                    text = f"{rule['id']} {rule['title']}" + (f"\n{body}" if body else "")
                    metadata = {"source": str(pdf_path), "section": rule["id"],
                                "section_title": rule["title"], "parent_section": rule["parent"],
                                "parent_section_title": rule["parent_title"], "page": rule["start_page"],
                                "end_page": rule["end_page"]}
                    for part, content in enumerate(self.splitter.split_text(text)):
                        documents.append(Document(page_content=content, metadata={**metadata, "rule_part": part}))
                rule = None

            # Pages 1–3 are the contents page, not authoritative rule clauses.
            for page_number, page in enumerate(reader.pages[3:], start=4):
                for raw_line in (page.extract_text() or "").splitlines():
                    line = re.sub(r"\s+", " ", raw_line).strip()
                    if not line or line.startswith("©") or line.endswith("Formula Student") or line == "2025 Rules":
                        continue
                    section = SECTION_PATTERN.match(line)
                    if section:
                        flush(); parent_id, parent_title = section.group("id").upper(), section.group("title").strip(); continue
                    heading = RULE_PATTERN.match(line)
                    if heading:
                        flush()
                        rule = {"id": heading.group("id").upper(), "title": heading.group("title").strip(),
                                "parent": parent_id, "parent_title": parent_title, "start_page": page_number,
                                "end_page": page_number, "lines": []}
                    elif rule:
                        rule["lines"].append(line); rule["end_page"] = page_number
            flush()
        for chunk_id, document in enumerate(documents):
            document.metadata["chunk_id"] = chunk_id
        print(f"Loaded {len(documents)} section-aware rule chunks")
        return documents

    def build_database(self):
        documents = self.load_documents()
        if not documents: raise RuntimeError("No rule chunks found")
        if self.chroma_dir.exists(): shutil.rmtree(self.chroma_dir)
        store = Chroma.from_documents(documents, self.embeddings, persist_directory=str(self.chroma_dir),
                                      collection_name=self.collection_name)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as output:
            pickle.dump({"bm25": BM25Okapi([re.findall(r"\w+", d.page_content.lower()) for d in documents]),
                         "corpus_texts": [d.page_content for d in documents],
                         "metadatas": [d.metadata for d in documents]}, output)
        print(f"Built index with {len(documents)} chunks using {self.embedding_model}")
        return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL); parser.add_argument("--chroma-dir", default=str(CHROMA_DIR))
    parser.add_argument("--bm25-path", default=str(BM25_PATH)); parser.add_argument("--collection-name", default=COLLECTION_NAME)
    args = parser.parse_args()
    ChromaDBBuilder(args.embedding_model, args.chroma_dir, args.bm25_path, args.collection_name).build_database()
