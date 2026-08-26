from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

class ValidationDecision(BaseModel):
    verdict: Literal["valid", "invalid", "insufficient_evidence"]
    justification: str = Field(description="Short reason supported only by the retrieved excerpt.")
    cited_section: Optional[str] = Field(default=None, description="Exact section identifier from the excerpt.")
    cited_page: Optional[int] = None

query_expansion_template = PromptTemplate.from_template(
    "Rewrite this Formula Student rulebook question for retrieval. Keep rule ids and all technical details. Question: {query}"
)
compression_template = PromptTemplate.from_template(
    "Extract only text relevant to {query}. Preserve rule wording and exceptions. Reply NO_RELEVANT_CONTENT if none.\n\n{chunk_text}"
)
validation_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You validate Formula Student car updates from retrieved excerpts only.
Each excerpt has [section=...; page=...]. Cite only an exact section identifier present there.
Return insufficient_evidence whenever the excerpts do not conclusively decide the update.
Do not assume facts not in the excerpts; include applicable exceptions in the justification."""),
    ("system", "Retrieved excerpts:\n{context}"),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

GREETING_RESPONSE = "Hello. Describe a proposed car update and I will validate it against the rulebook."
OFF_TOPIC_RESPONSE = "I can only validate Formula Student car updates against the available rulebook."
NEEDS_CLARIFICATION_RESPONSE = "Please describe the proposed car change in enough detail to validate it."
NO_CONTEXT_RESPONSE = "I could not find a relevant rulebook section for that update."

def is_greeting(query: str) -> bool: return query.lower().strip() in {"hi", "hello", "hey"}
def is_off_topic(query: str) -> bool: return False
def is_vague_query(query: str) -> bool: return len(query.split()) < 3
def format_chat_history(messages: List[Dict]):
    return [HumanMessage(content=m["content"]) if m.get("role") == "user" else AIMessage(content=m["content"])
            for m in messages if m.get("role") in {"user", "assistant"}]
