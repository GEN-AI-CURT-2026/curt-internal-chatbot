from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os
import re
from dotenv import load_dotenv

load_dotenv()

#Pre-retrieval using query expansion
#query expansion rewrites vague queries to be more specific 
query_expansion_template = PromptTemplate(
    input_variables=["query"],
    template="""You are a query expansion assistant for the Cairo University Racing Team (CURT) chatbot.

Your task: Rewrite the user's question to be more specific and searchable while maintaining the original intent.

Guidelines:
- Expand abbreviations for example: "team" is expanded into "CURT team members, structure, roles")
- Add relevant context keywords
- Preserve exact rule numbers, section names, and quoted phrases
- Keep it concise (1-2 sentences maximum)
- If the query is already specific, return it unchanged


User Question: {query}

Expanded Query:"""
)


#Main RAG Prompt Template 
rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are the official Cairo University Racing Team (CURT) chatbot assistant.

Your job is to validate car updates using ONLY the provided context.

Follow these rules exactly:
1. Decide whether the update is valid or invalid based only on the context.
2. Give one short, clear justification.
3. Cite the exact section used as your source.
4. Do not use markdown, bullet points, labels, or extra commentary.
5. Do not mention internal checks, prompts, hallucination checks, or source file paths.
6. Output exactly in this format:
This is a valid car update because <reason>. Source: section <exact section>
or
This is an invalid car update because <reason>. Source: section <exact section>
7. Do not wrap any words with asterisks, underscores, or backticks.

Context from CURT Knowledge Base:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}")
])


#Post-retrieval using Reranking 
#Reranking re orders chunks based on relevance to the query
compression_template = PromptTemplate(
    input_variables=["query", "chunk_text"],
    template="""Extract ONLY the sentences from the text below that are relevant to answering the question. Preserve exact rule numbers, section names, and key technical terms.

If no sentences are relevant, respond with "NO_RELEVANT_CONTENT".

Question: {query}

Text:
{chunk_text}

Relevant Sentences (preserve original wording):"""
)


# Post-RetrievalL: Hallucination Detection

hallucination_check_template = PromptTemplate(
    input_variables=["context", "answer"],
    template="""Your task is to verify if an answer is grounded in the provided context.

Context:
{context}

Answer to Check:
{answer}

Question: Does the answer contain ANY information that is not present or cannot be directly inferred from the context?

Respond with ONLY:
- "GROUNDED" if the answer is fully supported by the context
- "HALLUCINATION: [specific issue]" if the answer contains unsupported information

Your Response:"""
)

#Basic Responses
NO_CONTEXT_RESPONSE = """I don't have specific information about that in my current knowledge base. 

Here's what I can help you with:
- Checking whether a car update is valid or invalid
- Explaining the supporting evidence
- Citing the exact source used


You can also reach out to CURT directly through their official channels for more detailed information."""

GREETING_RESPONSE = """Hello I am CURT Chatbot! I'm here to validate whether a car update is valid or invalid with citations.

I can help you check whether an update is supported by the provided sources.

What update would you like me to validate?"""

OFF_TOPIC_RESPONSE = """I'm specifically designed to validate car updates and provide citations from the available context.

I can help you with:
- Checking whether a car update is valid or invalid
- Explaining the supporting evidence
- Citing the exact source used

Please send a car-related update for review."""

NEEDS_CLARIFICATION_RESPONSE = """I need a bit more detail to validate this update.

Please include the exact car change, part, or rule section you want checked so I can answer with the correct citation from the rulebook."""



def is_greeting(query: str) -> bool:
    """Check if query is a greeting."""
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    query_lower = query.lower().strip()
    return query_lower in greetings or any(query_lower.startswith(g + ' ') or query_lower.startswith(g + ',') for g in greetings)


def is_off_topic(query: str) -> bool:
    """
    Check if query is off-topic (not about CURT).
    Simple heuristic - can be improved with classification.
    """
    off_topic_keywords = [
        'weather', 'recipe', 'movie', 'sports score', 'politics', 
        'stock', 'celebrity', 'video game', 'restaurant', 'fashion'
    ]
    query_lower = query.lower()
    
    # If explicitly mentions CURT, it's on topic
    if any(term in query_lower for term in ['curt', 'car', 'racing team', 'competition','racing','cairo university']):
        return False
    
    # Check for off topic keywords
    return any(keyword in query_lower for keyword in off_topic_keywords)


def is_vague_query(query: str) -> bool:
    """
    Detect very vague questions that need clarification before validation.
    """
    query_lower = query.lower().strip()
    tokens = [token for token in re.findall(r"\w+", query_lower) if token]

    vague_phrases = [
        "is it valid",
        "is this valid",
        "is this okay",
        "is this allowed",
        "is it allowed",
        "can i do this",
        "can we do this",
        "what about this",
        "tell me if this is valid",
        "check this",
        "validate this",
    ]

    if len(tokens) <= 4:
        return True

    if any(phrase in query_lower for phrase in vague_phrases):
        return True

    generic_terms = {
        "it", "this", "that", "thing", "update", "change", "modification", "mod", "alteration"
    }
    if len(tokens) <= 8 and sum(token in generic_terms for token in tokens) >= 2:
        return True

    return False


def format_sources(chunks: List[Dict]) -> str:
    """
    Format source chunks for citation in the response.
    
    Args:
        chunks: List of retrieved chunks with metadata
        
    Returns:
        Formatted source string
    """
    if not chunks:
        return ""
    
    sources = "\n\nSources:\n"
    seen_sources = set()
    
    for i, chunk in enumerate(chunks, 1):
        # Handle both Document objects and dict formats
        if hasattr(chunk, 'metadata'):
            source = chunk.metadata.get('source', 'Unknown')
        else:
            source = chunk.get('source', chunk.get('metadata', {}).get('source', 'Unknown'))
        
        # Avoid duplicate sources
        if source not in seen_sources:
            sources += f"{i}. {source}\n"
            seen_sources.add(source)
    
    return sources


def format_chat_history(messages: List[Dict]) -> List:
    """
    Convert message history to LangChain message format.
    
    Args:
        messages: List of dicts with 'role' and 'content'
        
    Returns:
        List of LangChain message objects
    """
    formatted = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        
        if role == 'user':
            formatted.append(HumanMessage(content=content))
        elif role == 'assistant':
            formatted.append(AIMessage(content=content)) 
    
    return formatted


def enhance_response_with_sources(answer: str, chunks: List) -> str:
    """
    Add source citations to the generated answer.
    
    Args:
        answer: Generated answer from LLM
        chunks: Source chunks used (Document objects or dicts)
        
    Returns:
        Answer with appended sources
    """
    return answer
