# CURT Internal Chatbot

A Formula Student rulebook chatbot that validates proposed car updates against retrieved rule sections. It uses OpenAI embeddings and an LLM for retrieval-augmented generation (RAG), with dense or hybrid retrieval modes.

## Requirements

- Python 3.10 or later
- Node.js 18 or later
- An OpenAI API key
- A Formula Student rulebook PDF placed anywhere under `data/`

## Setup

From the project root, create and activate a virtual environment, then install the backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
# Optional: defaults to gpt-5.4-mini
CHAT_MODEL=gpt-5.4-mini
```

Install the frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

## Build the rulebook index

Put the rulebook PDF under `data/`, then run:

```powershell
python backend/build_chroma.py
```

This creates the local Chroma vector database in `backend/chroma/` and the BM25 index at `backend/bm25_index.pkl`. Both are generated files and are ignored by Git.

## Run the chatbot

Start the API in one terminal:

```powershell
python backend/api.py
```

The API is available at `http://localhost:8000`; interactive API documentation is at `http://localhost:8000/docs`.

Start the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

Open the URL printed by Vite, normally `http://localhost:5173`.

## API

`POST /chat` accepts a user message and an optional persistent session ID:

```json
{
  "message": "Validate this proposed vehicle update: Tyre warmers are not allowed.",
  "session_id": "optional-existing-session-id"
}
```

The response includes the answer, rulebook sources, server session ID, and number of messages retained in memory. The server stores conversations in `backend/conversation_memory.sqlite3` and passes the latest three messages to the model.

Reset a session with `DELETE /sessions/{session_id}`.

## Evaluate retrieval and decisions

The evaluator runs labelled cases from `data/evaluation_cases.json` against the configurations in `data/rag_experiments.json`.

Run the complete evaluation:

```powershell
python backend/evaluate_rag.py --workers 1
```

Run a five-case smoke test without replacing the main report:

```powershell
python backend/evaluate_rag.py --limit 5 --workers 1 --output data/evaluation_smoke.json
```

`--workers` controls concurrent model requests. Start with `1`, then increase only if OpenAI rate limits allow it. Results contain verdict accuracy and exact cited-section match.

### Important evaluation note

Review labels in `data/evaluation_cases.json` before treating the verdict score as a quality metric. Several cases labelled `invalid` state requirements that the cited rule explicitly confirms. The saved report is therefore useful for checking citation retrieval, but its verdict-accuracy figure is not currently a reliable measure of chatbot quality.

## Retrieval modes

`data/rag_experiments.json` defines the experiments:

- `dense-large`: semantic search using `text-embedding-3-large`
- `hybrid-large`: intended to combine semantic and BM25 retrieval

Both modes use the same Chroma collection and model. Rebuild the index whenever the source rulebook or indexing logic changes.

## Troubleshooting

- **`Could not import chromadb` / `cygrpc` DLL blocked:** Windows application-control policy is blocking Chroma's native gRPC dependency. Use a Python environment permitted by your organization, or ask IT to allow the installed `grpcio` native module. Reinstalling packages alone may not bypass this policy.
- **No rulebook results:** make sure a PDF exists under `data/` and run `python backend/build_chroma.py` again.
- **Authentication errors:** confirm `OPENAI_API_KEY` is set in `.env` and restart the API.

## Project structure

```text
backend/   API, index builder, RAG pipeline, prompts, and evaluator
frontend/  React/Vite chat interface
data/      Rulebook PDFs, evaluation cases, and reports
```
