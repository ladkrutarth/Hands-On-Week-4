"""
Veriscan — FastAPI Microservices Backend
Decoupled REST API for Fraud Prediction, GuardAgent, and RAG Engine.

Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
# ---------------------------------------------------------------------------
# System Stability Guards (Fixes SIGABRT on macOS Sequoia)
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import sys
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List
import anyio

from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil

# Ensure project root is on the path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Request

from api.schemas import (
    HighRiskTransactionsResponse,
    UserRiskResponse,
    AgentActionStep,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGResult,
    HealthResponse,
    AdvisorChatRequest,
    AdvisorChatResponse,
    SpendingDNAResponse,
    DNACompareRequest,
    DNACompareResponse,
    AuthLoginRequest,
    AuthLoginResponse,
    SecurityChatRequest,
    SecurityChatResponse,
    DocChatRequest,
    DocChatResponse,
)

from models.auth_store import get_user_store

# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------
_agent = None
_rag_engine = None
_advisor_agent = None
_advisor_load_error: Optional[str] = None  # Captured when advisor fails to load
_dna_agent = None
_login_failures: dict[str, int] = {}


def _session_id(request: Request, body_session_id: Optional[str] = None) -> Optional[str]:
    """Session ID from body, X-Session-ID header, or session_id query."""
    return body_session_id or request.headers.get("X-Session-ID") or request.query_params.get("session_id")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server boots.
    Agents are loaded as singletons to avoid redundant data reloading.
    """
    global _agent, _rag_engine, _advisor_agent, _advisor_load_error, _dna_agent
    print("🚀 Veriscan API — Loading resources...")

    # 1. RAG Engine
    try:
        from models.rag_engine_local import RAGEngineLocal
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
        print("✅ RAG Engine loaded.")
    except Exception as e:
        print(f"⚠️  RAG Engine failed: {e}")

    # 2. GuardAgent (MLX)
    try:
        from models.guard_agent_local import LocalGuardAgent
        _agent = LocalGuardAgent()
        print("✅ GuardAgent (Security Analyst) loaded.")
    except Exception as e:
        print(f"ℹ️  GuardAgent not loaded: {e}")

    # 3. Financial Advisor Agent (fast path + smart path)
    try:
        from agents.financial_advisor_agent import FinancialAdvisorAgent
        _advisor_agent = FinancialAdvisorAgent(llm=getattr(_agent, "llm", None) if _agent else None)
        _ = _advisor_agent.df
        _advisor_load_error = None
        print("✅ Financial Advisor Agent loaded (fast+smart path).")
    except Exception as e:
        _advisor_load_error = str(e)
        _advisor_agent = None
        print(f"⚠️  Financial Advisor failed: {e}")

    # 4. Spending DNA Agent
    try:
        from agents.spending_dna_agent import SpendingDNAAgent
        _dna_agent = SpendingDNAAgent()
        print("✅ Spending DNA Agent loaded.")
    except Exception as e:
        print(f"⚠️  DNA Agent failed: {e}")

    print("🟢 Veriscan API is ready.")
    yield
    print("🔴 Veriscan API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Veriscan — Fraud Intelligence API",
    description="Microservices backend for ML fraud prediction, agentic investigation, and RAG-powered knowledge retrieval.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 1. Health Check
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return API health status and loaded services."""
    return HealthResponse(
        status="operational",
        version="2.0.0",
        services={
            "guard_agent": "loaded" if _agent else "unavailable",
            "rag_engine": "loaded" if _rag_engine else "unavailable",
            "advisor_agent": "loaded" if _advisor_agent else (f"unavailable: {_advisor_load_error}" if _advisor_load_error else "unavailable"),
            "dna_agent": "loaded" if _dna_agent else "unavailable",
        },
    )


# ---------------------------------------------------------------------------
# 2. Auth: Login (ADDF-aware)
# ---------------------------------------------------------------------------
@app.post("/api/auth/login", response_model=AuthLoginResponse, tags=["Auth"])
async def auth_login(req: AuthLoginRequest):
    """
    Simple demo login endpoint.

    - Verifies username/password against an in-memory user store.
    - Computes a basic login risk score.
    - Uses DeceptionRouter to decide if this session should be diverted into ADDF.
    """
    store = get_user_store()
    username = req.username.strip()
    password = req.password

    # Verify credentials
    if not store.verify_user(username, password):
        # Track failures to increase risk on later attempts (per-process only).
        _login_failures[username] = _login_failures.get(username, 0) + 1
        return AuthLoginResponse(
            authenticated=False,
            session_id=None,
            diverted=False,
            message="Invalid username or password.",
        )

    # Successful auth → create session_id
    session_id = str(uuid.uuid4())

    # Basic login risk scoring (for demo only)
    risk = 0.0
    lower_name = username.lower()
    weak_pw = password.lower()

    # Suspicious usernames
    if lower_name in {"root", "admin_test", "hacker", "pentest"}:
        risk += 30.0

    # Obviously weak passwords (only for demo accounts)
    if weak_pw in {"password", "password123", "123456", "admin"}:
        risk += 15.0

    # Prior failures in this process
    failures = _login_failures.get(username, 0)
    if failures >= 3:
        risk += 10.0

    # Reset failure counter on successful login
    _login_failures[username] = 0

    return AuthLoginResponse(
        authenticated=True,
        session_id=session_id,
        diverted=False,
        message="Login successful.",
    )




# ---------------------------------------------------------------------------
# 3. High-Risk Transactions (session-aware: diverted → decoy)
# ---------------------------------------------------------------------------
@app.get("/api/fraud/high-risk", response_model=HighRiskTransactionsResponse, tags=["Fraud ML"])
async def get_high_risk_transactions(request: Request, limit: int = Query(default=10, ge=1, le=100), session_id: Optional[str] = Query(None)):
    """Get the top N highest-risk transactions."""
    from models.agent_tools_data import tool_get_high_risk_transactions
    results = tool_get_high_risk_transactions(limit=limit)
    return HighRiskTransactionsResponse(count=len(results), transactions=results)


# ---------------------------------------------------------------------------
# 4. User Risk Profile (session-aware: diverted → decoy)
# ---------------------------------------------------------------------------
@app.get("/api/user/{user_id}/risk", response_model=UserRiskResponse, tags=["User Intelligence"])
async def get_user_risk(user_id: str, request: Request, session_id: Optional[str] = Query(None)):
    """Retrieve the risk profile for a specific user."""
    from models.agent_tools_data import tool_get_user_risk_profile
    result = tool_get_user_risk_profile(user_id)
    return UserRiskResponse(**result)




# ---------------------------------------------------------------------------
# 6. RAG Query (ADDF: diverted → decoy)
# ---------------------------------------------------------------------------
@app.post("/api/rag/query", response_model=RAGQueryResponse, tags=["Knowledge Base"])
async def rag_query(req: RAGQueryRequest, request: Request):
    """Semantic search over the knowledge base."""
    if not _rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not loaded.")
    results = _rag_engine.query(req.query, n_results=req.n_results)
    parsed = [
        RAGResult(text=r["text"], confidence=r["confidence"], metadata=r.get("metadata"))
        for r in results
    ]
    return RAGQueryResponse(query=req.query, count=len(parsed), results=parsed)


@app.post("/api/rag/upload", tags=["Knowledge Base"])
async def rag_upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload one or more PDFs for indexing in the local RAG engine."""
    from models.rag_engine_local import PDF_DATA_PATH
    PDF_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    results = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename, "status": "skipped", "error": "Only PDF files are supported."})
            continue
        
        file_path = PDF_DATA_PATH / file.filename
        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            results.append({"filename": file.filename, "status": "indexed successfully"})
        except Exception as e:
            results.append({"filename": file.filename, "status": "failed", "error": str(e)})
    
    # Trigger re-indexing once after all uploads
    if _rag_engine:
        _rag_engine.index_data(force=True)
        
    return {"uploads": results}


@app.post("/api/rag/chat", response_model=DocChatResponse, tags=["Knowledge Base"])
async def rag_chat(req: DocChatRequest):
    """Conversational interface using RAG context."""
    if not _rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not loaded.")
    if not _agent:
        raise HTTPException(status_code=503, detail="LLM (GuardAgent) not loaded.")

    # 1. Retrieve context
    results = _rag_engine.query(req.message, n_results=4)
    context = "\n\n".join([f"[{r['type'].upper()}]: {r['text']}" for r in results])
    
    # 2. Build messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intelligent Document Assistant. Use the provided context to answer the user's question. "
                "If the answer is not in the context, say that you don't know based on the documents. "
                "Keep your response concise and professional.\n\n"
                f"CONTEXT:\n{context}"
            )
        },
        {
            "role": "user",
            "content": req.message
        }
    ]
    
    # 3. Generate reply
    try:
        # Use generate_chat_async for proper multi-turn / system prompt handling
        reply = await _agent.llm.generate_chat_async(messages, max_tokens=400, temp=0.1)
        sources = [
            {"text": r["text"], "metadata": r.get("metadata", {})}
            for r in results
        ]
        return DocChatResponse(reply=reply, sources=sources, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ===========================================================================
# NEW FEATURE ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Feature 1: AI Financial Advisor Chat (ADDF: diverted → decoy)
# ---------------------------------------------------------------------------
@app.post("/api/advisor/chat", response_model=AdvisorChatResponse, tags=["AI Financial Advisor"])
async def advisor_chat(req: AdvisorChatRequest, request: Request):
    """Fast path or smart path financial advice."""
    if not _advisor_agent:
        detail = f"Financial Advisor not loaded: {_advisor_load_error}" if _advisor_load_error else "Financial Advisor not loaded."
        raise HTTPException(status_code=503, detail=detail)
    result = await anyio.to_thread.run_sync(_advisor_agent.chat, req.message, req.user_id)
    return AdvisorChatResponse(
        user_id=req.user_id,
        message=req.message,
        reply=result.get("reply", ""),
        tool_results=result.get("tool_results", []),
    )


@app.get("/api/advisor/users", tags=["AI Financial Advisor"])
async def advisor_users(request: Request, session_id: Optional[str] = Query(None)):
    """Return all user IDs in the financial advisor dataset."""
    if not _advisor_agent:
        detail = f"Financial Advisor not loaded: {_advisor_load_error}" if _advisor_load_error else "Financial Advisor not loaded."
        raise HTTPException(status_code=503, detail=detail)
    return {"users": _advisor_agent.get_all_users()}


# ---------------------------------------------------------------------------
# Feature 2: AI Security Analyst Chat (risk-based + keyword diversion → fast decoy)
# ---------------------------------------------------------------------------
@app.post("/api/security/chat", response_model=SecurityChatResponse, tags=["AI Security Analyst"])
async def security_chat(req: SecurityChatRequest):
    """Security analyst."""
    if not _agent:
        raise HTTPException(status_code=503, detail="GuardAgent not loaded.")
    try:
        system_prompt = (
            "You are an elite AI Security Analyst. Your job is strictly to analyze "
            "security data, explain fraud risks, and provide safety protocols. "
            "Never provide financial advice, budgets, or savings plans.\n\n"
            "Format your response professionally:\n"
            "1. Be extremely concise (under 200 words max).\n"
            "2. Use bullet points and bold text for key findings.\n"
            "3. Provide direct, actionable safety advice without unnecessary filler text."
        )
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{req.message}<|im_end|>\n<|im_start|>assistant\n"
        # Slightly lower max_tokens for faster replies; model size controlled by VERISCAN_FAST_MODE/VERISCAN_LLM_MODEL.
        reply = await _agent.llm.generate_async(prompt, max_tokens=140, temp=0.2)
        return SecurityChatResponse(reply=reply, actions=[], status="completed", session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Feature 3: Spending DNA (ADDF: diverted → decoy)
# ---------------------------------------------------------------------------
@app.get("/api/dna/profile/{user_id}", response_model=SpendingDNAResponse, tags=["Spending DNA"])
async def get_dna_profile(user_id: str, request: Request, session_id: Optional[str] = Query(None)):
    """8-axis Spending DNA for a user."""
    if not _dna_agent:
        raise HTTPException(status_code=503, detail="DNA Agent not loaded.")
    result = _dna_agent.compute_dna(user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return SpendingDNAResponse(**result)


@app.post("/api/dna/compare", response_model=DNACompareResponse, tags=["Spending DNA"])
async def compare_dna(req: DNACompareRequest):
    """Compare session vs. DNA baseline."""
    if not _dna_agent:
        raise HTTPException(status_code=503, detail="DNA Agent not loaded.")
    result = _dna_agent.compare_session(req.user_id, session_overrides=req.session_overrides)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return DNACompareResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
