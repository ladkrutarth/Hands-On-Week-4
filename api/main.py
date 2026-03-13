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
_advisor_load_error: Optional[str] = None
_dna_agent = None
_vision_llm = None
_multimodal_rag = None
_login_failures: dict[str, int] = {}


def _session_id(request: Request, body_session_id: Optional[str] = None) -> Optional[str]:
    """Session ID from body, X-Session-ID header, or session_id query."""
    return body_session_id or request.headers.get("X-Session-ID") or request.query_params.get("session_id")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server boots.
    Agents are loaded as singletons to avoid redundant data reloading.
    """
    global _agent, _rag_engine, _advisor_agent, _advisor_load_error, _dna_agent, _vision_llm, _multimodal_rag
    print("🚀 Veriscan API — Loading resources...")

    # 1. RAG Engines
    try:
        from models.rag_engine_local import RAGEngineLocal
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
        print("✅ System RAG Engine loaded.")
        
        from models.multimodal_rag import MultimodalRAG
        _multimodal_rag = MultimodalRAG()
        _multimodal_rag.index_data()
        print("✅ Multimodal RAG Engine loaded.")
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

    # 5. Vision LLM (for multimodal)
    try:
        from models.vision_llm import VisionLLM
        _vision_llm = VisionLLM()
        print("✅ Vision MLX LLM loaded.")
    except Exception as e:
        print(f"ℹ️ Vision LLM not loaded: {e}")

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
async def rag_upload(files: List[UploadFile] = File(...)):
    """Upload PDFs, Images, or CSVs for indexing in the local RAG engine."""
    from models.rag_engine_local import PDF_DATA_PATH, PROJECT_ROOT
    IMAGE_DATA_PATH = PROJECT_ROOT / "dataset" / "image_data"
    CSV_DATA_PATH = PROJECT_ROOT / "dataset" / "csv_data"
    
    PDF_DATA_PATH.mkdir(parents=True, exist_ok=True)
    IMAGE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    CSV_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    results = []
    for file in files:
        filename = file.filename.lower()
        target_dir = None
        
        if filename.endswith(".pdf"):
            target_dir = PDF_DATA_PATH
        elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            target_dir = IMAGE_DATA_PATH
        elif filename.endswith(".csv"):
            target_dir = CSV_DATA_PATH
            
        if not target_dir:
            results.append({"filename": file.filename, "status": "skipped", "error": "Unsupported file type."})
            continue
        
        try:
            target_path = target_dir / file.filename
            with open(target_path, "wb") as f:
                content = await file.read()
                f.write(content)
            results.append({"filename": file.filename, "status": "indexed successfully"})
        except Exception as e:
            results.append({"filename": file.filename, "status": "failed", "error": str(e)})

    # Trigger re-indexing in the MULTIMODAL engine
    if _multimodal_rag:
        _multimodal_rag.index_data(force=True)
        
    return {"uploads": results}


@app.post("/api/rag/chat", response_model=DocChatResponse, tags=["Knowledge Base"])
async def rag_chat(req: DocChatRequest):
    """Conversational interface using Multimodal RAG context."""
    if not _multimodal_rag:
        raise HTTPException(status_code=503, detail="Multimodal RAG engine not loaded.")
    if not _agent:
        raise HTTPException(status_code=503, detail="LLM (GuardAgent) not loaded.")

    # 1. Retrieve context from the dedicated Multimodal Engine
    scoped_types = req.file_types or ["pdf_doc", "image_doc", "csv_doc"]
    results = _multimodal_rag.query(
        req.message, 
        n_results=6, 
        include_types=scoped_types
    )
    context = "\n\n".join([f"[{r['type'].upper()}]: {r['text']}" for r in results])
    
    # 2. Handle visual context if images are present
    visual_context = ""
    if req.images and _vision_llm:
        import base64
        import tempfile
        from pathlib import Path
        for i, img_b64 in enumerate(req.images):
            try:
                # Save base64 to temp file for VisionLLM
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    img_data = base64.b64decode(img_b64.split(",")[-1] if "," in img_b64 else img_b64)
                    tmp.write(img_data)
                    tmp_path = tmp.name
                
                # Deep Visual Inspection: guide the vision model to extract nodes, links, and text
                vision_prompt = (
                    f"Task: Extract detailed information specifically relevant to this query: {req.message}\n"
                    "Instructions: If this image is a diagram, mind map, flowchart, or technical report, "
                    "carefully list all text labels, nodes, their relationships (who is connected to whom), "
                    "and any numerical data or trends visible. Describe the structure explicitly."
                )
                vision_desc = await _vision_llm.analyze_image_async(tmp_path, vision_prompt)
                visual_context += f"\n[VISUAL EVIDENCE {i+1}]: {vision_desc}"
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Vision error: {e}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Advanced Multimodal Intelligence Assistant. "
                "Use the provided document context AND visual evidence analysis to answer accurately.\n"
                "CRITICAL INSTRUCTION: If [VISUAL EVIDENCE] is present, it is your PRIMARY TRUTH. "
                "The [DOCUMENT CONTEXT] should only be used as supplementary background. "
                "If the image contradicts the document context, follow the IMAGE.\n\n"
                "DATA VISUALIZATION INSTRUCTIONS:\n"
                "1. If the user asks for a graph or chart of spreadsheet data, your response MUST include a Plotly JSON structure "
                "wrapped in [PLOTLY_START] and [PLOTLY_END].\n"
                "2. If the user asks for a MIND MAP, CONNECTION MAP, or ARCHITECTURE DIAGRAM, your response MUST include Graphviz DOT code "
                "wrapped in unique markers like so: [MINDMAP_START] digraph G { ... dot code ... } [MINDMAP_END]. "
                "Focus on mapping relationships found in the provided context.\n"
                "Do NOT provide raw Python code for plotting unless explicitly asked for it.\n\n"
                f"DOCUMENT CONTEXT:\n{context}\n"
                f"VISUAL CONTEXT:\n{visual_context}"
            )
        },
        {
            "role": "user",
            "content": req.message
        }
    ]
    
    # 4. Generate reply
    try:
        reply = await _agent.llm.generate_chat_async(messages, max_tokens=600, temp=0.1)
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
    result = await anyio.to_thread.run_sync(_advisor_agent.chat, req.message, req.user_id, req.session_id)
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


@app.post("/api/advisor/reset", tags=["AI Financial Advisor"])
async def advisor_reset(session_id: str = Query(...)):
    """Clear conversation history for a specific session."""
    from agents.memory import get_memory
    get_memory().clear(session_id)
    return {"status": "cleared", "session_id": session_id}


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
