"""
Veriscan — FastAPI Microservices Backend
Decoupled REST API for Fraud Prediction, GuardAgent, and RAG Engine.

Run with: ./run_api.sh
  (or: PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload)
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
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
    DNAChallengeRequest,
    DNAChallengeResponse,
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

    # 5. Vision LLM (for multimodal) — skip with VERISCAN_SKIP_VISION=1 for faster advisor boot
    if os.environ.get("VERISCAN_SKIP_VISION", "").strip() == "1":
        print("ℹ️ Vision LLM skipped (VERISCAN_SKIP_VISION=1).")
    else:
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
async def rag_upload(files: List[UploadFile] = File(...), session_id: Optional[str] = Query(None)):
    """Upload PDFs, Images, CSVs, or Text for indexing in the local RAG engine."""
    from models.rag_engine_local import PROJECT_ROOT
    
    # Session-isolated storage
    SESS_ID = session_id or "global"
    BASE_UPLOAD_DIR = PROJECT_ROOT / "data" / "user_uploads" / SESS_ID
    BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"DEBUG: Receiving upload for session: {SESS_ID}")
    
    results = []
    for file in files:
        filename = file.filename.lower()
        # Supported types: PDF, Images, CSV, Text/JSON
        if not filename.endswith((".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".csv", ".txt", ".json")):
            results.append({"filename": file.filename, "status": "skipped", "error": "Unsupported file type."})
            continue
            
        try:
            target_path = BASE_UPLOAD_DIR / file.filename
            with open(target_path, "wb") as f:
                content = await file.read()
                f.write(content)
            print(f"DEBUG: Saved {file.filename} to {target_path}")
            results.append({"filename": file.filename, "status": "indexed successfully"})
        except Exception as e:
            print(f"DEBUG: Failed to save {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "failed", "error": str(e)})

    # Trigger re-indexing in the MULTIMODAL engine using byte pipeline (OCR-aware)
    if _multimodal_rag:
        for file in files:
            filename = (file.filename or "").lower()
            if not filename.endswith((".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".csv", ".txt", ".json", ".webp")):
                continue
            try:
                target_path = BASE_UPLOAD_DIR / file.filename
                raw = target_path.read_bytes()
                _multimodal_rag.index_file_bytes(file.filename, raw, session_id=SESS_ID)
            except Exception as e:
                print(f"DEBUG: Re-index failed for {file.filename}: {e}")
        
    return {"uploads": results}


@app.post("/api/rag/chat", response_model=DocChatResponse, tags=["Knowledge Base"])
async def rag_chat(req: DocChatRequest):
    """Conversational interface using Multimodal RAG context."""
    if not _multimodal_rag:
        raise HTTPException(status_code=503, detail="Multimodal RAG engine not loaded.")

    from models.rag_relevance import format_grounded_context, grounded_system_prompt, is_visual_query

    scoped_types = req.file_types or [
        "pdf_doc", "image_doc", "csv_doc", "text_doc", "pdf_summary", "csv_summary"
    ]
    print(f"DEBUG: Session ID: {req.session_id}")
    print(f"DEBUG: Querying RAG for: '{req.message}'")
    results = _multimodal_rag.query(
        req.message,
        n_results=6,
        include_types=scoped_types,
        session_id=req.session_id,
    )

    # Visual questions: always pull image docs + live OCR even if dense retrieval misses
    visual_q = is_visual_query(req.message)
    if visual_q and req.session_id and not results:
        results = _multimodal_rag.force_image_hits(req.session_id)

    # Text PDFs/CSVs: if dense retrieval missed, still ground on indexed page chunks
    if req.session_id and not results:
        results = _multimodal_rag.force_document_hits(req.session_id)

    print(f"DEBUG: Retrieved {len(results)} relevant results.")
    for i, r in enumerate(results):
        fname = r.get("filename") or r.get("metadata", {}).get("filename", "Unknown Source")
        conf = r.get("confidence", 0.0)
        page = r.get("page") or r.get("metadata", {}).get("page")
        print(f"DEBUG: Result {i+1} conf={conf:.3f} page={page} Type: {r.get('type')} | File: {fname}")

    context = format_grounded_context(results)

    # Vision / live OCR
    visual_context = ""
    import base64
    import tempfile

    vision_targets: list[str] = []
    if req.images:
        for img_b64 in req.images[:3]:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    img_data = base64.b64decode(img_b64.split(",")[-1] if "," in img_b64 else img_b64)
                    tmp.write(img_data)
                    vision_targets.append(tmp.name)
            except Exception as e:
                print(f"Vision decode error: {e}")

    want_vision = bool(
        _multimodal_rag.needs_vision(req.message, results, req.session_id)
        or visual_q
        or req.images
    )
    if want_vision and req.session_id and not vision_targets:
        for p in _multimodal_rag.resolve_session_image_paths(req.session_id)[:2]:
            vision_targets.append(str(p))

    if _vision_llm and vision_targets:
        vision_prompt = (
            f"Task: Extract detailed information relevant to this query: {req.message}\n"
            "If this is a bank statement, receipt, UI screenshot, or photo, describe what is visible "
            "and list every readable date, amount, merchant, label, and notable UI element. "
            "Do not invent text that is not visible."
        )
        for tmp_path in vision_targets:
            try:
                vision_desc = _vision_llm.analyze_image(tmp_path, vision_prompt, max_tokens=500)
                visual_context += f"\n[VISUAL EVIDENCE]: {vision_desc}"
            except Exception as e:
                print(f"Vision error: {e}")
            finally:
                if tmp_path.startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    # If Vision LLM missing/failed, still try live OCR on session images
    if want_vision and req.session_id and not visual_context.strip():
        live = _multimodal_rag.live_analyze_session_images(req.session_id, req.message)
        if live.get("has_useful_ocr") and live.get("text"):
            visual_context = live["text"]
        elif live.get("text"):
            # Keep diagnostic OCR notes as weak context for the LLM / user message
            visual_context = live["text"]

    # Refuse only when still no usable context
    from models.rag_relevance import is_unreadable_visual_evidence

    # Prefer real document text over weak OCR stubs when both exist
    readable_hits = [
        r
        for r in (results or [])
        if (r.get("metadata") or {}).get("ocr_weak") != "true"
        and (r.get("metadata") or {}).get("empty") != "true"
        and not is_unreadable_visual_evidence(r.get("text") or "")
    ]
    if readable_hits and len(readable_hits) < len(results or []):
        results = readable_hits
        context = format_grounded_context(results)

    weak_stubs = bool(results) and all(
        (r.get("metadata") or {}).get("ocr_weak") == "true"
        or is_unreadable_visual_evidence(r.get("text") or "")
        for r in results
    )
    vision_useful = bool(visual_context.strip()) and not is_unreadable_visual_evidence(visual_context)
    useful = (bool(context.strip()) and not weak_stubs) or vision_useful
    if not useful:
        inventory = _multimodal_rag.get_file_inventory(req.session_id) if req.session_id else []
        names = ", ".join(f.get("filename", "?") for f in inventory) or "(none indexed)"
        has_images = bool(
            req.images
            or (
                req.session_id
                and _multimodal_rag.resolve_session_image_paths(req.session_id)
            )
        )
        kind = "image" if has_images else "document"
        reply = (
            f"I can see that a {kind} was uploaded, but I cannot read its contents yet "
            "(no usable OCR text and Vision LLM did not return a description).\n\n"
            f"**Indexed files:** {names}\n\n"
            "**Fix:** re-index after OCR is available (Tesseract via `brew install tesseract`, "
            "or the bundled RapidOCR fallback), restart the API **without** "
            "`VERISCAN_SKIP_VISION=1`, click **Index All Documents** again, and re-ask. "
            "I will not invent tables or amounts from a blank OCR result."
        )
        return DocChatResponse(
            reply=reply,
            sources=[
                {
                    "text": r.get("text"),
                    "metadata": r.get("metadata", {}),
                    "confidence": r.get("confidence"),
                    "filename": r.get("filename"),
                    "page": r.get("page"),
                }
                for r in (results or [])
            ],
            session_id=req.session_id,
        )
    system_content = grounded_system_prompt(context, visual_context)
    system_content += (
        "\n\nDATA VISUALIZATION:\n"
        "If the user asks for a graph/chart, include Plotly JSON between [PLOTLY_START] and [PLOTLY_END] "
        "with ONLY the raw JSON object (no markdown fences)."
    )

    sources = [
        {
            "text": r["text"],
            "metadata": r.get("metadata", {}),
            "confidence": r.get("confidence"),
            "filename": r.get("filename") or (r.get("metadata") or {}).get("filename"),
            "page": r.get("page") or (r.get("metadata") or {}).get("page"),
        }
        for r in results
    ]

    if not _agent or not getattr(_agent, "llm", None):
        from models.rag_relevance import snippet_fallback_answer
        reply = snippet_fallback_answer(req.message, results)
        if visual_context and "OCR extracted little text" not in visual_context:
            reply = f"{reply}\n\n**Visual notes:**\n{visual_context[:1200]}"
        elif visual_context:
            reply = (
                "I can see an uploaded image, but could not read enough text from it yet.\n\n"
                f"{visual_context[:800]}"
            )
        return DocChatResponse(reply=reply, sources=sources, session_id=req.session_id)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": req.message},
    ]
    try:
        if hasattr(_agent.llm, "generate_chat"):
            reply = _agent.llm.generate_chat(messages, max_tokens=700, temp=0.1)
        else:
            prompt = system_content + "\n\nUser: " + req.message + "\nAssistant:"
            reply = _agent.llm.generate(prompt, max_tokens=700, temp=0.1)
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
    """Agentic (ReAct / supervisor) or keyword financial advice.

    Note: MLX inference must stay on the main thread (no anyio worker thread),
    otherwise Metal can abort with Completed-handler assertions.
    """
    if not _advisor_agent:
        detail = f"Financial Advisor not loaded: {_advisor_load_error}" if _advisor_load_error else "Financial Advisor not loaded."
        raise HTTPException(status_code=503, detail=detail)

    result = _advisor_agent.chat(
        req.message,
        req.user_id,
        session_id=req.session_id,
        approved=req.approved,
    )
    return AdvisorChatResponse(
        user_id=req.user_id,
        message=req.message,
        reply=result.get("reply", ""),
        tool_results=result.get("tool_results", []),
        trace=result.get("trace"),
        step_count=result.get("step_count"),
        tools_used=result.get("tools_used"),
        pending_approval=result.get("pending_approval"),
        status=result.get("status"),
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
    """Security analyst with tool-calling ReAct loop (RAG, risk profile, HITL challenge)."""
    if not _agent:
        raise HTTPException(status_code=503, detail="GuardAgent not loaded.")
    try:
        # MLX must run on the main thread (see advisor_chat note).
        result = _agent.analyze(
            req.message,
            session_id=req.session_id,
            approved=req.approved,
        )
        raw_actions = result.get("actions") or []
        actions = []
        for a in raw_actions:
            if isinstance(a, dict):
                actions.append(
                    AgentActionStep(
                        step=int(a.get("step", 0)),
                        tool=str(a.get("tool", "")),
                        args=a.get("args") or {},
                        result=a.get("result"),
                    )
                )
        return SecurityChatResponse(
            reply=result.get("reply") or result.get("answer") or "",
            actions=actions,
            status=result.get("status", "completed"),
            session_id=req.session_id,
            trace=result.get("trace"),
            tools_used=result.get("tools_used"),
            step_count=result.get("step_count"),
            pending_approval=result.get("pending_approval"),
        )
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
    """Compare session vs. DNA baseline. High deviation can require Challenge Auth approval."""
    if not _dna_agent:
        raise HTTPException(status_code=503, detail="DNA Agent not loaded.")
    result = _dna_agent.compare_session(req.user_id, session_overrides=req.session_overrides)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    trust = float(result.get("session_trust_score", 1.0))
    needs_challenge = trust < 0.55
    challenge = None
    if needs_challenge and req.approved:
        from models.guard_agent_local import tool_challenge_auth
        challenge = tool_challenge_auth(req.user_id, reason="high_dna_deviation")
    result["requires_approval"] = bool(needs_challenge and not req.approved)
    result["challenge"] = challenge
    return DNACompareResponse(**result)


@app.post("/api/dna/challenge", response_model=DNAChallengeResponse, tags=["Spending DNA"])
async def dna_challenge_auth(req: DNAChallengeRequest):
    """HITL-gated Challenge Auth — will not execute unless approved=true."""
    if not req.approved:
        return DNAChallengeResponse(
            user_id=req.user_id,
            status="pending_approval",
            challenge=None,
            message="Challenge Auth requires human approval. Re-submit with approved=true.",
            pending_approval=True,
        )
    from models.guard_agent_local import tool_challenge_auth
    challenge = tool_challenge_auth(req.user_id, reason=req.reason)
    return DNAChallengeResponse(
        user_id=req.user_id,
        status="challenge_issued",
        challenge=challenge,
        message=challenge.get("message", "Challenge Auth issued."),
        pending_approval=False,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
