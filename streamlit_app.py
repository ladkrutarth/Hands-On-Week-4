"""
GraphGuard — Streamlit Dashboard
AI-Powered Fraud Detection & Dynamic Authentication System
Uses Google Gemini LLM + RAG for intelligent features.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "features_output.csv"
FRAUD_SCORES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "fraud_scores_output.csv"
AUTH_PROFILES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "auth_profiles_output.csv"
PIPELINE_LOGS_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "pipeline_logs.csv"
# Kaggle dataset (fallback to legacy synthetic dataset)
TRANSACTIONS_PATH = (
    PROJECT_ROOT / "dataset" / "csv_data" / "transactions_kaggle.csv"
    if (PROJECT_ROOT / "dataset" / "csv_data" / "transactions_kaggle.csv").exists()
    else PROJECT_ROOT / "dataset" / "csv_data" / "transactions_3000.csv"
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphGuard — AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Friendly Name Mapping
# ---------------------------------------------------------------------------
USER_NAMES = {
    "USER_001": "Alice Smith", "USER_002": "Bob Jones", "USER_003": "Charlie Brown",
    "USER_004": "Diana Prince", "USER_005": "Evan Davis", "USER_006": "Fiona Gallagher",
    "USER_007": "George Miller", "USER_008": "Hannah Abbott", "USER_009": "Ian Wright",
    "USER_010": "Julia Roberts", "USER_011": "Kevin Hart", "USER_012": "Liam Neeson",
    "USER_013": "Mila Kunis", "USER_014": "Noah Centineo", "USER_015": "Olivia Wilde",
    "USER_016": "Paul Rudd", "USER_017": "Quinn Fabray", "USER_018": "Ryan Reynolds",
    "USER_019": "Scarlett Johansson", "USER_020": "Tom Holland"
}
# Fallback for USER_021+ will just use the ID

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #a0a0b0;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .risk-low { background: rgba(76, 175, 80, 0.2); color: #4caf50; border: 1px solid rgba(76,175,80,0.3); }
    .risk-medium { background: rgba(255, 193, 7, 0.2); color: #ffc107; border: 1px solid rgba(255,193,7,0.3); }
    .risk-high { background: rgba(255, 152, 0, 0.2); color: #ff9800; border: 1px solid rgba(255,152,0,0.3); }
    .risk-critical { background: rgba(244, 67, 54, 0.2); color: #f44336; border: 1px solid rgba(244,67,54,0.3); }

    .question-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        color: #1a202c;
    }
    .question-card .q-num {
        color: #667eea;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .question-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 0.8rem;
        line-height: 1.5;
    }

    .rag-result {
        background: rgba(102, 126, 234, 0.05);
        border-left: 3px solid #667eea;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    .gemini-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: linear-gradient(135deg, #4285f4, #34a853);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .sidebar-info {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_fraud_scores():
    if FRAUD_SCORES_PATH.exists():
        return pd.read_csv(FRAUD_SCORES_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_features():
    if FEATURES_PATH.exists():
        return pd.read_csv(FEATURES_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_auth_profiles():
    if AUTH_PROFILES_PATH.exists():
        return pd.read_csv(AUTH_PROFILES_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_pipeline_logs():
    if PIPELINE_LOGS_PATH.exists():
        return pd.read_csv(PIPELINE_LOGS_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_transactions():
    """Load raw transaction data (Kaggle or legacy synthetic)."""
    if TRANSACTIONS_PATH.exists():
        return pd.read_csv(TRANSACTIONS_PATH, parse_dates=["TRANSACTION_DATE"])
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Initialize LLM / RAG (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_engine(api_key: str):
    """v2: Updated signature for user-specific filtering"""
    from models.rag_engine import RAGEngine
    engine = RAGEngine(api_key=api_key)
    engine.index_data()
    return engine


@st.cache_resource
def get_question_generator(api_key: str):
    from models.gemini_question_gen import GeminiQuestionGenerator
    return GeminiQuestionGenerator(api_key=api_key)


def get_api_key():
    """Get API key from sidebar, secrets, or environment."""
    api_key = ""
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    elif os.environ.get("GOOGLE_API_KEY"):
        api_key = os.environ["GOOGLE_API_KEY"]
    return api_key


def risk_badge(level: str) -> str:
    """Return styled HTML badge for risk level."""
    cls = f"risk-{level.lower()}"
    return f'<span class="risk-badge {cls}">{level}</span>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ GraphGuard")
        st.markdown("AI-Powered Fraud Detection")
        st.divider()

        api_key = st.text_input(
            "🔑 Google API Key",
            value=get_api_key(),
            type="password",
            help="Enter your Google Gemini API key for AI features. Leave empty to use demo mode.",
        )

        if api_key:
            st.markdown('<span class="gemini-badge">✨ Gemini Active</span>', unsafe_allow_html=True)
        else:
            st.info("ℹ️ Demo mode — AI features use static fallbacks.")

        st.divider()

        # System status
        scores_df = load_fraud_scores()
        features_df = load_features()
        auth_df = load_auth_profiles()

        st.markdown("### 📊 System Status")
        st.markdown(f"""
<div class="sidebar-info">
📄 Transactions: <b>{len(features_df)}</b><br>
🎯 Scored: <b>{len(scores_df)}</b><br>
👤 Users: <b>{auth_df['user_id'].nunique() if not auth_df.empty else 0}</b><br>
</div>
""", unsafe_allow_html=True)

        return api_key


# ---------------------------------------------------------------------------
# Tab 1: Authentication (Sequential Multiple-Choice)
# ---------------------------------------------------------------------------
def render_auth_tab(api_key: str):
    st.markdown("### 🔑 Dynamic Authentication")
    st.markdown("AI-generated multiple-choice security questions adapted to user risk profiles.")

    scores_df = load_fraud_scores()
    auth_df = load_auth_profiles()

    if scores_df.empty:
        st.warning("No fraud scores found. Run the pipeline first.")
        return

    # Initialize state machine
    if "auth_state" not in st.session_state:
        st.session_state["auth_state"] = "start"  # states: start, qa, mfa_fallback, passed
        st.session_state["auth_questions"] = []
        st.session_state["auth_q_idx"] = 0
        st.session_state["auth_wrong_count"] = 0
        st.session_state["auth_used_questions"] = set()  # track used questions for uniqueness
        st.session_state["auth_q_start_time"] = None  # timer start per question

    TIMER_SECONDS = 30

    col1, col2 = st.columns([1, 2])

    with col1:
        users = sorted(scores_df["USER_ID"].unique())
        
        # User selection resets state if changed
        selected_user = st.selectbox(
            "Select User", 
            users, 
            key="auth_user",
            format_func=lambda x: f"{USER_NAMES.get(x, x)} ({x})",
            on_change=lambda: st.session_state.update({"auth_state": "start", "auth_used_questions": set()})
        )

        # Show user profile
        if not auth_df.empty:
            profile_row = auth_df[auth_df["user_id"] == selected_user]
            if not profile_row.empty:
                p = profile_row.iloc[0]
                level = p["recommended_security_level"]
                st.markdown(f"**Security Level:** {risk_badge(level)}", unsafe_allow_html=True)
                st.metric("Avg Risk Score", f"{p['avg_risk']:.4f}")
                
        # Generate Button
        if st.session_state["auth_state"] == "start":
            generate = st.button("🧠 Generate Questions", type="primary", use_container_width=True)
            if generate:
                with st.spinner("Generating unique questions with Gemini + RAG..."):
                    from models.auth_decision import compute_user_risk_profile
                    profile = compute_user_risk_profile(selected_user, scores_df)

                    rag = get_rag_engine(api_key)
                    gen = get_question_generator(api_key)
                    rag_context = rag.get_context_for_user(selected_user)

                    num_questions = min(int(profile["num_questions"]), 3)
                    
                    questions = gen.generate_questions(
                        user_id=selected_user,
                        security_level=profile["recommended_security_level"],
                        num_questions=num_questions,
                        rag_context=rag_context,
                        user_profile=profile,
                    )

                    source = "gemini" if gen.model else "static"
                    used = {q["question"] for q in questions}
                    st.session_state.update({
                        "auth_state": "qa",
                        "auth_questions": questions,
                        "auth_source": source,
                        "auth_user_selected": selected_user,
                        "auth_q_idx": 0,
                        "auth_wrong_count": 0,
                        "auth_num_questions": len(questions),
                        "auth_used_questions": used,
                        "auth_q_start_time": time.time(),
                        "auth_security_level": profile["recommended_security_level"],
                        "auth_rag_context": rag_context,
                        "auth_profile": profile,
                    })
                st.rerun()

    with col2:
        state = st.session_state["auth_state"]

        if state == "start":
            st.info("Select a user and click 'Generate Questions' to begin.")
            
        elif state == "qa":
            questions = st.session_state["auth_questions"]
            idx = st.session_state["auth_q_idx"]
            total = st.session_state["auth_num_questions"]
            current_q = questions[idx]
            
            # --- Timer logic ---
            start_time = st.session_state.get("auth_q_start_time", time.time())
            elapsed = time.time() - start_time
            remaining = max(0, TIMER_SECONDS - elapsed)
            timer_pct = remaining / TIMER_SECONDS

            # Timer expired — treat as missed
            if remaining <= 0:
                st.session_state["auth_wrong_count"] += 1
                if st.session_state["auth_wrong_count"] >= 3:
                    st.session_state["auth_state"] = "mfa_fallback"
                    st.rerun()
                else:
                    # Generate a replacement question
                    _replace_current_question(api_key, idx)
                    st.rerun()

            # Progress bar
            progress = (idx) / total
            st.progress(progress, text=f"Question {idx+1} of {total}")

            # --- Live JavaScript countdown timer ---
            import streamlit.components.v1 as components
            components.html(f"""
<div id="timer-container" style="margin: 0.5rem 0 1rem 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
        <span style="font-size: 0.85rem; color: #a0a0b0;">⏱️ Time remaining</span>
        <span id="timer-text" style="font-size: 1.4rem; font-weight: 700; color: #4caf50;">{int(remaining)}s</span>
    </div>
    <div style="background: rgba(255,255,255,0.1); border-radius: 6px; height: 8px;">
        <div id="timer-bar" style="background: #4caf50; height: 100%; width: {timer_pct*100:.1f}%; border-radius: 6px; transition: width 0.3s ease, background 0.5s ease;"></div>
    </div>
</div>
<script>
(function() {{
    var remaining = {remaining:.1f};
    var total = {TIMER_SECONDS};
    var textEl = document.getElementById('timer-text');
    var barEl = document.getElementById('timer-bar');
    if (!textEl || !barEl) return;

    function getColor(pct) {{
        if (pct > 0.5) return '#4caf50';
        if (pct > 0.2) return '#ff9800';
        return '#f44336';
    }}

    function tick() {{
        remaining -= 1;
        if (remaining < 0) remaining = 0;
        var pct = remaining / total;
        var color = getColor(pct);
        textEl.textContent = Math.ceil(remaining) + 's';
        textEl.style.color = color;
        barEl.style.width = (pct * 100) + '%';
        barEl.style.background = color;

        if (remaining <= 0) {{
            // Time expired — reload to trigger server-side expiry
            window.parent.location.reload();
        }} else {{
            setTimeout(tick, 1000);
        }}
    }}
    setTimeout(tick, 1000);
}})();
</script>
""", height=70)

            # Question card
            st.markdown(
                f'<div class="question-card">'
                f'<span class="q-num">Question {idx+1}</span> '
                f'<span style="font-size:0.8rem;color:#a0aec0;margin-left:8px;font-style:italic;">({current_q.get("difficulty", "medium")})</span>'
                f'<div class="question-text">{current_q["question"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            
            # Multiple Choice Radio
            options = current_q.get("options", [])
            selected_option = st.radio("Select your answer:", options, key=f"q_{idx}_{int(start_time)}")
            
            if st.button("Submit Answer", type="primary"):
                correct_ans = current_q.get("correct_answer", "")
                if selected_option == correct_ans:
                    st.success("✅ Correct!")
                    st.session_state["auth_q_idx"] += 1
                    
                    # Check if finished
                    if st.session_state["auth_q_idx"] >= total:
                        st.session_state["auth_state"] = "passed"
                    else:
                        # Reset timer for next question
                        st.session_state["auth_q_start_time"] = time.time()
                    st.rerun()
                else:
                    st.error("❌ Incorrect answer.")
                    st.session_state["auth_wrong_count"] += 1
                    
                    if st.session_state["auth_wrong_count"] >= 3:
                        st.session_state["auth_state"] = "mfa_fallback"
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Wrong! Generating a new question... ({3 - st.session_state['auth_wrong_count']} attempts left)")
                        _replace_current_question(api_key, idx)
                        st.rerun()

        elif state == "passed":
            st.progress(1.0, text="Authentication Complete")
            st.success("🎉 Authentication PASSED! Identity verified successfully.")
            if st.button("Restart Authentication"):
                st.session_state["auth_state"] = "start"
                st.session_state["auth_used_questions"] = set()
                st.rerun()

        elif state == "mfa_fallback":
            st.error("🚫 **Authentication Failed**")
            st.warning("You have answered 3 questions incorrectly or timed out. Your account context could not be verified.")
            
            st.markdown("""
            <div style="border: 2px solid #f44336; border-radius: 8px; padding: 20px; text-align: center; background: rgba(244, 67, 54, 0.1); margin-top: 1rem;">
                <h3 style="color: #f44336; margin-top: 0">MFA Login Required</h3>
                <p>To protect your account, we require Multi-Factor Authentication to proceed.</p>
                <button style="background: #f44336; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer; margin-top: 10px;">Send SMS Code</button>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Back to Start"):
                st.session_state["auth_state"] = "start"
                st.session_state["auth_used_questions"] = set()
                st.rerun()


def _replace_current_question(api_key: str, idx: int):
    """Generate a new unique replacement question via LLM when user gets it wrong or times out."""
    gen = get_question_generator(api_key)
    rag_context = st.session_state.get("auth_rag_context", "")
    user_id = st.session_state.get("auth_user_selected", "")
    sec_level = st.session_state.get("auth_security_level", "MEDIUM")
    used = st.session_state.get("auth_used_questions", set())

    # Try to generate a unique question via LLM
    new_q = None
    if gen.model and rag_context:
        try:
            avoid_str = "\n".join(f"- {q}" for q in used) if used else "None"
            prompt = f"""You are a bank security system. Generate exactly 1 NEW multiple-choice question to verify customer {user_id}.

Recent transaction data:
{rag_context}

IMPORTANT: You MUST generate a COMPLETELY DIFFERENT question. Do NOT repeat any of these previously used questions:
{avoid_str}

RULES:
1. Ask about stores, locations, categories, amounts, or shopping days from the last 5 days.
2. NEVER ask about risk scores or internal metrics.
3. Make it feel like a natural bank verification call.
4. Provide exactly 4 options. ONE correct, three plausible but wrong.

Return ONLY a JSON array with 1 object:
[{{"question": "...", "options": ["A", "B", "C", "D"], "correct_answer": "B", "difficulty": "medium"}}]
"""
            import json
            resp = gen._client.models.generate_content(model=gen.model, contents=prompt)
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            parsed = json.loads(text)
            if parsed and isinstance(parsed, list) and len(parsed) > 0:
                import random
                q = parsed[0]
                opts = q.get("options", [])
                random.shuffle(opts)
                new_q = {
                    "question": q.get("question", "Verify your identity."),
                    "options": opts,
                    "correct_answer": q.get("correct_answer", opts[0] if opts else ""),
                    "difficulty": q.get("difficulty", "medium"),
                }
        except Exception:
            pass

    # Fallback: pick a random static question not yet used
    if not new_q:
        import random
        from models.gemini_question_gen import STATIC_QUESTIONS
        all_static = []
        for qs in STATIC_QUESTIONS.values():
            all_static.extend(qs)
        unused = [q for q in all_static if q["question"] not in used]
        if unused:
            new_q = random.choice(unused)
        elif all_static:
            new_q = random.choice(all_static)

    if new_q:
        st.session_state["auth_questions"][idx] = new_q
        st.session_state["auth_used_questions"].add(new_q["question"])

    st.session_state["auth_q_start_time"] = time.time()


# ---------------------------------------------------------------------------
# Tab 2: Fraud Detection
# ---------------------------------------------------------------------------
def render_fraud_tab(api_key: str):
    st.markdown("### 🚨 AI Fraud Analysis")
    st.markdown("RAG-enhanced fraud explanations powered by Gemini.")

    scores_df = load_fraud_scores()
    features_df = load_features()

    if scores_df.empty:
        st.warning("No fraud scores found. Run the pipeline first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        # Risk level filter
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=["HIGH", "CRITICAL"],
        )

        filtered = scores_df[scores_df["RISK_LEVEL"].isin(risk_filter)] if risk_filter else scores_df
        st.write(f"Showing {len(filtered)} transactions")

        if not filtered.empty:
            txn_id = st.selectbox(
                "Select Transaction",
                filtered["TRANSACTION_ID"].tolist(),
                key="fraud_txn",
            )
        else:
            st.info("No transactions match the filter.")
            return

    with col2:
        txn = scores_df[scores_df["TRANSACTION_ID"] == txn_id].iloc[0]
        level = txn["RISK_LEVEL"]

        # Transaction details
        st.markdown(f"#### Transaction: `{txn_id}`")
        st.markdown(f"**Risk Level:** {risk_badge(level)}", unsafe_allow_html=True)

        # Ground truth badge (Kaggle dataset with is_fraud labels)
        if "IS_FRAUD_ACTUAL" in txn.index:
            is_fraud = int(txn["IS_FRAUD_ACTUAL"])
            fraud_badge_html = (
                '<span style="background:rgba(244,67,54,0.15);color:#f44336;padding:0.2rem 0.7rem;'
                'border-radius:12px;font-size:0.8rem;font-weight:700;border:1px solid rgba(244,67,54,0.3);">'
                '🔴 Known Fraud</span>' if is_fraud else
                '<span style="background:rgba(76,175,80,0.15);color:#4caf50;padding:0.2rem 0.7rem;'
                'border-radius:12px;font-size:0.8rem;font-weight:700;border:1px solid rgba(76,175,80,0.3);">'
                '🟢 Verified Legit</span>'
            )
            st.markdown(f"**Ground Truth:** {fraud_badge_html}", unsafe_allow_html=True)

        if "MERCHANT_NAME" in txn.index and pd.notna(txn["MERCHANT_NAME"]):
            st.markdown(f"**Merchant:** `{txn['MERCHANT_NAME']}`")
        if "CARDHOLDER_NAME" in txn.index and pd.notna(txn["CARDHOLDER_NAME"]):
            st.markdown(f"**Cardholder:** {txn['CARDHOLDER_NAME']}")


        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Combined Score", f"{txn['COMBINED_RISK_SCORE']:.4f}")
        mcol2.metric("Z-Score Flag", f"{txn['ZSCORE_FLAG']:.4f}")
        mcol3.metric("Velocity Flag", f"{txn['VELOCITY_FLAG']:.4f}")

        mcol4, mcol5, mcol6 = st.columns(3)
        mcol4.metric("Category Risk", f"{txn['CATEGORY_RISK_SCORE']:.4f}")
        mcol5.metric("Geographic Risk", f"{txn['GEOGRAPHIC_RISK_SCORE']:.4f}")
        mcol6.metric("Isolation Forest", f"{txn['ISOLATION_FOREST_SCORE']:.4f}")

        # AI Analysis
        if st.button("🧠 Analyze with Gemini + RAG", type="primary", use_container_width=True, key="fraud_analyze"):
            with st.spinner("Generating AI fraud analysis..."):
                gen = get_question_generator(api_key)
                rag = get_rag_engine(api_key)

                # Merge feature data for richer context
                txn_data = txn.to_dict()
                if not features_df.empty:
                    feat_row = features_df[features_df["TRANSACTION_ID"] == txn_id]
                    if not feat_row.empty:
                        txn_data.update(feat_row.iloc[0].to_dict())

                rag_context = rag.get_context_for_query(
                    f"transaction {txn_id} user {txn['USER_ID']} risk {level}"
                )

                analysis = gen.analyze_fraud_with_context(txn_data, rag_context)
                st.markdown(analysis)

        # Recommendation
        st.markdown(f"**📋 Recommendation:** {txn['RECOMMENDATION']}")


# ---------------------------------------------------------------------------
# Tab 3: Dashboard
# ---------------------------------------------------------------------------
def render_dashboard_tab():
    st.markdown("### 📊 System Dashboard")

    scores_df = load_fraud_scores()
    features_df = load_features()
    auth_df = load_auth_profiles()

    if scores_df.empty:
        st.warning("No data available. Run the pipeline first.")
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
<div class="metric-card">
    <h3>Total Transactions</h3>
    <div class="value" style="color:#667eea">{len(scores_df)}</div>
</div>
""", unsafe_allow_html=True)
    with col2:
        high_risk = len(scores_df[scores_df["RISK_LEVEL"].isin(["HIGH", "CRITICAL"])])
        st.markdown(f"""
<div class="metric-card">
    <h3>High Risk Alerts</h3>
    <div class="value" style="color:#ff9800">{high_risk}</div>
</div>
""", unsafe_allow_html=True)
    with col3:
        avg_score = scores_df["COMBINED_RISK_SCORE"].mean()
        st.markdown(f"""
<div class="metric-card">
    <h3>Avg Risk Score</h3>
    <div class="value" style="color:#4caf50">{avg_score:.3f}</div>
</div>
""", unsafe_allow_html=True)
    with col4:
        n_users = scores_df["USER_ID"].nunique()
        st.markdown(f"""
<div class="metric-card">
    <h3>Active Users</h3>
    <div class="value" style="color:#e040fb">{n_users}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Risk Level Distribution")
        risk_counts = scores_df["RISK_LEVEL"].value_counts()
        st.bar_chart(risk_counts, color="#667eea")

    with chart_col2:
        st.markdown("#### Risk Score by User")
        user_risk = scores_df.groupby("USER_ID")["COMBINED_RISK_SCORE"].mean().sort_values(ascending=False)
        st.bar_chart(user_risk, color="#764ba2")

    # Auth profiles table
    if not auth_df.empty:
        st.markdown("#### 👤 User Security Profiles")
        styled = auth_df.copy()
        styled["user_id"] = styled["user_id"].apply(lambda x: f"{USER_NAMES.get(x, x)} ({x})")
        styled.columns = ["User", "Avg Risk", "Max Risk", "High-Risk Count", "Security Level", "Questions"]
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Pipeline logs
    logs_df = load_pipeline_logs()
    if not logs_df.empty:
        st.markdown("#### 🔄 Recent Pipeline Runs")
        st.dataframe(logs_df.tail(10), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4: RAG Explorer — Production-Ready
# ---------------------------------------------------------------------------
def render_rag_tab(api_key: str):
    st.markdown("### 🔍 RAG Explorer")
    st.markdown("Ask questions about your fraud detection data using natural language.")

    rag = get_rag_engine(api_key)
    gen = get_question_generator(api_key)

    if api_key:
        st.markdown('<span class="gemini-badge">✨ Gemini + RAG Active</span>', unsafe_allow_html=True)
    else:
        st.info("ℹ️ Without a Gemini API key, raw retrieved context will be shown instead of AI-generated answers.")

    # Initialize query history
    if "rag_query_history" not in st.session_state:
        st.session_state["rag_query_history"] = []

    # Sub-tabs for Q&A and Evaluation
    rag_tab1, rag_tab2, rag_tab3 = st.tabs(["💬 Ask Questions", "📊 RAG Evaluation", "📜 Query History"])

    # Sidebar-like configuration for RAG depth (placed inside tab for focus)
    with st.sidebar:
        if st.session_state.get("active_tab") == "🔍 RAG Explorer":
            st.markdown("### ⚙️ RAG Settings")
            st.info("These settings control the depth and focus of the RAG retrieval.")
            
            # User Filter
            try:
                feat_df = pd.read_csv("dataset/csv_data/features.csv")
                user_list = ["All Users"] + sorted(feat_df["USER_ID"].unique().tolist())
            except Exception:
                user_list = ["All Users"]
                
            selected_user = st.selectbox(
                "Filter by User ID",
                options=user_list,
                index=0,
                help="Focus retrieval on a specific user's transactions and profile."
            )
            rag_user_id = None if selected_user == "All Users" else selected_user
            
            # Retrieval Depth
            rag_top_k = st.slider(
                "Retrieval Depth (Top-K)",
                min_value=3,
                max_value=20,
                value=8,
                help="Number of documents to retrieve. Higher values provide more context but may be slower."
            )
        else:
            rag_user_id = None
            rag_top_k = 5

    with rag_tab1:
        # Suggested questions
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "Which users have the highest fraud risk?",
            "What categories are associated with the most fraud?",
            "What is the overall risk distribution across all transactions?",
            "Compare risk levels across different locations",
            "What is the total transaction volume and average amount?",
        ]
        scol1, scol2, scol3 = st.columns(3)
        cols = [scol1, scol2, scol3]
        for i, s in enumerate(suggestions):
            if cols[i % 3].button(f"💬 {s[:40]}...", key=f"suggest_{i}", use_container_width=True):
                st.session_state["rag_question"] = s

        question = st.text_area(
            "Your Question",
            value=st.session_state.get("rag_question", ""),
            height=80,
            placeholder="Ask anything about the fraud detection data...",
        )

        if st.button("🔍 Search & Answer", type="primary", use_container_width=True) and question:
            with st.spinner("Rewriting query → Retrieving → Re-ranking → Generating answer..."):
                t0 = time.time()

                # RAG retrieval with detailed results
                detailed = rag.get_detailed_results(
                    question, 
                    n_results=rag_top_k, 
                    user_id=rag_user_id
                )
                retrieval_time = time.time() - t0

                # LLM answer
                answer = gen.answer_question_with_rag(question, detailed["context"])
                total_time = time.time() - t0

                # Save to query history
                st.session_state["rag_query_history"].append({
                    "question": question,
                    "confidence": detailed["avg_confidence"],
                    "sources": detailed["num_sources"],
                    "time": f"{total_time:.2f}s",
                })

                # --- Confidence Meter ---
                conf = detailed["avg_confidence"]
                conf_color = "#4caf50" if conf >= 0.7 else "#ff9800" if conf >= 0.4 else "#f44336"
                conf_label = "HIGH" if conf >= 0.7 else "MEDIUM" if conf >= 0.4 else "LOW"

                st.markdown(f"""
<div style="background: linear-gradient(145deg, #1e1e2e, #2a2a3e); border-radius: 12px; padding: 1rem 1.5rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.08);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #a0a0b0;">Retrieval Confidence</span>
        <span style="color: {conf_color}; font-weight: 700; font-size: 1.1rem;">{conf_label} ({conf:.0%})</span>
    </div>
    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 8px; overflow: hidden;">
        <div style="background: {conf_color}; height: 100%; width: {conf*100:.0f}%; border-radius: 8px; transition: width 0.5s;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem; color: #666;">
        <span>{detailed['num_sources']} sources used</span>
        <span>Types: {', '.join(detailed['source_types'])}</span>
        <span>{retrieval_time:.2f}s retrieval / {total_time:.2f}s total</span>
    </div>
</div>
""", unsafe_allow_html=True)

                # --- Structured Answer ---
                st.markdown("---")
                st.markdown("#### 💬 Answer")
                st.markdown(answer)

                # --- Source Attribution Cards ---
                with st.expander(f"📚 Source Attribution ({detailed['num_sources']} documents retrieved)"):
                    for i, r in enumerate(detailed["results"]):
                        source = r["metadata"].get("source", "unknown")
                        doc_type = r["metadata"].get("type", "unknown")
                        r_conf = r.get("confidence", 0)
                        r_conf_color = "#4caf50" if r_conf >= 0.7 else "#ff9800" if r_conf >= 0.4 else "#f44336"

                        st.markdown(f"""
<div style="background: rgba(102, 126, 234, 0.05); border-left: 3px solid {r_conf_color}; padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
        <span style="font-weight: 600;">📄 {source} <span style="color: #a0aec0; font-style: italic; font-weight: 400;">({doc_type})</span></span>
        <span style="background: {r_conf_color}22; color: {r_conf_color}; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; font-weight: 600;">{r_conf:.0%} relevant</span>
    </div>
    <div style="font-size: 0.85rem; color: #ccc; line-height: 1.4;">{r['text'][:250]}{'...' if len(r['text']) > 250 else ''}</div>
</div>
""", unsafe_allow_html=True)

    with rag_tab2:
        st.markdown("#### 📊 RAG Pipeline Evaluation")
        st.markdown("Run automated tests to measure retrieval accuracy and answer quality.")

        eval_col1, eval_col2 = st.columns([1, 2])

        with eval_col1:
            max_tests = st.slider("Number of tests", 5, 15, 10)
            run_answer_eval = st.checkbox("Include answer quality eval (requires Gemini)", value=bool(api_key))

            if st.button("🧪 Run Evaluation", type="primary", use_container_width=True):
                with st.spinner("Running RAG evaluation suite..."):
                    from models.rag_evaluator import RAGEvaluator
                    evaluator = RAGEvaluator(api_key=api_key)
                    from models.rag_evaluator import GROUND_TRUTH_TESTS

                    tests = GROUND_TRUTH_TESTS[:max_tests]
                    retrieval_report = evaluator.evaluate_retrieval(rag, tests)

                    st.session_state["rag_eval_retrieval"] = retrieval_report

                    if run_answer_eval and api_key:
                        quality_report = evaluator.evaluate_answer_quality(rag, gen, tests, max_tests=5)
                        st.session_state["rag_eval_quality"] = quality_report

        with eval_col2:
            if "rag_eval_retrieval" in st.session_state:
                report = st.session_state["rag_eval_retrieval"]
                m = report["metrics"]

                # Metrics cards
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    hr_color = "#4caf50" if m["hit_rate"] >= 0.8 else "#ff9800" if m["hit_rate"] >= 0.6 else "#f44336"
                    st.markdown(f"""
<div class="metric-card">
    <h3>Hit Rate</h3>
    <div class="value" style="color:{hr_color}">{m['hit_rate']:.0%}</div>
</div>""", unsafe_allow_html=True)
                with mc2:
                    mrr_color = "#4caf50" if m["mrr"] >= 0.7 else "#ff9800" if m["mrr"] >= 0.5 else "#f44336"
                    st.markdown(f"""
<div class="metric-card">
    <h3>MRR</h3>
    <div class="value" style="color:{mrr_color}">{m['mrr']:.3f}</div>
</div>""", unsafe_allow_html=True)
                with mc3:
                    st.markdown(f"""
<div class="metric-card">
    <h3>Type Match</h3>
    <div class="value" style="color:#667eea">{m['type_match_rate']:.0%}</div>
</div>""", unsafe_allow_html=True)
                with mc4:
                    st.markdown(f"""
<div class="metric-card">
    <h3>Avg Latency</h3>
    <div class="value" style="color:#e040fb">{m['avg_retrieval_ms']:.0f}ms</div>
</div>""", unsafe_allow_html=True)

                # Per-test details table
                st.markdown("##### Per-Test Results")
                test_data = []
                for d in report["details"]:
                    test_data.append({
                        "ID": d["test_id"],
                        "Question": d["question"][:45] + "..." if len(d["question"]) > 45 else d["question"],
                        "Hit": "✅" if d["hit"] else "❌",
                        "Type Match": "✅" if d["type_match"] else "❌",
                        "RR": f"{d['reciprocal_rank']:.3f}",
                        "Confidence": f"{d['avg_confidence']:.0%}",
                        "Latency": f"{d['retrieval_time_ms']:.0f}ms",
                    })
                st.dataframe(pd.DataFrame(test_data), use_container_width=True, hide_index=True)

            if "rag_eval_quality" in st.session_state:
                quality = st.session_state["rag_eval_quality"]
                if "avg_quality_score" in quality:
                    st.markdown(f"##### Answer Quality Score: **{quality['avg_quality_score']:.1f}/5.0**")
                    for d in quality.get("details", []):
                        scores = d.get("scores", {})
                        st.markdown(
                            f"- **{d['test_id']}**: Accuracy={scores.get('accuracy','?')}/5 "
                            f"Completeness={scores.get('completeness','?')}/5 "
                            f"Readability={scores.get('readability','?')}/5 "
                            f"→ Overall **{scores.get('overall','?')}/5**"
                        )

    with rag_tab3:
        st.markdown("#### 📜 Query History")
        history = st.session_state.get("rag_query_history", [])
        if not history:
            st.info("No queries yet. Ask a question in the 'Ask Questions' tab to start.")
        else:
            for i, h in enumerate(reversed(history)):
                conf = h["confidence"]
                conf_color = "#4caf50" if conf >= 0.7 else "#ff9800" if conf >= 0.4 else "#f44336"
                st.markdown(f"""
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0; border: 1px solid rgba(255,255,255,0.06);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 500;">💬 {h['question'][:60]}{'...' if len(h['question']) > 60 else ''}</span>
        <span style="display: flex; gap: 0.8rem; font-size: 0.8rem; color: #888;">
            <span style="color: {conf_color}; font-weight: 600;">{conf:.0%}</span>
            <span>{h['sources']} sources</span>
            <span>{h['time']}</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    api_key = render_sidebar()

    # Header
    st.markdown(
        '<div class="main-header">'
        '<h1>🛡️ GraphGuard</h1>'
        '<p>AI-Powered Fraud Detection & Dynamic Authentication System — Gemini LLM + RAG</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔑 Authentication",
        "🚨 Fraud Detection",
        "📊 Dashboard",
        "🔍 RAG Explorer",
        "🧠 User Habits",
    ])

    with tab1:
        render_auth_tab(api_key)
    with tab2:
        render_fraud_tab(api_key)
    with tab3:
        render_dashboard_tab()
    with tab4:
        render_rag_tab(api_key)
    with tab5:
        render_habits_tab()


# ---------------------------------------------------------------------------
# Cached habit model
# ---------------------------------------------------------------------------
@st.cache_resource
def get_habit_model():
    from models.habit_model import HabitPredictionModel
    model = HabitPredictionModel()
    model.train()
    return model


# ---------------------------------------------------------------------------
# Tab 5: User Habits — Transaction Habit Prediction
# ---------------------------------------------------------------------------
def render_habits_tab():
    st.markdown("### 🧠 User Transaction Habits")
    st.markdown("AI-learned spending patterns, preferences, and behavior predictions for each user.")

    model = get_habit_model()

    if not model.profiles:
        st.warning("No transaction data found. Run the pipeline first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        users = sorted(model.profiles.keys())
        selected_user = st.selectbox(
            "Select User",
            users,
            key="habit_user",
            format_func=lambda x: f"{USER_NAMES.get(x, x)} ({x})",
        )

        profile = model.get_profile(selected_user)
        if profile:
            # Habit score gauge
            score = profile.habit_score
            score_color = "#4caf50" if score >= 70 else "#ff9800" if score >= 40 else "#f44336"
            score_label = "Very Consistent" if score >= 70 else "Moderate" if score >= 40 else "Unpredictable"

            st.markdown(f"""
<div style="background: linear-gradient(145deg, #1e1e2e, #2a2a3e); border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; border: 1px solid rgba(255,255,255,0.08); text-align: center;">
    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: #a0a0b0;">Habit Consistency</div>
    <div style="font-size: 2.5rem; font-weight: 700; color: {score_color}; margin: 0.3rem 0;">{score}</div>
    <div style="font-size: 0.85rem; color: {score_color};">{score_label}</div>
    <div style="background: rgba(255,255,255,0.1); border-radius: 8px; height: 6px; margin-top: 0.5rem;">
        <div style="background: {score_color}; height: 100%; width: {score}%; border-radius: 8px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

            st.metric("Total Transactions", profile.total_transactions)
            st.metric("Total Spent", f"${profile.total_spent:,.0f}")

    with col2:
        profile = model.get_profile(selected_user)
        if not profile:
            st.info("No profile available.")
            return

        # --- Habit Profile Cards ---
        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            st.markdown(f"""
<div class="metric-card">
    <h3>🏪 Favorite Store</h3>
    <div class="value" style="color:#667eea; font-size:1.3rem;">{profile.favorite_category}</div>
</div>""", unsafe_allow_html=True)
        with hcol2:
            st.markdown(f"""
<div class="metric-card">
    <h3>📍 Primary City</h3>
    <div class="value" style="color:#764ba2; font-size:1.3rem;">{profile.primary_location}</div>
</div>""", unsafe_allow_html=True)
        with hcol3:
            st.markdown(f"""
<div class="metric-card">
    <h3>⏰ Peak Time</h3>
    <div class="value" style="color:#e040fb; font-size:1.3rem;">{profile.peak_hour_label}</div>
</div>""", unsafe_allow_html=True)

        hcol4, hcol5, hcol6 = st.columns(3)
        with hcol4:
            st.markdown(f"""
<div class="metric-card">
    <h3>📅 Busiest Day</h3>
    <div class="value" style="color:#4caf50; font-size:1.3rem;">{profile.peak_day_label}</div>
</div>""", unsafe_allow_html=True)
        with hcol5:
            st.markdown(f"""
<div class="metric-card">
    <h3>💰 Avg Spend</h3>
    <div class="value" style="color:#ff9800; font-size:1.3rem;">${profile.avg_amount:.0f}</div>
</div>""", unsafe_allow_html=True)
        with hcol6:
            emoji = "🏖️" if profile.prefers_weekend else "💼"
            label = "Weekend" if profile.prefers_weekend else "Weekday"
            st.markdown(f"""
<div class="metric-card">
    <h3>{emoji} Prefers</h3>
    <div class="value" style="color:#00bcd4; font-size:1.3rem;">{label}</div>
</div>""", unsafe_allow_html=True)

        # --- Top 5 Most Visited Stores with Avg Spend ---
        st.markdown("##### 🏪 Top 5 Most Visited Stores")
        if profile.top_stores_with_spend:
            max_visits = profile.top_stores_with_spend[0]["visits"]
            for s in profile.top_stores_with_spend:
                bar_pct = (s["visits"] / max_visits) * 100 if max_visits > 0 else 0
                st.markdown(f"""
<div style="display: flex; align-items: center; margin: 0.4rem 0; gap: 0.6rem;">
    <span style="width: 110px; font-size: 0.85rem; font-weight: 500;">{s['store']}</span>
    <div style="flex: 1; background: rgba(255,255,255,0.08); border-radius: 6px; height: 28px; position: relative;">
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {bar_pct:.0f}%; border-radius: 6px; display: flex; align-items: center; padding-left: 8px;">
            <span style="font-size: 0.75rem; color: white; font-weight: 600;">{s['visits']} visits</span>
        </div>
    </div>
    <span style="background: rgba(255,152,0,0.15); color: #ff9800; padding: 0.2rem 0.6rem; border-radius: 8px; font-size: 0.8rem; font-weight: 600; white-space: nowrap;">avg ${s['avg_spend']:.0f}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No store data available.")

        st.markdown("")

        # --- Predictions ---
        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.markdown("##### 🔮 Next Purchase Prediction")
            cat_preds = model.predict_next_category(selected_user)
            for p in cat_preds[:4]:
                bar_width = p["probability"] * 100
                st.markdown(f"""
<div style="display: flex; align-items: center; margin: 0.3rem 0; gap: 0.5rem;">
    <span style="width: 100px; font-size: 0.85rem;">{p['category']}</span>
    <div style="flex: 1; background: rgba(255,255,255,0.1); border-radius: 4px; height: 20px;">
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {bar_width:.0f}%; border-radius: 4px; display: flex; align-items: center; padding-left: 6px; font-size: 0.7rem; color: white; font-weight: 600;">{p['probability']:.0%}</div>
    </div>
</div>""", unsafe_allow_html=True)

        with pred_col2:
            st.markdown("##### 👥 Similar Users")
            similar = model.find_similar_users(selected_user)
            for s in similar:
                sim_pct = s["similarity"] * 100
                st.markdown(f"""
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.3rem 0; border: 1px solid rgba(255,255,255,0.06);">
    <div style="display: flex; justify-content: space-between;">
        <span style="font-weight: 500;">{USER_NAMES.get(s['user_id'], s['user_id'])}</span>
        <span style="color: #667eea; font-weight: 600;">{sim_pct:.0f}% match</span>
    </div>
    <span style="font-size: 0.75rem; color: #888;">{s['shared_categories']} shared categories</span>
</div>""", unsafe_allow_html=True)

        # --- Anomaly Checker ---
        st.markdown("---")
        st.markdown("##### 🔍 Transaction Anomaly Check")
        st.markdown("Test if a new transaction matches this user's habits:")

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            test_amount = st.number_input("Amount ($)", value=100.0, min_value=0.01, step=10.0, key="habit_amount")
        with ac2:
            categories = list(profile.top_categories.keys()) + ["Jewelry", "Gas Stations", "Travel"]
            test_cat = st.selectbox("Category", sorted(set(categories)), key="habit_cat")
        with ac3:
            locations = list(profile.top_locations.keys()) + ["Anchorage, AK", "Las Vegas, NV"]
            test_loc = st.selectbox("Location", sorted(set(locations)), key="habit_loc")

        if st.button("🧪 Check Transaction", type="primary"):
            result = model.check_transaction(selected_user, test_amount, test_cat, test_loc)
            if result.get("is_anomalous"):
                st.error(f"⚠️ **Unusual Transaction** (anomaly score: {result['anomaly_score']:.0%})")
                for flag in result.get("flags", []):
                    st.markdown(f"- {flag}")
            else:
                st.success(f"✅ **Normal Transaction** — matches {USER_NAMES.get(selected_user, selected_user)}'s habits")

    # --- All Users Summary Table ---
    st.markdown("---")
    st.markdown("##### 📊 All Users — Habit Summary")
    summary_df = model.get_all_profiles_summary()
    summary_df["User"] = summary_df["User"].apply(lambda x: f"{USER_NAMES.get(x, x)} ({x})")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

