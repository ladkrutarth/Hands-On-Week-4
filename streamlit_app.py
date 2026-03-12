"""
Veriscan — Unified AI Fraud & Compliance Dashboard
AI-Powered Fraud Detection | Dynamic Auth | CFPB Analysis & RAG
"""

import os
# ---------------------------------------------------------------------------
# System Stability Guards (Fixes SIGABRT on macOS Sequoia)
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import sys
from pathlib import Path

import uuid
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests

# API Backend URL
API_BASE_URL = os.environ.get("VERISCAN_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "financial_advisor_dataset.csv"
CFPB_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "cfpb_credit_card.csv"

# ---------------------------------------------------------------------------
# Premium Palette & Chart Helpers
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#F7F7F7",        # Clean white background
    "surface": "#FFFFFF",
    "border": "#E5E7EB",
    "text": "#0B1220",
    "muted": "#5B6474",
    "grid": "#EEF2F7",
    "brand": "#1E2761",     # Midnight Blue
    "brand_2": "#408EC6",   # Royal Blue
    "success": "#7A2048",   # Burgundy accent (used sparingly)
    "warn": "#F97316",      # Orange
    "danger": "#DC2626",    # Red
    "slate": "#111827",
}

# Qualitative & sequential palettes (Seaborn-like choices)
QUAL_CATEGORICAL = ["#1E2761", "#FF7F0E", "#2CA02C", "#D62728"]  # blue, orange, green, red
SEQ_BLUE = ["#EFF6FF", "#BFDBFE", "#60A5FA", "#1D4ED8", "#1E2761"]
DIV_BLUE_RED = ["#7A2048", "#F97316", "#FACC15", "#22C55E", "#1D4ED8"]

CHART_TEXT_COLOR = PALETTE["text"]
CHART_FONT = {"family": "Inter", "size": 14, "color": CHART_TEXT_COLOR}


def apply_accessible_theme(fig, *, title: str | None = None):
    """Apply a unified, premium chart theme."""
    if title:
        fig.update_layout(title={"text": title, "x": 0.0, "xanchor": "left"})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["surface"],
        font=CHART_FONT,
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        title_font={"size": 20, "family": "Inter", "color": CHART_TEXT_COLOR, "weight": "bold"},
        legend_font={"size": 13, "color": PALETTE["muted"]},
        colorway=[PALETTE["brand"], PALETTE["brand_2"], PALETTE["success"], PALETTE["warn"], PALETTE["danger"]],
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        tickfont={"size": 12, "color": PALETTE["muted"]},
        title_font={"size": 13, "color": PALETTE["muted"], "weight": "bold"},
        linecolor=PALETTE["border"],
        linewidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=PALETTE["grid"],
        tickfont={"size": 12, "color": PALETTE["muted"]},
        title_font={"size": 13, "color": PALETTE["muted"], "weight": "bold"},
        linecolor=PALETTE["border"],
        linewidth=1,
    )
    return fig

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Veriscan — Unified Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (Mobbin Minimalist Aesthetic)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Mobbin Minimalist Aesthetic + Premium Palette */
    .stApp {
        background-color: #F7F7F7;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #111827;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Clean Header */
    .main-header {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 2.5rem 3.5rem;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #6B7280;
        font-size: 0.98rem;
        margin-top: 0.5rem;
        max-width: 52rem;
    }

    /* Minimalist Bento Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: left;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        border-color: #D1D5DB;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }

    .metric-card h3 { 
        margin: 0; 
        font-size: 0.875rem; 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
        color: #6B7280; 
        font-weight: 600; 
    }
    
    .metric-card .value { 
        font-size: 1.9rem; 
        font-weight: 700; 
        margin: 0.5rem 0 0 0; 
        color: #111827; 
        letter-spacing: -0.02em;
    }

    hr, div[data-testid="stDivider"] hr {
        border-color: #E5E7EB !important;
    }

    /* Stark Risk Badges */
    .risk-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .risk-low { background: #F3F4F6; color: #374151; border: 1px solid #E5E7EB; }
    .risk-medium { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .risk-high { background: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
    .risk-critical { background: #111827; color: #FFFFFF; border: 1px solid #000000; }

    /* Clean typography for standard markdown */
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #111827 !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    .stMarkdown h2 { font-size: 1.45rem !important; }
    .stMarkdown h3 { font-size: 1.25rem !important; }
    .stMarkdown h4 { font-size: 1.08rem !important; }
    
    /* Clean Inputs — light background, friendly focus (Royal Blue) */
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stTextArea"] div[data-baseweb="textarea"],
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        color: #1E2761 !important;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #1E2761 !important;
        font-size: 1rem !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="textarea"]:focus-within {
        border-color: #408EC6 !important;
        box-shadow: 0 0 0 2px rgba(64, 142, 198, 0.25) !important;
    }

    /* Monochromatic Buttons */
    div.stButton > button {
        background: #111827 !important;
        color: #FFFFFF !important;
        border: 1px solid #111827 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: #374151 !important;
        border-color: #374151 !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary outline buttons (like Streamlit defaults, overridden) */
    div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #E5E7EB !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #F9FAFB !important;
        border-color: #D1D5DB !important;
    }

    /* RAG response box */
    .rag-answer {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1rem;
        line-height: 1.6;
        color: #374151;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Tabs Override */
    div[data-testid="stTabs"] button {
        color: #6B7280;
        font-weight: 500;
        font-size: 1rem;
        border-bottom-color: transparent !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #111827 !important;
        border-bottom-color: #111827 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F7F7F7 !important;
        border-right: 1px solid #E5E7EB;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    section[data-testid="stSidebar"] label {
        color: #4B5563 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #111827 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        font-size: 1.2rem !important;
    }

    /* Status pills row */
    .top-row {
        display: flex;
        gap: 0.75rem;
        align-items: stretch;
        flex-wrap: wrap;
        margin: 0.75rem 0 1.25rem 0;
    }
    .pill {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 999px;
        padding: 0.45rem 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.05);
        font-size: 0.85rem;
        color: #0B1220;
    }
    .dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: #9CA3AF;
    }
    .dot.ok { background: #16A34A; }
    .dot.bad { background: #DC2626; }

    /* High-Contrast Widget Labels & Info Text */
    div[data-testid="stWidgetLabel"] p,
    .stAlert p,
    table, th, td, 
    div[data-testid="stTable"] td, 
    div[data-testid="stTable"] th {
        color: #111827 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

import random

# ---------------------------------------------------------------------------
# Data & Resource Loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_fraud_data(path: Path, mtime: float):
    if path.exists():
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        # Compatibility fix: ensure is_fraud_flag exists if it's the advisor dataset
        if 'is_fraud_flag' not in df.columns:
            if 'is_fraud' in df.columns:
                df = df.rename(columns={'is_fraud': 'is_fraud_flag'})
            elif 'is_fraud_flag' not in df.columns:
                # Last resort: create the column if missing
                df['is_fraud_flag'] = False
        return df
    return pd.DataFrame()

@st.cache_data(ttl=600)
def load_cfpb_data():
    if CFPB_PATH.exists():
        # Large scale analysis enabled
        return pd.read_csv(CFPB_PATH, nrows=50000)
    return pd.DataFrame()

def api_available() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=10)
def api_health() -> dict | None:
    """Lightweight cached health check for status pills."""
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

@st.cache_resource
def get_llm_via_api(prompt: str, max_tokens: int = 500) -> str:
    """Calls the backend API for LLM generation to avoid GPU contention in the frontend."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/llm/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("response", "Error: No response from API.")
        return f"Error: API returned {resp.status_code}"
    except Exception as e:
        return f"Error connecting to AI Backend: {e}"

# ---------------------------------------------------------------------------
# Aesthetics & Accessibility Helpers
# ---------------------------------------------------------------------------

def risk_badge_html(level: str) -> str:
    cls = f"risk-{level.lower()}"
    return f'<span class="risk-badge {cls}">{level}</span>'


# ---------------------------------------------------------------------------
# Auth & Session Helpers
# ---------------------------------------------------------------------------

def _login_via_api(username: str, password: str) -> dict:
    """Call FastAPI /api/auth/login and return JSON or raise."""
    resp = requests.post(
        f"{API_BASE_URL}/api/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Auth API error: {resp.status_code}")
    return resp.json()


def render_login_page():
    """Render a simple login screen before exposing the main dashboard."""
    st.markdown("""
    <div class="main-header">
        <h1>Veriscan Login</h1>
        <p>Sign in to access the Fraud & Intelligence Dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    if not api_available():
        st.error("Backend API is not available. Please start the FastAPI server (uvicorn api.main:app --port 8000).")
        return

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign In")

    if submitted:
        if not username or not password:
            st.warning("Please enter both username and password.")
            return
        try:
            data = _login_via_api(username, password)
        except Exception as e:
            st.error(f"Login failed: {e}")
            return

        if not data.get("authenticated"):
            st.error(data.get("message", "Invalid username or password."))
            return

        # Successful login → store session state
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.session_state["session_id"] = data.get("session_id")
        # Reuse the same session_id for security chat routing
        st.session_state["security_session_id"] = st.session_state["session_id"]
        st.success("Login successful.")


def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='color:#0B1220;'>🛡️ Veriscan</h2>", unsafe_allow_html=True)
        st.markdown("<span style='color:#5B6474;font-weight:500;'>Security & Intelligence Hub</span>", unsafe_allow_html=True)
        st.divider()

        if st.session_state.get("authenticated"):
            st.markdown(f"**User:** {st.session_state.get('username', '')}")
            if st.button("Logout", key="sidebar_logout"):
                for key in ["authenticated", "username", "session_id", "security_session_id"]:
                    st.session_state.pop(key, None)
                st.experimental_rerun()

        st.divider()
        st.markdown("### Navigation")
        st.session_state.setdefault("nav", "🛡️ Security AI")
        st.session_state["nav"] = st.radio(
            "Go to",
            ["🛡️ Security AI", "💰 Financial AI", "📄 PDF Intelligence", "📊 Market Dash", "🔍 CFPB Market Intel", "🧬 Spending DNA"],
            index=["🛡️ Security AI", "💰 Financial AI", "📄 PDF Intelligence", "📊 Market Dash", "🔍 CFPB Market Intel", "🧬 Spending DNA"].index(
                st.session_state.get("nav", "🛡️ Security AI")
            ),
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### System")
        h = api_health()
        if h:
            st.success("Backend: online")
            st.caption(f"API v{h.get('version','—')}")
        else:
            st.error("Backend: offline")

        fraud_df = load_fraud_data(FEATURES_PATH, FEATURES_PATH.stat().st_mtime if FEATURES_PATH.exists() else 0)
        cfpb_df = load_cfpb_data()
        st.markdown(f"""
        - 💸 Transactions: **{len(fraud_df):,}**
        - 📝 Complaints: **{len(cfpb_df):,}**
        """)

        st.divider()
        st.markdown("### Runtime")
        st.markdown("- **LLM**: Local MLX-LM (Llama 3 class)")
        st.markdown("- **RAG**: Chroma + MiniLM embeddings")

# ---------------------------------------------------------------------------
# Tab 1: Fraud Dashboard
# ---------------------------------------------------------------------------
def render_dashboard_tab(df):
    if df.empty:
        st.warning("Fraud dataset not found.")
        return

    st.markdown("### 📊 Market Fraud Overview")
    
    st.subheader("Top 10 Online Scam & Crime Types by Losses (Global)")
    st.markdown("Global reported losses by category. *Note: Actual losses are likely higher due to underreporting.*")
    
    scam_data = pd.DataFrame({
        "Scam Type": [
            "Investment Scams", "Business Email Compromise", "Tech Support Scams", 
            "Non-Payment/Delivery", "Confidence/Romance Scams", "Govt. Impersonation", 
            "Employment Scams", "Credit Card/Check Fraud", "Real Estate/Rental", 
            "Misc. Cyber-enabled"
        ],
        "Losses ($B)": [46.9, 19.8, 10.5, 5.6, 4.8, 2.9, 2.0, 1.9, 1.4, 1.2]
    }).sort_values(by="Losses ($B)", ascending=True)

    fig_scam = px.bar(
        scam_data, x="Losses ($B)", y="Scam Type", orientation='h',
        color="Losses ($B)", color_continuous_scale='Sunset',
        template='plotly_white', text="Losses ($B)"
    )
    fig_scam.update_traces(texttemplate='$%{text}B', textposition='outside')
    fig_scam.update_layout(showlegend=False, coloraxis_showscale=False, yaxis_title=None, height=450)
    
    st.plotly_chart(apply_accessible_theme(fig_scam), use_container_width=True)

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='metric-card'><h3>Processed</h3><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h3>Fraud Cases</h3><div class='value' style='color:#dc2626'>{df['is_fraud_flag'].sum():,}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h3>Fraud Risk</h3><div class='value'>{(df['is_fraud_flag'].mean()*100):.1f}%</div></div>", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud Heatmap by Category")
        cat_data = df.groupby('category')['is_fraud_flag'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat_data, x='is_fraud_flag', y='category', orientation='h', color='is_fraud_flag', 
                     color_continuous_scale='Sunset', template='plotly_white')


        st.plotly_chart(apply_accessible_theme(fig), use_container_width=True)
    
    with col2:
        st.subheader("Interactive Fraud Bubble Map (US)")
        state_data = df.groupby("state")["is_fraud_flag"].sum().reset_index()
        state_data = state_data[state_data["state"].notna()]
        if not state_data.empty:
            min_cases = int(state_data["is_fraud_flag"].min())
            max_cases = int(state_data["is_fraud_flag"].max())
            cutoff = st.slider(
                "Show states with at least this many fraud cases:",
                min_value=min_cases,
                max_value=max_cases,
                value=min(max_cases // 10, max_cases),
                key="fraud_map_cutoff",
            )
            filtered = state_data[state_data["is_fraud_flag"] >= cutoff]
            fig = px.scatter_geo(
                filtered,
                locations="state",
                locationmode="USA-states",
                size="is_fraud_flag",
                color="is_fraud_flag",
                scope="usa",
                hover_name="state",
                labels={"is_fraud_flag": "Fraud cases"},
                color_continuous_scale=SEQ_BLUE,
            )
            fig.update_traces(marker_line_color="white", marker_line_width=0.6, opacity=0.9)
            st.plotly_chart(apply_accessible_theme(fig, title="Fraud cases by state (bubble map)"), use_container_width=True)
        else:
            st.info("No state-level data available for the map.")



# ---------------------------------------------------------------------------
# Tab 3: CFPB Market Intelligence
# ---------------------------------------------------------------------------
def render_cfpb_tab(df):
    st.markdown("### 🔍 CFPB Credit Card Intelligence")
    if df.empty:
        st.warning("CFPB dataset missing or empty.")
        return

    reporting_cos = [
        "EQUIFAX, INC.", 
        "TRANSUNION INTERMEDIATE HOLDINGS, INC.", 
        "EXPERIAN INFORMATION SOLUTIONS, INC.",
        "LEXISNEXIS RISK DATA MANAGEMENT INC."
    ]
    
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown("#### 🏛️ Reporting Agencies")
        rep_df = df[df['Company'].isin(reporting_cos)]
        rep_counts = rep_df['Company'].value_counts().reset_index()
        fig_rep = px.bar(rep_counts, x='count', y='Company', orientation='h', template='plotly_white', 
                         color_discrete_sequence=[QUAL_CATEGORICAL[1]], title="Credit Reporting Volume")
        st.plotly_chart(apply_accessible_theme(fig_rep), use_container_width=True)

    with m2:
        st.markdown("#### 💳 Card Issuers")
        iss_df = df[~df['Company'].isin(reporting_cos)]
        iss_counts = iss_df['Company'].value_counts().head(8).reset_index()
        fig_iss = px.bar(iss_counts, x='count', y='Company', orientation='h', template='plotly_white', 
                         color_discrete_sequence=[QUAL_CATEGORICAL[0]], title="Top Issuer Complaints")
        st.plotly_chart(apply_accessible_theme(fig_iss), use_container_width=True)

    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### 🗺️ Geographic Distribution (Bubble Map)")
        state_data = df.groupby("State").size().reset_index(name="Complaints")
        state_data = state_data[state_data["State"].notna()]
        if not state_data.empty:
            min_c = int(state_data["Complaints"].min())
            max_c = int(state_data["Complaints"].max())
            cutoff_c = st.slider(
                "Show states with at least this many complaints:",
                min_value=min_c,
                max_value=max_c,
                value=min(max_c // 15 or max_c, max_c),
                key="cfpb_map_cutoff",
            )
            filtered_c = state_data[state_data["Complaints"] >= cutoff_c]
            fig2 = px.scatter_geo(
                filtered_c,
                locations="State",
                locationmode="USA-states",
                size="Complaints",
                color="Complaints",
                scope="usa",
                hover_name="State",
                labels={"Complaints": "Complaints"},
                color_continuous_scale=DIV_BLUE_RED,
            )
            fig2.update_traces(marker_line_color="white", marker_line_width=0.6, opacity=0.9)
            st.plotly_chart(apply_accessible_theme(fig2, title="Complaint distribution by state (bubble map)"), use_container_width=True)
        else:
            st.info("No state-level complaint data available for the map.")

    with c2:
        st.markdown("#### 💬 AI Intelligence & Search")
        st.info("Query the knowledge base for fraud patterns or policy details.")
        query = st.text_input("Search context:", placeholder="e.g. Find high-value travel anomalies...", key="cfpb_rag_q")
        if st.button("Query Knowledge Base", key="cfpb_rag_btn") and query:
            with st.spinner("Analyzing..."):
                try:
                    sid = st.session_state.get("session_id")
                    payload = {"query": query, "n_results": 5}
                    if sid:
                        payload["session_id"] = sid
                    resp = requests.post(f"{API_BASE_URL}/api/rag/query", json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data["results"]:
                            for r in data["results"]:
                                st.markdown(f"<div class='rag-answer'>{r}</div>", unsafe_allow_html=True)
                        else:
                            st.info("No matching context found.")
                except Exception as e:
                    st.error(f"RAG Error: {e}")



# ---------------------------------------------------------------------------
# Tab 1: AI Specialist Models (Security & Financial)
# ---------------------------------------------------------------------------
def render_omni_tab():
    from agents.financial_advisor_agent import FinancialAdvisorAgent
    
    st.markdown("### 🤖 AI Specialist Models")
    st.caption("Select your specialized AI agent for focused analysis.")

    # 1. User Context Selector
    try:
        adv = FinancialAdvisorAgent()
        all_users = adv.get_all_users()
    except Exception:
        all_users = ["USER_0001"]

    colX, colY = st.columns([1, 2])
    with colX:
        st.markdown("<p style='color: black; font-weight: bold; font-size: 14px; margin-bottom: 5px; margin-left: 2px;'>👤 Target Context:</p>", unsafe_allow_html=True)
        selected_user = st.selectbox("", all_users[:50], key="omni_user", label_visibility="collapsed")
    
    with colY:
        st.markdown("<p style='color: black; font-weight: bold; font-size: 14px; margin-bottom: 5px; margin-left: 2px;'>🧠 Select AI Model:</p>", unsafe_allow_html=True)
        selected_model = st.selectbox(
            "",
            ["🛡️ Security AI Analyst", "💰 Financial AI Advisor"],
            key="model_selector",
            label_visibility="collapsed"
        )
        
    is_security_model = "Security" in selected_model
    is_financial_model = "Financial" in selected_model

    # 2. Dynamic Top Widget (Shield vs. Chart preview)
    if is_security_model:
        try:
            monitor = adv.tool_suspicious_activity_monitor(selected_user)
            if monitor["alert_count"] > 0:
                overall = monitor["overall_status"]
                border = "#dc2626" if "CRITICAL" in overall else "#d97706"
                st.markdown(
                    f"""<div style='background:rgba(220,38,38,0.05);border:1px solid {border};border-radius:12px;padding:1rem;margin-bottom:1.5rem;'>
                    <div style='font-weight:700;font-size:1rem;margin-bottom:0.5rem;'>🛡️ System Shield Live Monitor — <span style='color:{border};'>{overall}</span></div>
                    <p style='font-size:0.9rem; margin-bottom:0.5rem;'>Detected {monitor['alert_count']} anomalies requiring immediate attention.</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ **System Shield:** Active monitoring shows no suspicious activity.", icon="🛡️")
        except Exception:
            pass
    elif is_financial_model:
        try:
            summary = adv.tool_spending_summary(selected_user)
            st.info(f"📊 **Quick Review:** {summary['archetype'].replace('_', ' ').title()} spender. Monthly Avg: **${summary['avg_monthly_spend']:,.2f}**")
        except Exception:
            pass

    # 3. Dedicated Chat & Investigation Interface
    st.markdown(f"#### 💬 {selected_model} Interface")
    
    if "run_omni" not in st.session_state:
        st.session_state.run_omni = False

    def set_preset(query):
        st.session_state.omni_input = query
        st.session_state.run_omni = True

    # Dynamic Quick Answer Buttons based on model
    if is_security_model:
        qa1, qa2 = st.columns(2)
        qa1.button("🔍 Run Fraud Scan", on_click=set_preset, args=("Scan my recent transactions for any signs of fraud.",), use_container_width=True, key="sec_qa1")
        qa2.button("🛡️ Anomaly Protocol", on_click=set_preset, args=("Execute the anomaly detection protocol and report risks.",), use_container_width=True, key="sec_qa2")
        placeholder = "e.g. 'Investigate the recent high-value transactions for fraud alerts.'"
    else:
        # High Level Overview
        qa1, qa2, qa3 = st.columns(3)
        qa1.button("📊 Spend Review", on_click=set_preset, args=("Provide a full spending portfolio review.",), use_container_width=True, key="fin_qa1")
        qa2.button("💰 Savings Plan", on_click=set_preset, args=("Generate a targeted savings strategy for my archetype.",), use_container_width=True, key="fin_qa2")
        qa3.button("💳 Credit Impact", on_click=set_preset, args=("Estimate my credit health and identifies risk factors.",), use_container_width=True, key="fin_qa3")
        
        # Advanced Analytics
        qa4, qa5, qa6 = st.columns(3)
        qa4.button("📉 Cash Flow Forecast", on_click=set_preset, args=("Forecast my cash flow for the next 30 days.",), use_container_width=True, key="fin_qa4")
        qa5.button("🔔 Price Hike Alert", on_click=set_preset, args=("Detect any recent price hikes in my subscriptions.",), use_container_width=True, key="fin_qa5")
        qa6.button("🧾 Tax-Deductible Finder", on_click=set_preset, args=("Find potential tax-deductible expenses.",), use_container_width=True, key="fin_qa6")
        
        # Optimization
        qa7, qa8 = st.columns(2)
        qa7.button("📈 Surplus Optimizer", on_click=set_preset, args=("Optimize my monthly surplus.",), use_container_width=True, key="fin_qa7")
        qa8.button("💧 Liquidity Guard", on_click=set_preset, args=("Check my upcoming bills vs assumed balance.",), use_container_width=True, key="fin_qa8")
        
        placeholder = "e.g. 'How much did I spend on coffee this month, and how can I save?'"

    user_q = st.text_area("Request Intelligence:", 
                         placeholder=placeholder,
                         height=100, key="omni_input")
    
    execute_clicked = st.button(f"🚀 Execute {selected_model.split()[1]} Reasoning", key="omni_btn", type="primary")
    
    if (execute_clicked or st.session_state.run_omni) and user_q:
        st.session_state.run_omni = False  # Reset the trigger
        with st.spinner(f"{selected_model.split()[1]} Agent is analyzing..."):
            try:
                if is_security_model:
                    # Route to Security Endpoint with global session_id for ADDF diversion state
                    sid = st.session_state.get("session_id") or st.session_state.setdefault("security_session_id", str(uuid.uuid4())[:12])
                    resp = requests.post(f"{API_BASE_URL}/api/security/chat", json={"message": user_q, "session_id": sid}, timeout=120)
                else:
                    # Route strictly to Financial Advisor
                    payload = {"user_id": selected_user, "message": user_q}
                    sid = st.session_state.get("session_id")
                    if sid:
                        payload["session_id"] = sid
                    resp = requests.post(f"{API_BASE_URL}/api/advisor/chat", json=payload, timeout=120)

                if resp.status_code == 200:
                    res = resp.json()
                    
                    st.info(f"🧠 Reasoning Mode: **{selected_model}**")
                    
                    # Show final response with proper markdown rendering
                    with st.container():
                        st.markdown(res["reply"])

                    # Show Chart *only* for Financial Advisor if requested
                    if is_financial_model and res.get("show_chart") and res.get("chart_data"):
                        with st.container():
                            st.markdown("#### 📊 Diagnostic Portfolio Visualization")
                            data = res["chart_data"]
                            fig = px.bar(
                                x=list(data.keys()), 
                                y=list(data.values()), 
                                labels={'x': 'Category', 'y': 'Total Allocation ($)'},
                                color_discrete_sequence=['#111827'],
                                text_auto='.2s'
                            )
                            fig.update_layout(
                                height=400, 
                                margin=dict(l=20, r=20, t=20, b=20),
                                paper_bgcolor="white",
                                plot_bgcolor="white",
                                font=dict(family="Outfit, sans-serif", size=13, color="#000000"),
                                coloraxis_showscale=False,
                                title_font=dict(color="#000000")
                            )
                            fig.update_xaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                            fig.update_yaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                            fig.update_traces(textposition='outside', cliponaxis=False, textfont=dict(color="#000000"))
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Trace rendering removed to enforce a clean, professional, sub-300 word summary via the Agent.
                else:
                    st.error(f"API Error: {resp.status_code}")
            except Exception as e:
                st.error(f"🔴 Connection Error: {e}")

    # =======================================================================
    # Bottom Domain Diagnostics (Context Aware)
    # =======================================================================
    st.divider()
    st.markdown("#### 💡 Diagnostic Shortcuts")
    
    if is_financial_model:
        st.info("Visualizing deep financial diagnostics.")
        
        # ── Spending Analysis: Local chart prompt ────────────────────────────
        if st.button("📊 View Category Breakdown", use_container_width=True):
            with st.spinner("Analyzing spending categories..."):
                try:
                    adv_local = FinancialAdvisorAgent()
                    chart_data = adv_local.get_chart_data(selected_user)
                    if chart_data:
                        st.markdown("#### 📊 Spending Category Breakdown")
                        fig = px.bar(
                            x=list(chart_data.keys()),
                            y=list(chart_data.values()),
                            labels={'x': 'Category', 'y': 'Total Spend ($)'},
                            color_discrete_sequence=['#111827'],
                            text_auto='.2s'
                        )
                        fig.update_layout(
                            height=400,
                            margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            font=dict(family="Outfit, sans-serif", size=13, color="#000000"),
                            coloraxis_showscale=False
                        )
                        fig.update_xaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                        fig.update_yaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
                        fig.update_traces(textposition='outside', cliponaxis=False, textfont=dict(color="#000000"))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No spending data found for this user.")
                except Exception as e:
                    st.error(f"Analysis Error: {e}")

        # Summary Metrics - Force Black Text
        try:
            adv_local = FinancialAdvisorAgent()
            summary = adv_local.tool_spending_summary(selected_user)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div style='color: black; font-size: 0.9em; opacity: 0.8;'>💰 Total Spend</div><div style='color: black; font-size: 1.8em; font-weight: bold;'>${summary.get('total_spend', 0):,.2f}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div style='color: black; font-size: 0.9em; opacity: 0.8;'>📅 Monthly Avg</div><div style='color: black; font-size: 1.8em; font-weight: bold;'>${summary.get('avg_monthly_spend', 0):,.2f}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div style='color: black; font-size: 0.9em; opacity: 0.8;'>🏷️ Top Merchant</div><div style='color: black; font-size: 1.8em; font-weight: bold;'>{summary.get('top_merchant', 'N/A')}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Summary Error: {e}")
    else:
        st.info("Running localized security sweeps.")
        # ── Anomaly Protocol: Instant local fraud scan ────────────────────────
        if st.button("🔍 Execute Anomaly Protocol", use_container_width=True):
            with st.spinner("Scanning for anomalies..."):
                try:
                    from agents.financial_advisor_agent import FinancialAdvisorAgent
                    adv_local = FinancialAdvisorAgent()
                    fraud = adv_local.tool_realtime_fraud_check(selected_user)
                    monitor = adv_local.tool_suspicious_activity_monitor(selected_user)

                    status_fraud = fraud.get("overall_status", "✅ CLEAR")
                    status_monitor = monitor.get("overall_status", "✅ CLEAR")
                    alert_count = fraud.get("alerts_found", 0) + monitor.get("alert_count", 0)
                    avg_risk = fraud.get("avg_risk_score", 0)

                    # Build concise ~150-word report from real data
                    report_parts = [
                        f"**Anomaly Detection Protocol — Complete**\n",
                        f"**Fraud Scan:** {status_fraud}",
                        f"**Activity Monitor:** {status_monitor}",
                        f"**Transactions Scanned:** {fraud.get('transactions_scanned', 0)} | **Alerts Found:** {alert_count} | **Avg Risk Score:** {avg_risk:.3f}\n",
                    ]

                    if fraud.get("alerts"):
                        report_parts.append("**⚠️ Flagged Transactions:**")
                        for a in fraud["alerts"][:5]:
                            flags_str = ", ".join(a["flags"][:2])
                            report_parts.append(f"- {a['merchant']} (${a['amount']}): {flags_str}")
                    elif monitor.get("alerts"):
                        report_parts.append("**⚠️ Suspicious Activity:**")
                        for a in monitor["alerts"][:5]:
                            report_parts.append(f"- {a['title']}: {a['detail']}")

                    st.markdown("\n".join(report_parts))
                except Exception as e:
                    st.error(f"Anomaly Protocol Error: {e}")


# ---------------------------------------------------------------------------
# Premium AI Pages: Security & Financial (separate)
# ---------------------------------------------------------------------------
def _get_all_users_financial() -> list[str]:
    from agents.financial_advisor_agent import FinancialAdvisorAgent
    try:
        adv = FinancialAdvisorAgent()
        return adv.get_all_users() or ["USER_0001"]
    except Exception:
        return ["USER_0001"]


def _top_status_row(*, model_label: str):
    h = api_health()
    backend_ok = bool(h)
    dot_cls = "ok" if backend_ok else "bad"
    st.markdown(
        f"""
        <div class="top-row">
          <div class="pill"><span class="dot {dot_cls}"></span><strong>Backend</strong> <span style="color:#5B6474;">{("online" if backend_ok else "offline")}</span></div>
          <div class="pill"><span class="dot ok"></span><strong>Current model</strong> <span style="color:#5B6474;">{model_label}</span></div>
          <div class="pill"><span class="dot ok"></span><strong>Runtime</strong> <span style="color:#5B6474;">Local MLX-LM + RAG</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_security_ai_page():
    from agents.financial_advisor_agent import FinancialAdvisorAgent

    st.markdown("### 🛡️ Security AI Analyst")
    st.caption("Threat scanning, anomaly protocols, and fraud-focused security responses.")
    _top_status_row(model_label="Security Analyst")

    all_users = _get_all_users_financial()
    st.markdown("#### Context")
    selected_user = st.selectbox("Target user", all_users[:50], key="sec_user")

    # Live monitor
    try:
        adv = FinancialAdvisorAgent()
        monitor = adv.tool_suspicious_activity_monitor(selected_user)
        if monitor.get("alert_count", 0) > 0:
            overall = monitor.get("overall_status", "ALERT")
            border = PALETTE["danger"] if "CRITICAL" in str(overall).upper() else PALETTE["warn"]
            st.markdown(
                f"""<div style='background:rgba(220,38,38,0.05);border:1px solid {border};border-radius:12px;padding:1rem;margin-bottom:1.0rem;'>
                <div style='font-weight:700;font-size:1rem;margin-bottom:0.5rem;'>🛡️ Shield Monitor — <span style='color:{border};'>{overall}</span></div>
                <p style='font-size:0.95rem; margin-bottom:0;'>Detected {monitor.get('alert_count',0)} anomalies requiring attention.</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.success("✅ **Shield Monitor:** No suspicious activity detected.", icon="🛡️")
    except Exception:
        pass

    st.divider()
    st.markdown("#### 💬 Security Chat")
    qa1, qa2 = st.columns(2)
    if qa1.button("🔍 Run Fraud Scan", use_container_width=True, key="sec_preset_1"):
        st.session_state["sec_input"] = "Scan my recent transactions for any signs of fraud."
    if qa2.button("🧯 Incident Playbook", use_container_width=True, key="sec_preset_2"):
        st.session_state["sec_input"] = "Provide an incident response playbook for suspicious card activity."

    user_q = st.text_area(
        "Request security intelligence",
        placeholder="e.g. 'Investigate recent high-value transactions and recommend immediate safeguards.'",
        height=110,
        key="sec_input",
    )
    if st.button("🚀 Execute Security Analysis", type="primary", key="sec_btn") and user_q:
        with st.spinner("Security Analyst is analyzing..."):
            try:
                sid = st.session_state.get("session_id") or st.session_state.setdefault("security_session_id", str(uuid.uuid4())[:12])
                resp = requests.post(f"{API_BASE_URL}/api/security/chat", json={"message": user_q, "session_id": sid}, timeout=120)
                if resp.status_code == 200:
                    res = resp.json()
                    st.markdown(res.get("reply", ""))
                else:
                    st.error(f"API Error: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")


def render_financial_ai_page():
    st.markdown("### 💰 Financial AI Advisor")
    st.caption("Premium spending insights, forecasting, optimization, and advisory reports.")
    _top_status_row(model_label="Financial Advisor")

    all_users = _get_all_users_financial()
    st.markdown("#### Context")
    selected_user = st.selectbox("Target user", all_users[:50], key="fin_user")

    # Quick summary
    try:
        from agents.financial_advisor_agent import FinancialAdvisorAgent
        adv = FinancialAdvisorAgent()
        summary = adv.tool_spending_summary(selected_user)
        st.info(
            f"📊 **Quick Review:** {summary.get('archetype','').replace('_',' ').title()} spender · "
            f"Monthly Avg: **${summary.get('avg_monthly_spend', 0):,.2f}**"
        )
    except Exception:
        pass

    st.divider()
    st.markdown("#### 💬 Advisor Chat")
    qa1, qa2, qa3 = st.columns(3)
    if qa1.button("📊 Spend Review", use_container_width=True, key="fin_preset_1"):
        st.session_state["fin_input"] = "Provide a full spending portfolio review."
    if qa2.button("💰 Savings Plan", use_container_width=True, key="fin_preset_2"):
        st.session_state["fin_input"] = "Generate a targeted savings strategy for my archetype."
    if qa3.button("📉 Cash Flow Forecast", use_container_width=True, key="fin_preset_3"):
        st.session_state["fin_input"] = "Forecast my cash flow for the next 30 days."

    user_q = st.text_area(
        "Request financial intelligence",
        placeholder="e.g. 'Where is my overspending, and what 3 changes will save the most this month?'",
        height=120,
        key="fin_input",
    )
    if st.button("🚀 Generate Advisor Report", type="primary", key="fin_btn") and user_q:
        with st.spinner("Financial Advisor is analyzing..."):
            try:
                payload = {"user_id": selected_user, "message": user_q}
                sid = st.session_state.get("session_id")
                if sid:
                    payload["session_id"] = sid
                resp = requests.post(f"{API_BASE_URL}/api/advisor/chat", json=payload, timeout=120)
                if resp.status_code == 200:
                    res = resp.json()
                    st.markdown(res.get("reply", ""))
                else:
                    st.error(f"API Error: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")


# ---------------------------------------------------------------------------
# Tab 6: Spending DNA — Financial Fingerprint
# ---------------------------------------------------------------------------
def render_dna_tab():
    st.markdown("### 🧬 Spending DNA — Financial Fingerprint")
    st.info("Every user has a unique 8-axis financial signature. This is used for identity verification and anomaly detection.")

    try:
        from agents.spending_dna_agent import SpendingDNAAgent
        dna_agent = SpendingDNAAgent()
        all_users = dna_agent.get_all_users()
    except Exception as e:
        st.error(f"Could not load DNA agent: {e}")
        return

    col_sel, col_trust = st.columns([2, 1])
    with col_sel:
        selected = st.selectbox("Select User for DNA Profile:", all_users[:30], key="dna_user")

    dna = dna_agent.compute_dna(selected)
    if "error" in dna:
        st.error(dna["error"])
        return

    with col_trust:
        score = dna["avg_trust_score"]
        color = "#16a34a" if score >= 0.8 else ("#d97706" if score >= 0.6 else "#dc2626")
        st.markdown(f"""
        <div class='metric-card' style='margin-top:1.5rem;'>
            <h3>Trust Score</h3>
            <div class='value' style='color:{color};'>{score:.0%}</div>
            <p style='font-size:0.8rem;color:#64748b;'>{dna['trust_grade']}</p>
        </div>
        """, unsafe_allow_html=True)

    col_radar, col_stats = st.columns([3, 2])
    with col_radar:
        # Radar chart
        labels = dna["radar_labels"]
        values = dna["radar_values"]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(79, 70, 229, 0.3)",
            line=dict(color="#4338ca", width=4),
            name=selected,
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=12, color="#0f172a", weight="bold")),
                angularaxis=dict(tickfont=dict(size=13, color="#0f172a", weight="bold")),
                bgcolor="rgba(248,250,252,1)",
            ),
            showlegend=False,
            title=dict(text=f"🧬 {selected} — Spending DNA", font=dict(size=18, color="#0f172a", weight="bold")),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=70, r=70, t=70, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.markdown("#### DNA Axes (Raw Values)")
        raw = dna["raw_axes"]
        axis_df = pd.DataFrame([
            {"Axis": label, "Raw Value": round(raw.get(col, 0), 3), "Normalized": round(val, 3)}
            for (col, label), val in zip(
                [("avg_txn_amount","Avg Txn Amount"),("location_entropy","Location Entropy"),
                 ("weekend_ratio","Weekend Ratio"),("category_diversity","Category Diversity"),
                 ("time_of_day_pref","Time of Day Pref"),("risk_appetite_score","Risk Appetite"),
                 ("spending_velocity","Spending Velocity"),("merchant_loyalty_score","Merchant Loyalty")],
                dna["radar_values"]
            )
        ])
        st.dataframe(axis_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Time Preference:** {dna['time_preference']}")
        st.markdown(f"**Anomalous Sessions:** {dna['anomalous_count']:,} / {dna['total_sessions']:,}")

    # Session comparison
    st.divider()
    st.markdown("#### 🔍 Session vs. DNA Comparison")
    if st.button("Simulate New Session", key="dna_compare"):
        comparison = dna_agent.compare_session(selected)
        verdict_color = "#16a34a" if "Trusted" in comparison["verdict"] else ("#d97706" if "Moderate" in comparison["verdict"] else "#dc2626")

        st.markdown(f"<div class='rag-answer' style='border-color:{verdict_color};'><strong>{comparison['verdict']}</strong><br>Session Trust Score: <strong>{comparison['session_trust_score']:.0%}</strong> | Composite Deviation: {comparison['composite_deviation']:.3f}</div>", unsafe_allow_html=True)

        # Overlay radar
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=comparison["baseline_radar"] + [comparison["baseline_radar"][0]],
            theta=comparison["radar_labels"] + [comparison["radar_labels"][0]],
            fill="toself", fillcolor="rgba(99,102,241,0.2)", line=dict(color="#6366f1", width=2, dash="dash"), name="DNA Baseline",
        ))
        fig2.add_trace(go.Scatterpolar(
            r=comparison["session_radar"] + [comparison["session_radar"][0]],
            theta=comparison["radar_labels"] + [comparison["radar_labels"][0]],
            fill="toself", fillcolor="rgba(220,38,38,0.15)", line=dict(color="#dc2626", width=3), name="Current Session",
        ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="rgba(248,250,252,0.8)"),
            showlegend=True, title="Session vs. DNA Baseline",
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=60, r=60, t=50, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 6: PDF Intelligence Chat
# ---------------------------------------------------------------------------
def render_pdf_tab():
    st.markdown("### 📄 PDF Intelligence & Document Chat")
    st.markdown("Upload documents to the local RAG engine and chat with them using private AI.")
    
    col_upload, col_chat = st.columns([1, 2])
    
    with col_upload:
        st.markdown("#### 📤 Upload Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader")
        
        if uploaded_files:
            if st.button("Index Documents", key="btn_index_pdf"):
                with st.spinner(f"Uploading and indexing {len(uploaded_files)} document(s)..."):
                    try:
                        # Prepare multiple files for the request
                        files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
                        resp = requests.post(f"{API_BASE_URL}/api/rag/upload", files=files_payload, timeout=120)
                        
                        if resp.status_code == 200:
                            data = resp.json()
                            uploads = data.get("uploads", [])
                            success_count = sum(1 for u in uploads if u["status"] == "indexed successfully")
                            st.success(f"Successfully indexed {success_count} file(s).")
                            
                            with st.expander("Upload Details"):
                                for u in uploads:
                                    status_emoji = "✅" if u["status"] == "indexed successfully" else "❌"
                                    st.write(f"{status_emoji} {u['filename']}: {u['status']}")
                                    if "error" in u:
                                        st.caption(f"Error: {u['error']}")
                        else:
                            st.error(f"Error: {resp.status_code} - {resp.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to API: {e}")
        
        st.divider()
        st.markdown("#### ℹ️ About PDF RAG")
        st.caption("Documents are processed locally. Text is chunked (1000 chars) and stored in ChromaDB.")

    with col_chat:
        st.markdown("#### 💬 Chat with Documents")
        
        # Chat history for this tab
        if "pdf_chat_history" not in st.session_state:
            st.session_state.pdf_chat_history = []
            
        # Display chat history
        chat_container = st.container(height=450)
        with chat_container:
            for msg in st.session_state.pdf_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg and msg["sources"]:
                        with st.expander("View Sources"):
                            for i, src in enumerate(msg["sources"]):
                                st.caption(f"Source {i+1}: {src['text'][:200]}...")
        
        # User input
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.pdf_chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing documents..."):
                        try:
                            resp = requests.post(
                                f"{API_BASE_URL}/api/rag/chat",
                                json={"message": prompt, "session_id": st.session_state.get("session_id")},
                                timeout=60
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                reply = data["reply"]
                                sources = data.get("sources", [])
                                st.markdown(reply)
                                if sources:
                                    with st.expander("View Sources"):
                                        for i, src in enumerate(sources):
                                            st.caption(f"Source {i+1}: {src['text'][:200]}...")
                                st.session_state.pdf_chat_history.append({"role": "assistant", "content": reply, "sources": sources})
                            else:
                                st.error(f"API Error: {resp.status_code}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            if st.button("Clear Chat", key="clear_pdf_chat"):
                st.session_state.pdf_chat_history = []
                st.rerun()



# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    # Require login before showing the main dashboard
    if not st.session_state.get("authenticated"):
        render_login_page()
        return

    st.markdown("""
    <div class="main-header">
        <h1>Veriscan Dashboard</h1>
        <p>Real-time Fraud Prevention &amp; Consumer Compliance Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    fraud_df = load_fraud_data(FEATURES_PATH, FEATURES_PATH.stat().st_mtime if FEATURES_PATH.exists() else 0)
    cfpb_df  = load_cfpb_data()
    page = st.session_state.get("nav", "🛡️ Security AI")
    if page == "🛡️ Security AI":
        render_security_ai_page()
    elif page == "💰 Financial AI":
        render_financial_ai_page()
    elif page == "📄 PDF Intelligence":
        render_pdf_tab()
    elif page == "📊 Market Dash":
        render_dashboard_tab(fraud_df)
    elif page == "🔍 CFPB Market Intel":
        render_cfpb_tab(cfpb_df)
    elif page == "🧬 Spending DNA":
        render_dna_tab()

if __name__ == "__main__":
    main()
