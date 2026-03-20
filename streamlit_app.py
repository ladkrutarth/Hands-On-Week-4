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
import re
import json
from agents.financial_advisor_agent import FinancialAdvisorAgent

# API Backend URL
API_BASE_URL = os.environ.get("VERISCAN_API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "financial_advisor_dataset.csv"
CFPB_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "cfpb_credit_card.csv"
IC3_DIR = PROJECT_ROOT / "dataset" / "ic3_2024_csvs"

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
        /* Vibrant gradient to make glassmorphism pop */
        background: linear-gradient(135deg, #E0E7FF 0%, #F8FAFC 50%, #F1F5F9 100%);
        background-attachment: fixed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #000000 !important;
        font-size: 16px;
        line-height: 1.6;
        font-weight: 600;
    }

    /* Clean Header - Glassmorphism */
    .main-header {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        padding: 2.5rem 3.5rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05) !important;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        color: #000000 !important;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #000000 !important;
        font-weight: 600;
        font-size: 0.98rem;
        margin-top: 0.5rem;
        max-width: 52rem;
    }

    /* Minimalist Bento Cards - Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: left;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .metric-card:hover {
        background: rgba(255, 255, 255, 0.5) !important;
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
    }

    .metric-card h3 { 
        margin: 0; 
        font-size: 0.9rem; 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
        color: #000000 !important; 
        font-weight: 800 !important; 
    }
    
    .metric-card .value { 
        font-size: 2.1rem; 
        font-weight: 900 !important; 
        margin: 0.5rem 0 0 0; 
        color: #000000 !important; 
        letter-spacing: -0.03em;
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
        color: #000000 !important;
        font-weight: 800 !important;
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

    /* Monochromatic Buttons - Primary state use Royal Blue instead of Black */
    div.stButton > button[kind="primary"] {
        background: #408EC6 !important;
        color: #FFFFFF !important;
        border: 1px solid #408EC6 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 6px -1px rgba(64, 142, 198, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #1E2761 !important;
        border-color: #1E2761 !important;
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

    /* RAG response box - Glassmorphism */
    .rag-answer {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        padding: 1.5rem;
        border-radius: 20px;
        font-size: 1rem;
        line-height: 1.6;
        color: #000000 !important;
        font-weight: 700 !important;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04) !important;
    }

    /* Force transparency on Streamlit containers for glass effect */
    div[data-testid="stVerticalBlock"] > div,
    div[data-testid="stHorizontalBlock"] > div {
        background-color: transparent !important;
    }
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
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #000000 !important;
        font-weight: 900 !important;
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
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Premium Navigation Selection (Pill Style) */
    .nav-button {
        display: flex;
        align-items: center;
        width: 100%;
        padding: 0.6rem 1rem;
        margin-bottom: 0.25rem;
        border-radius: 10px;
        background: transparent;
        border: 1px solid transparent;
        color: #000000 !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        text-align: left;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none !important;
    }

    .nav-button:hover {
        background: #F3F4F6;
        color: #000000 !important;
    }

    .nav-button.active {
        background: #408EC6 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 6px -1px rgba(64, 142, 198, 0.4);
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

@st.cache_data(ttl=3600)
def load_ic3_data():
    """Loads all IC3 2024 datasets into a dictionary of DataFrames."""
    datasets = {}
    if IC3_DIR.exists():
        for csv_file in IC3_DIR.glob("*.csv"):
            key = csv_file.stem
            datasets[key] = pd.read_csv(csv_file)
    return datasets

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
        nav_options = {
            "🛡️ Security AI": "🛡️ Security AI",
            "💰 Financial AI": "💰 Financial AI",
            "🧬 Multimodal": "🧬 Multimodal Intelligence",
            "📊 Market Dash": "📊 Market Dash",
            "🔍 CFPB Intel": "🔍 CFPB Market Intel",
            "🧬 Spending DNA": "🧬 Spending DNA"
        }
        
        current_nav = st.session_state.get("nav", "🛡️ Security AI")
        
        # Helper to set nav and rerun
        def set_nav(selection):
            st.session_state["nav"] = selection
            st.rerun()

        for label, selection in nav_options.items():
            is_active = (current_nav == selection)
            active_class = "active" if is_active else ""
            
            # Using custom CSS for button-like appearance that tracks state
            if st.button(
                label, 
                key=f"nav_{selection}", 
                width='stretch',
                type="primary" if is_active else "secondary"
            ):
                st.session_state["nav"] = selection
                st.rerun()

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

    ic3_data = load_ic3_data()
    
    # Create internal sub-tabs
    tab_nat, tab_trans = st.tabs(["🌐 National Trends", "💳 Transaction Intelligence"])
    
    with tab_nat:
        st.markdown("### 🌐 National Reported Trends (IC3 2024)")
        if not ic3_data:
            st.warning("IC3 2024 datasets not found.")
        else:
            # 1. Global Scam Stats
            st.subheader("📉 Top 10 Online Scam & Crime Types by Losses (Global)")
            from models.agent_tools_data import GLOBAL_SCAM_STATS
            scam_data = pd.DataFrame({
                "Scam Type": list(GLOBAL_SCAM_STATS.keys()),
                "Losses ($B)": list(GLOBAL_SCAM_STATS.values())
            }).sort_values(by="Losses ($B)", ascending=True)

            fig_scam = px.bar(
                scam_data, x="Losses ($B)", y="Scam Type", orientation='h',
                color="Losses ($B)", color_continuous_scale='Sunset',
                template='plotly_white', text="Losses ($B)"
            )
            fig_scam.update_traces(texttemplate='$%{text}B', textposition='outside')
            fig_scam.update_layout(showlegend=False, coloraxis_showscale=False, yaxis_title=None, height=450)
            st.plotly_chart(apply_accessible_theme(fig_scam), use_container_width=True)

            # 2. Age Group Pie
            st.divider()
            st.subheader("👥 Financial Loss Distribution by Age Group")
            if "03_age_group_breakdowns" in ic3_data:
                losses_row = ic3_data["03_age_group_breakdowns"][ic3_data["03_age_group_breakdowns"]["Metric"] == "Losses_USD"]
                if not losses_row.empty:
                    age_cols = ["Under_20", "Age_20_29", "Age_30_39", "Age_40_49", "Age_50_59", "Age_60_Plus"]
                    age_vals = losses_row[age_cols].iloc[0].values
                    fig_age = px.pie(
                        values=age_vals, names=age_cols,
                        title="Relative Financial Risk by Demographic",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    st.plotly_chart(fig_age, use_container_width=True)

            # 3. Top 14 States
            st.divider()
            st.subheader("🏛️ Top 14 States by Financial Loss")
            if "02_state_statistics" in ic3_data:
                state_df = ic3_data["02_state_statistics"].sort_values("Losses_USD", ascending=False).head(14)
                fig_state = px.bar(
                    state_df, x="State", y="Losses_USD",
                    title="Geographic Financial Impact (Top 14 States)",
                    labels={"Losses_USD": "Total Loss ($)"},
                    color="State", color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_state.update_layout(showlegend=False)
                st.plotly_chart(apply_accessible_theme(fig_state), use_container_width=True)

    with tab_trans:
        st.markdown("### 💳 Transaction Intelligence & Analytics")
        
        # Credit/Debit Selector (Requested)
        col_ctrl, _ = st.columns([1, 2])
        with col_ctrl:
            method_filter = st.selectbox("Transaction Method", ["All Transactions", "Credit Card", "Debit Card"], index=0)
            st.caption(f"Currently analyzing: **{method_filter}**")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='metric-card'><h3>Processed</h3><div class='value'>{len(df):,}</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><h3>Fraud Cases</h3><div class='value' style='color:#dc2626'>{df['is_fraud_flag'].sum():,}</div></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'><h3>Fraud Risk</h3><div class='value'>{(df['is_fraud_flag'].mean()*100):.1f}%</div></div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Fraud Heatmap by Category")
        cat_data = df.groupby('category')['is_fraud_flag'].sum().sort_values(ascending=False).reset_index()
        
        cat_icons = {
            "Groceries": "🍎 Groceries", "Dining & Restaurants": "🍔 Dining & Restaurants",
            "Online Shopping": "🛒 Online Shopping", "Healthcare": "💊 Healthcare",
            "Travel & Leisure": "✈️ Travel & Leisure", "Entertainment": "🎬 Entertainment",
            "Subscriptions": "📅 Subscriptions", "Housing (Rent/Mortgage)": "🏠 Housing",
            "Utilities & Bills": "🧾 Utilities & Bills", "Misc / Cash": "📦 Misc / Cash",
            "Transportation & Gas": "⛽ Transportation & Gas", "Clothing & Fashion": "🛍️ Clothing & Fashion",
            "Electronics": "💻 Electronics", "Education": "📚 Education", "Insurance": "🛡️ Insurance"
        }
        cat_data['category'] = cat_data['category'].apply(lambda x: cat_icons.get(x, f"❓ {str(x).title()}"))
        
        fig_cat = px.bar(cat_data, x='is_fraud_flag', y='category', orientation='h', color='is_fraud_flag', 
                         color_continuous_scale='Sunset', template='plotly_white')
        fig_cat.update_layout(height=600)
        st.plotly_chart(apply_accessible_theme(fig_cat), use_container_width=True)
        
        st.divider()
        st.subheader("📍 Interactive Fraud Bubble Map (US)")
        state_data = df.groupby("state")["is_fraud_flag"].sum().reset_index()
        state_data = state_data[state_data["state"].notna()]
        if not state_data.empty:
            min_cases = int(state_data["is_fraud_flag"].min())
            max_cases = int(state_data["is_fraud_flag"].max())
            cutoff = st.slider("Min Fraud Cases:", min_cases, max_cases, min(max_cases // 10, max_cases), key="fraud_map_cutoff")
            filtered = state_data[state_data["is_fraud_flag"] >= cutoff]
            fig_map = px.scatter_geo(
                filtered, locations="state", locationmode="USA-states",
                size="is_fraud_flag", color="is_fraud_flag", scope="usa",
                hover_name="state", labels={"is_fraud_flag": "Cases"},
                color_continuous_scale=SEQ_BLUE
            )
            fig_map.update_traces(marker_line_color="white", marker_line_width=0.6, opacity=0.9)
            st.plotly_chart(apply_accessible_theme(fig_map, title="Fraud cases by state (bubble map)"), use_container_width=True)
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
        
        if "cfpb_run_q" not in st.session_state:
            st.session_state.cfpb_run_q = False

        def set_cfpb_preset(q):
            st.session_state.cfpb_rag_q = q
            st.session_state.cfpb_run_q = True

        # Quick Presets
        st.markdown("<p style='font-size:0.8rem; font-weight:bold; margin-bottom:5px; color:#64748b;'>Quick Insights:</p>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        p1.button("📑 Billing Disputes", on_click=set_cfpb_preset, args=("What are the standard procedures and timelines for resolving billing disputes according to consumer complaints?",), width='stretch', key="p1")
        p2.button("💳 Lost/Stolen Issues", on_click=set_cfpb_preset, args=("Summarize the most common issues consumers face when reporting a lost or stolen credit card.",), width='stretch', key="p2")
        
        p3, p4 = st.columns(2)
        p3.button("🆔 Identity Theft Scenarios", on_click=set_cfpb_preset, args=("Search for 'identity theft'. What are the most frequent scenarios where consumers realize their identity was stolen?",), width='stretch', key="p3")
        p4.button("🚫 Unauthorized Discovery", on_click=set_cfpb_preset, args=("Find context on 'unauthorized transactions'. What evidence do consumers typically provide to prove they did not make a charge?",), width='stretch', key="p4")
        
        st.button("🎧 Service Dissatisfaction", on_click=set_cfpb_preset, args=("What are the primary reasons consumers express dissatisfaction with customer service in the credit card industry?",), width='stretch', key="p5")

        st.markdown("<p style='font-size:0.8rem; font-weight:bold; margin-top:15px; margin-bottom:5px; color:#64748b;'>Expert Intelligence Presets:</p>", unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        e1.button("📊 IC3 Fraud Trends", on_click=set_cfpb_preset, args=("What are the key findings and fraud trends from the 2024 IC3 Report?",), width='stretch', key="e1")
        e2.button("💸 Scam Economy Costs", on_click=set_cfpb_preset, args=("Analyze the global financial impact and 'true cost' of the scam economy based on recent reports.",), width='stretch', key="e2")
        
        e3, e4 = st.columns(2)
        e3.button("🎯 High-Loss Scams", on_click=set_cfpb_preset, args=("Identify which scam types resulted in the highest victim losses in 2024 according to the data.",), width='stretch', key="e3")
        e4.button("🏢 BEC Scam Targets", on_click=set_cfpb_preset, args=("Search for 'Business Email Compromise'. How do these scams typically target organizations?",), width='stretch', key="e4")

        st.divider()
        query = st.text_input("Search context:", placeholder="e.g. Find high-value travel anomalies...", key="cfpb_rag_q")
        
        if (st.button("Query Knowledge Base", key="cfpb_rag_btn") or st.session_state.cfpb_run_q) and query:
            st.session_state.cfpb_run_q = False # Reset trigger
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
                            for res in data["results"]:
                                # Extract fields from the RAGResult
                                txt = res.get("text", "")
                                meta = res.get("metadata", {})
                                conf = res.get("confidence", 0)
                                rtype = res.get("type", "unknown")
                                
                                # Visual Badges
                                c_color = "#16a34a" if conf > 0.8 else ("#d97706" if conf > 0.6 else "#dc2626")
                                badges = f'<span style="background:{c_color}20; color:{c_color}; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; margin-right:5px;">{conf:.0%} Match</span>'
                                
                                if rtype == "complaint":
                                    badges += f'<span style="background:#eff6ff; color:#1d4ed8; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; margin-right:5px;">🏢 {meta.get("company","")}</span>'
                                    badges += f'<span style="background:#fef2f2; color:#991b1b; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; margin-right:5px;">📍 {meta.get("state","")}</span>'
                                elif rtype in ["expert_qa", "scam_profile"]:
                                    badges += f'<span style="background:#f0fdf4; color:#166534; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; margin-right:5px;">🌟 Expert Intel</span>'
                                elif rtype == "pdf_doc":
                                    fname = meta.get("filename", "Document")
                                    badges += f'<span style="background:#fff7ed; color:#9a3412; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; margin-right:5px;">📄 PDF Source: {fname}</span>'
                                
                                st.markdown(f"""
                                <div class='rag-answer' style='border-left: 4px solid {c_color};'>
                                    <div style='margin-bottom:8px;'>{badges}</div>
                                    <div style='font-size:0.95rem;'>{txt}</div>
                                </div>
                                """, unsafe_allow_html=True)
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
        names_map = adv.get_user_names_map()
    except Exception:
        all_users = ["USER_0001"]
        names_map = {"USER_0001": "Sample User"}

    colX, colY = st.columns([1, 2])
    with colX:
        st.markdown("<p style='color: black; font-weight: bold; font-size: 14px; margin-bottom: 5px; margin-left: 2px;'>👤 Target Context:</p>", unsafe_allow_html=True)
        selected_user = st.selectbox(
            "", 
            all_users, 
            key="omni_user", 
            label_visibility="collapsed",
            format_func=lambda x: names_map.get(x, x)
        )
    
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
                            font={"family": "Outfit, sans-serif", "size": 13, "color": "#000000"},
                            coloraxis_showscale=False
                        )
                        fig.update_xaxes(title_font={"color": "#000000"}, tickfont={"color": "#000000"})
                        fig.update_yaxes(title_font={"color": "#000000"}, tickfont={"color": "#000000"})
                        fig.update_traces(textposition='outside', cliponaxis=False, textfont={"color": "#000000"})
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
    try:
        adv = FinancialAdvisorAgent()
        names_map = adv.get_user_names_map()
    except Exception:
        names_map = {}

    st.markdown("#### Context")
    selected_user = st.selectbox(
        "Target user", 
        all_users, 
        key="sec_user",
        format_func=lambda x: names_map.get(x, x)
    )

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
    qa1, qa2, qa3 = st.columns(3)
    if qa1.button("🔍 Run Fraud Scan", use_container_width=True, key="sec_preset_1"):
        st.session_state["sec_input"] = "Scan my recent transactions for any signs of fraud."
    if qa2.button("🧯 Incident Playbook", use_container_width=True, key="sec_preset_2"):
        st.session_state["sec_input"] = "Provide an incident response playbook for suspicious card activity."
    if qa3.button("🛡️ Risk Audit", use_container_width=True, key="sec_preset_3"):
        st.session_state["sec_input"] = "Perform a full security risk audit for my account profile."

    qa4, qa5, qa6 = st.columns(3)
    if qa4.button("📍 Access Monitor", use_container_width=True, key="sec_preset_4"):
        st.session_state["sec_input"] = "Review my recent login locations and flag any geographical anomalies."
    if qa5.button("💳 Card Security", use_container_width=True, key="sec_preset_5"):
        st.session_state["sec_input"] = "Check for any unauthorized card-not-present transaction patterns."
    if qa6.button("🌐 Network Health", use_container_width=True, key="sec_preset_6"):
        st.session_state["sec_input"] = "Analyze the network protocols and IP reputation of my recent sessions."

    qa7, qa8, qa9 = st.columns(3)
    if qa7.button("👤 Identity Check", use_container_width=True, key="sec_preset_7"):
        st.session_state["sec_input"] = "Verify if any of my personal identity data has been flagged in recent breaches."
    if qa8.button("📱 Device Trust", use_container_width=True, key="sec_preset_8"):
        st.session_state["sec_input"] = "Evaluate the trust score of the devices used to access my account."
    if qa9.button("🔒 Auth Audit", use_container_width=True, key="sec_preset_9"):
        st.session_state["sec_input"] = "Review my multi-factor authentication history for any suspicious bypass attempts."

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

    # Session Management Implementation
    if "fin_sessions" not in st.session_state:
        st.session_state.fin_sessions = {}
    if "active_fin_session_id" not in st.session_state:
        st.session_state.active_fin_session_id = None

    all_users = _get_all_users_financial()

    # Sidebar for session management (or top section)
    col_sessions, col_chat = st.columns([1, 3])

    with col_sessions:
        st.markdown("#### 📁 Sessions")
        if st.button("➕ New Advisor Chat", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.fin_sessions[new_id] = {
                "name": f"Chat {len(st.session_state.fin_sessions)+1}",
                "user_id": all_users[0],
                "history": []
            }
            st.session_state.active_fin_session_id = new_id
            st.rerun()

        if not st.session_state.fin_sessions:
            st.info("No active sessions. Create one to start.")
            return

        session_options = {sid: s["name"] for sid, s in st.session_state.fin_sessions.items()}
        selected_sid = st.radio("Select Session", options=list(session_options.keys()), 
                                 format_func=lambda x: session_options[x],
                                 key="session_selector")
        st.session_state.active_fin_session_id = selected_sid
        
        active_sess = st.session_state.fin_sessions[selected_sid]
        
        st.divider()
        st.markdown("#### ⚙️ Settings")
        try:
            adv = FinancialAdvisorAgent()
            names_map = adv.get_user_names_map()
        except Exception:
            names_map = {}

        new_user = st.selectbox("Target user", all_users, 
                                index=all_users.index(active_sess["user_id"]) if active_sess["user_id"] in all_users else 0,
                                key="fin_user_select",
                                format_func=lambda x: names_map.get(x, x))
        if new_user != active_sess["user_id"]:
            # Context clearing logic
            active_sess["user_id"] = new_user
            active_sess["history"] = []
            try:
                requests.post(f"{API_BASE_URL}/api/advisor/reset?session_id={selected_sid}", timeout=5)
            except Exception:
                pass
            st.rerun()
        
        if st.button("🗑️ Delete Session", type="secondary"):
            del st.session_state.fin_sessions[selected_sid]
            st.session_state.active_fin_session_id = None
            st.rerun()

    with col_chat:
        active_sess = st.session_state.fin_sessions[st.session_state.active_fin_session_id]
        selected_user = active_sess["user_id"]

        # Quick summary header
        try:
            adv = FinancialAdvisorAgent()
            summary = adv.tool_spending_summary(selected_user)
            user_name = names_map.get(selected_user, selected_user)
            st.markdown(f"👤 **User:** {user_name} | **Context:** {summary.get('archetype','').replace('_',' ').title()} spender · Avg: **${summary.get('avg_monthly_spend', 0):,.2f}/mo**")
        except Exception:
            pass

        st.divider()

        # Chat display container
        chat_container = st.container(height=500)
        with chat_container:
            if not active_sess["history"]:
                st.info("Ask a financial question or use a preset below.")
            
            for msg in active_sess["history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Input section: Row 1
        qa1, qa2, qa3 = st.columns(3)
        preset_q = None
        if qa1.button("📊 Spend Review", use_container_width=True, key="fin_preset_1"):
            preset_q = "Provide a full spending portfolio review."
        if qa2.button("💰 Savings Plan", use_container_width=True, key="fin_preset_2"):
            preset_q = "Generate a targeted savings strategy for my archetype."
        if qa3.button("📉 Cash Flow Forecast", use_container_width=True, key="fin_preset_3"):
            preset_q = "Forecast my cash flow for the next 30 days."

        # Input section: Row 2
        qb1, qb2, qb3 = st.columns(3)
        if qb1.button("🧾 Tax-Deductible Finder", use_container_width=True, key="fin_preset_4"):
            preset_q = "Find my potential tax-deductible expenses."
        if qb2.button("📈 Price Hike Alerts", use_container_width=True, key="fin_preset_5"):
            preset_q = "Scan my subscriptions for price hikes."
        if qb3.button("💳 Credit Score Impact", use_container_width=True, key="fin_preset_6"):
            preset_q = "Analyze my spending impact on my credit score."

        # Input section: Row 3
        qc1, qc2, qc3 = st.columns(3)
        if qc1.button("🛡️ Fraud Market Scan", use_container_width=True, key="fin_preset_7"):
            preset_q = "How do global market fraud trends compare to this dataset?"
        if qc2.button("📈 Surplus Optimizer", use_container_width=True, key="fin_preset_8"):
            preset_q = "Identify areas where I can optimize surplus funds."
        if qc3.button("💸 Liquidity Guard", use_container_width=True, key="fin_preset_9"):
            preset_q = "Review my upcoming bills and liquidity status."

        user_q = st.chat_input("Request financial intelligence...")
        
        final_q = user_q or preset_q
        
        if final_q:
            # Handle user message
            active_sess["history"].append({"role": "user", "content": final_q})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(final_q)
                
                with st.chat_message("assistant"):
                    with st.spinner("Financial Advisor is analyzing..."):
                        try:
                            payload = {
                                "user_id": selected_user, 
                                "message": final_q,
                                "session_id": st.session_state.active_fin_session_id
                            }
                            resp = requests.post(f"{API_BASE_URL}/api/advisor/chat", json=payload, timeout=120)
                            if resp.status_code == 200:
                                res = resp.json()
                                reply = res.get("reply", "")
                                st.markdown(reply)
                                active_sess["history"].append({"role": "assistant", "content": reply})
                            else:
                                st.error(f"API Error: {resp.status_code}")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 6: Spending DNA — Financial Fingerprint
# ---------------------------------------------------------------------------
def render_dna_tab():
    st.markdown("### 🧬 Spending DNA — Financial Fingerprint")
    st.info("Every user has a unique 8-axis financial signature. This is used for identity verification and anomaly detection.")

    try:
        from agents.spending_dna_agent import SpendingDNAAgent
        dna_agent = SpendingDNAAgent()
        adv = FinancialAdvisorAgent()
        all_users = dna_agent.get_all_users()
        names_map = adv.get_user_names_map()
    except Exception as e:
        st.error(f"Could not load DNA agent: {e}")
        return

    col_sel, col_trust = st.columns([2, 1])
    with col_sel:
        selected = st.selectbox(
            "Select User for DNA Profile:", 
            all_users, 
            key="dna_user",
            format_func=lambda x: names_map.get(x, x)
        )

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
            line={"color": "#4338ca", "width": 4},
            name=selected,
        ))
        fig.update_layout(
            polar={
                "radialaxis": {"visible": True, "range": [0, 1], "tickfont": {"size": 12, "color": "#0f172a", "weight": "bold"}},
                "angularaxis": {"tickfont": {"size": 13, "color": "#0f172a", "weight": "bold"}},
                "bgcolor": "rgba(248,250,252,1)",
            },
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

    # Yearly & Monthly Evolution
    st.divider()
    st.markdown("#### 📅 Historical DNA Analysis")
    
    col_yearly, col_monthly = st.columns(2)
    
    with col_yearly:
        st.markdown("**Year-over-Year DNA Comparison**")
        dna_2024 = dna_agent.compute_yearly_dna(selected, 2024)
        dna_2025 = dna_agent.compute_yearly_dna(selected, 2025)
        
        if "error" not in dna_2024 and "error" not in dna_2025:
            labels = dna["radar_labels"]
            fig_yoy = go.Figure()
            fig_yoy.add_trace(go.Scatterpolar(
                r=dna_2024["radar_values"] + [dna_2024["radar_values"][0]],
                theta=labels + [labels[0]],
                fill="toself", name="2024 DNA", line={"color": "#94a3b8"}
            ))
            fig_yoy.add_trace(go.Scatterpolar(
                r=dna_2025["radar_values"] + [dna_2025["radar_values"][0]],
                theta=labels + [labels[0]],
                fill="toself", name="2025 DNA", line={"color": "#4f46e5", "width": 3}
            ))
            fig_yoy.update_layout(
                polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                height=400, margin={"l": 40, "r": 40, "t": 30, "b": 30}, showlegend=True
            )
            st.plotly_chart(fig_yoy, use_container_width=True)
        else:
            st.warning("Insufficient historical data for year-over-year comparison.")

    with col_monthly:
        st.markdown("**Monthly DNA Trust Evolution**")
        evolution = dna_agent.compute_monthly_evolution(selected)
        if "error" not in evolution:
            evol_df = pd.DataFrame({
                "Month": evolution["labels"],
                "Trust Score": evolution["trust_scores"],
                "Deviation": evolution["deviations"]
            })
            fig_evol = px.line(evol_df, x="Month", y=["Trust Score", "Deviation"],
                              color_discrete_sequence=["#16a34a", "#dc2626"],
                              markers=True)
            fig_evol.update_layout(height=400, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                                 legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1})
            st.plotly_chart(fig_evol, use_container_width=True)
        else:
            st.warning("No monthly trend data available.")

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
            fill="toself", fillcolor="rgba(99,102,241,0.2)", line={"color": "#6366f1", "width": 2, "dash": "dash"}, name="DNA Baseline",
        ))
        fig2.add_trace(go.Scatterpolar(
            r=comparison["session_radar"] + [comparison["session_radar"][0]],
            theta=comparison["radar_labels"] + [comparison["radar_labels"][0]],
            fill="toself", fillcolor="rgba(220,38,38,0.15)", line={"color": "#dc2626", "width": 3}, name="Current Session",
        ))
        fig2.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}, "bgcolor": "rgba(248,250,252,0.8)"},
            showlegend=True, title="Session vs. DNA Baseline",
            paper_bgcolor="rgba(0,0,0,0)", margin={"l": 60, "r": 60, "t": 50, "b": 30},
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 6: Multimodal Intelligence (Direct Local Processing)
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_multimodal_rag():
    from models.multimodal_rag import MultimodalRAG
    return MultimodalRAG()

def _generate_csv_chart(df, query=""):
    """Smart chart generation from a DataFrame based on query keywords."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = []
    
    # Detect date columns
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "time", "timestamp", "period", "day", "month", "year"]):
            date_cols.append(c)
        elif c not in num_cols:
            try:
                if pd.to_datetime(df[c].head(3), errors='coerce').notnull().all():
                    date_cols.append(c)
            except:
                continue
    
    query_lower = query.lower()
    
    # Determine chart type from query
    if any(k in query_lower for k in ["pie", "proportion", "percentage", "share", "distribution"]):
        chart_type = "pie"
    elif any(k in query_lower for k in ["line", "trend", "over time", "timeline"]):
        chart_type = "line"
    elif any(k in query_lower for k in ["scatter", "correlation", "relationship"]):
        chart_type = "scatter"
    elif any(k in query_lower for k in ["histogram", "frequency"]):
        chart_type = "histogram"
    else:
        chart_type = "bar"
    
    fig = None
    title = ""
    
    if chart_type == "pie" and cat_cols and num_cols:
        fig = px.pie(df.head(20), names=cat_cols[0], values=num_cols[0], title=f"{num_cols[0]} by {cat_cols[0]}")
        title = f"Pie chart: {num_cols[0]} by {cat_cols[0]}"
    elif chart_type == "line" and date_cols and num_cols:
        df_sorted = df.copy()
        df_sorted[date_cols[0]] = pd.to_datetime(df_sorted[date_cols[0]], errors='coerce')
        df_sorted = df_sorted.dropna(subset=[date_cols[0]]).sort_values(date_cols[0])
        fig = px.line(df_sorted, x=date_cols[0], y=num_cols[0], title=f"{num_cols[0]} over Time", markers=True)
        title = f"Line chart: {num_cols[0]} over {date_cols[0]}"
    elif chart_type == "scatter" and len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title=f"{num_cols[1]} vs {num_cols[0]}")
        title = f"Scatter: {num_cols[1]} vs {num_cols[0]}"
    elif chart_type == "histogram" and num_cols:
        fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
        title = f"Histogram: {num_cols[0]}"
    elif date_cols and num_cols:
        df_sorted = df.copy()
        df_sorted[date_cols[0]] = pd.to_datetime(df_sorted[date_cols[0]], errors='coerce')
        df_sorted = df_sorted.dropna(subset=[date_cols[0]]).sort_values(date_cols[0])
        fig = px.line(df_sorted, x=date_cols[0], y=num_cols[0], title=f"{num_cols[0]} over Time", markers=True)
        title = f"Line chart: {num_cols[0]} over {date_cols[0]}"
    elif cat_cols and num_cols:
        fig = px.bar(df.head(25), x=cat_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {cat_cols[0]}")
        title = f"Bar chart: {num_cols[0]} by {cat_cols[0]}"
    elif num_cols:
        fig = px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
        title = f"Histogram: {num_cols[0]}"
    
    if fig:
        apply_accessible_theme(fig)
        fig.update_layout(height=400)
    
    return fig, title

def render_multimodal_tab():
    # Ensure session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:12]
    if "pdf_chat_history" not in st.session_state:
        st.session_state.pdf_chat_history = []
    if "mm_uploaded_files" not in st.session_state:
        st.session_state.mm_uploaded_files = {}  # filename -> {"bytes": bytes, "type": str, "df": optional}
    if "mm_indexed" not in st.session_state:
        st.session_state.mm_indexed = False
        
    st.markdown("### 🧬 Multimodal Intelligence & Evidence Analysis")
    st.markdown("Upload PDFs, Images, and CSVs to analyze evidence using local RAG and AI.")
    
    session_id = st.session_state.session_id
    
    col_upload, col_chat = st.columns([1, 2])
    
    with col_upload:
        st.markdown("#### 📤 Upload Evidence")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, PNG, JPG, CSV, TXT)", 
            type=["pdf", "png", "jpg", "jpeg", "csv", "txt", "json"], 
            accept_multiple_files=True, 
            key="multimodal_uploader"
        )
        
        # Process uploaded files
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.mm_uploaded_files:
                    file_bytes = f.getvalue()
                    file_info = {"bytes": file_bytes, "type": f.type or "unknown", "name": f.name}
                    # Pre-load CSV DataFrames
                    if f.name.lower().endswith(".csv"):
                        try:
                            import io
                            file_info["df"] = pd.read_csv(io.BytesIO(file_bytes))
                        except:
                            pass
                    st.session_state.mm_uploaded_files[f.name] = file_info
            
            if st.button("📥 Index All Documents", key="btn_index_mm", type="primary"):
                rag = _get_multimodal_rag()
                with st.spinner(f"Indexing {len(st.session_state.mm_uploaded_files)} file(s)..."):
                    results = []
                    for fname, finfo in st.session_state.mm_uploaded_files.items():
                        result = rag.index_file_bytes(fname, finfo["bytes"], session_id=session_id)
                        results.append(result)
                    
                    success = sum(1 for r in results if r.get("status") == "indexed")
                    st.success(f"✅ Indexed {success}/{len(results)} files successfully!")
                    st.session_state.mm_indexed = True
                    
                    with st.expander("Index Details", expanded=False):
                        for r in results:
                            emoji = "✅" if r.get("status") == "indexed" else "❌"
                            detail = ""
                            if "pages" in r: detail = f" ({r['pages']} pages, {r.get('chunks', 0)} chunks)"
                            elif "rows" in r: detail = f" ({r['rows']} rows, {len(r.get('columns', []))} columns)"
                            elif "ocr_text" in r: detail = f" (OCR extracted)"
                            st.write(f"{emoji} **{r['filename']}**: {r['status']}{detail}")
        
        # Show indexed file inventory
        st.divider()
        st.markdown("#### 📂 Loaded Files")
        if st.session_state.mm_uploaded_files:
            for fname, finfo in st.session_state.mm_uploaded_files.items():
                ext = fname.split(".")[-1].upper()
                emoji = {"PDF": "📕", "CSV": "📊", "PNG": "🖼️", "JPG": "🖼️", "JPEG": "🖼️", "TXT": "📝", "JSON": "📋"}.get(ext, "📄")
                size_kb = len(finfo["bytes"]) / 1024
                extra = ""
                if "df" in finfo:
                    df = finfo["df"]
                    extra = f" · {len(df)} rows × {len(df.columns)} cols"
                st.markdown(f"{emoji} **{fname}** ({size_kb:.1f} KB){extra}")
        else:
            st.caption("No files uploaded yet.")

        # CSV Quick Visualizations
        csv_files = {k: v for k, v in st.session_state.mm_uploaded_files.items() if "df" in v}
        if csv_files:
            st.divider()
            st.markdown("#### 📊 Quick Data Preview")
            for fname, finfo in csv_files.items():
                df = finfo["df"]
                with st.expander(f"📊 {fname}", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    fig, title = _generate_csv_chart(df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.markdown("#### ℹ️ About Multimodal RAG")
        st.caption("All data is processed locally. PDFs are text-extracted. CSVs are analyzed structurally. Images use OCR when available.")

    with col_chat:
        chat_header_col1, chat_header_col2 = st.columns([4, 1])
        with chat_header_col1:
            st.markdown("#### 💬 Chat with Documents")
        with chat_header_col2:
            if st.button("🧹 Clear", key="clear_pdf_chat_top", help="Clear chat history"):
                st.session_state.pdf_chat_history = []
                st.rerun()
        
        # Show context indicator
        if st.session_state.mm_uploaded_files:
            file_names = list(st.session_state.mm_uploaded_files.keys())
            indexed_status = "✅ Indexed" if st.session_state.mm_indexed else "⚠️ Not indexed yet — click 'Index All Documents'"
            st.info(f"📎 **{len(file_names)} file(s) loaded**: {', '.join(file_names)} | {indexed_status}")
        else:
            st.info("Upload files on the left panel, then ask questions about them here.")
            
        # Display chat history
        chat_container = st.container(height=450)
        with chat_container:
            for msg in st.session_state.pdf_chat_history:
                with st.chat_message(msg["role"]):
                    text = msg["content"]
                    st.markdown(text)
                    # Render chart if attached
                    if "chart" in msg and msg["chart"] is not None:
                        st.plotly_chart(msg["chart"], use_container_width=True)
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
                        chart_fig = None
                        sources = []
                        reply = ""
                        
                        # Check if this is a chart/graph request for CSV data
                        is_chart_query = any(k in prompt.lower() for k in [
                            "chart", "graph", "plot", "visualize", "show", "draw",
                            "histogram", "scatter", "pie", "bar", "line", "trend"
                        ])
                        
                        csv_data = {k: v for k, v in st.session_state.mm_uploaded_files.items() if "df" in v}
                        
                        if is_chart_query and csv_data:
                            # Direct CSV charting — generate from actual data
                            for fname, finfo in csv_data.items():
                                df = finfo["df"]
                                chart_fig, chart_title = _generate_csv_chart(df, query=prompt)
                                if chart_fig:
                                    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                                    reply = (
                                        f"📊 **Chart generated from {fname}**\n\n"
                                        f"**Dataset:** {len(df)} rows × {len(df.columns)} columns\n\n"
                                        f"**Columns:** {', '.join(df.columns.tolist())}\n\n"
                                        f"**Numeric:** {', '.join(num_cols)}\n\n"
                                    )
                                    # Add basic stats
                                    if num_cols:
                                        stats = df[num_cols].describe().round(2)
                                        reply += f"**Quick Stats:**\n"
                                        for col in num_cols[:3]:
                                            reply += f"- {col}: mean={stats.loc['mean', col]:.2f}, min={stats.loc['min', col]:.2f}, max={stats.loc['max', col]:.2f}\n"
                                    break
                            if not reply:
                                reply = "Could not generate a chart from the available CSV data. Please check that your CSV has numeric columns."
                        
                        # RAG-based answer (for non-chart queries, or in addition to chart)
                        if not is_chart_query or not chart_fig:
                            try:
                                # Try API first
                                payload = {
                                    "message": prompt,
                                    "session_id": session_id,
                                    "images": [],
                                    "file_types": ["pdf_doc", "image_doc", "csv_doc", "text_doc", "pdf_summary", "csv_summary"]
                                }
                                resp = requests.post(f"{API_BASE_URL}/api/rag/chat", json=payload, timeout=15)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    reply = data["reply"]
                                    sources = data.get("sources", [])
                                else:
                                    raise Exception(f"API returned {resp.status_code}")
                            except Exception:
                                # Fallback: Direct local RAG query
                                try:
                                    rag = _get_multimodal_rag()
                                    results = rag.query(prompt, n_results=10, session_id=session_id)
                                    
                                    if results:
                                        sources = [{"text": r["text"], "metadata": r.get("metadata", {})} for r in results]
                                        reply = f"📄 **Based on your uploaded documents:**\n\n"
                                        
                                        # Group by file
                                        file_contexts = {}
                                        for r in results:
                                            fname = r.get("metadata", {}).get("filename", "Unknown")
                                            if fname not in file_contexts:
                                                file_contexts[fname] = []
                                            file_contexts[fname].append(r["text"])
                                        
                                        for fname, texts in file_contexts.items():
                                            reply += f"**From {fname}:**\n"
                                            for t in texts[:3]:
                                                clean = t.replace(f"[From PDF: {fname}]", "").replace(f"[From CSV: {fname}]", "").strip()
                                                if len(clean) > 500:
                                                    clean = clean[:500] + "..."
                                                reply += f"> {clean}\n\n"
                                    else:
                                        reply = "No relevant content found in your uploaded documents. Please ensure you've uploaded and indexed your files first."
                                except Exception as e:
                                    reply = f"⚠️ Could not query documents: {str(e)}. Ensure files are uploaded and indexed."
                        
                        # Render the response
                        st.markdown(reply)
                        if chart_fig:
                            st.plotly_chart(chart_fig, use_container_width=True)
                        if sources:
                            with st.expander("View Sources"):
                                for i, src in enumerate(sources):
                                    st.caption(f"Source {i+1}: {src['text'][:200]}...")
                        
                        # Save to history
                        st.session_state.pdf_chat_history.append({
                            "role": "assistant", 
                            "content": reply, 
                            "sources": sources,
                            "chart": chart_fig
                        })


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
    elif page == "🧬 Multimodal Intelligence":
        render_multimodal_tab()
    elif page == "📊 Market Dash":
        render_dashboard_tab(fraud_df)
    elif page == "🔍 CFPB Market Intel":
        render_cfpb_tab(cfpb_df)
    elif page == "🧬 Spending DNA":
        render_dna_tab()

if __name__ == "__main__":
    main()
