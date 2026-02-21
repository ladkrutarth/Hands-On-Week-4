"""
GraphGuard — Gemini-Powered Dynamic Question Generator
Generates adaptive, multiple-choice security questions using Google Gemini LLM + RAG.
Falls back to static multiple-choice questions when no API key is available.
"""

import json
import os
import random
from typing import List, Optional

# ---------------------------------------------------------------------------
# Fallback static multiple-choice questions by security level
# Questions focus on stores, locations, categories, spending patterns.
# NO risk scores or internal system metrics.
# ---------------------------------------------------------------------------

STATIC_QUESTIONS = {
    "LOW": [
        {
            "question": "Which type of store did you shop at most frequently in the last 5 days?",
            "options": ["Grocery store", "Electronics store", "Clothing store", "Restaurant"],
            "correct_answer": "Grocery store",
            "difficulty": "easy",
        },
        {
            "question": "In which city was your most recent purchase?",
            "options": ["New York, NY", "Chicago, IL", "Houston, TX", "Seattle, WA"],
            "correct_answer": "New York, NY",
            "difficulty": "easy",
        },
        {
            "question": "What is the approximate average amount you spend per transaction?",
            "options": ["Under $50", "$50 – $150", "$150 – $300", "Over $300"],
            "correct_answer": "$50 – $150",
            "difficulty": "easy",
        },
    ],
    "MEDIUM": [
        {
            "question": "On which day of the week do you usually shop the most?",
            "options": ["Monday", "Wednesday", "Friday", "Saturday"],
            "correct_answer": "Friday",
            "difficulty": "medium",
        },
        {
            "question": "Did you make any purchase at a Gas Station in the last 5 days?",
            "options": ["Yes, once", "Yes, multiple times", "No, I did not", "I don't remember"],
            "correct_answer": "No, I did not",
            "difficulty": "medium",
        },
        {
            "question": "Have you shopped at a Jewelry store recently?",
            "options": ["Yes, in the last 5 days", "Yes, but more than a week ago", "No, never", "Yes, today"],
            "correct_answer": "Yes, in the last 5 days",
            "difficulty": "medium",
        },
    ],
    "HIGH": [
        {
            "question": "In which city did you make your largest purchase in the last 5 days?",
            "options": ["Miami, FL", "Phoenix, AZ", "Boston, MA", "Denver, CO"],
            "correct_answer": "Phoenix, AZ",
            "difficulty": "hard",
        },
        {
            "question": "What was the approximate total amount you spent in the last 5 days?",
            "options": ["Under $100", "$100 – $500", "$500 – $1,000", "Over $1,000"],
            "correct_answer": "$500 – $1,000",
            "difficulty": "hard",
        },
        {
            "question": "How many different store categories did you shop at in the last 5 days?",
            "options": ["Just 1", "2 to 3", "4 to 5", "More than 5"],
            "correct_answer": "2 to 3",
            "difficulty": "hard",
        },
    ],
    "CRITICAL": [
        {
            "question": "What was the exact amount of your most recent purchase at an Electronics store?",
            "options": ["$45.99", "$102.95", "$225.44", "$89.50"],
            "correct_answer": "$102.95",
            "difficulty": "very_hard",
        },
        {
            "question": "Which city did you shop in that you had NOT visited before recently?",
            "options": ["Seattle, WA", "Miami, FL", "Phoenix, AZ", "Chicago, IL"],
            "correct_answer": "Seattle, WA",
            "difficulty": "very_hard",
        },
        {
            "question": "At what time of day did you make your most recent purchase?",
            "options": ["Morning (6am–12pm)", "Afternoon (12pm–6pm)", "Evening (6pm–12am)", "Late Night (12am–6am)"],
            "correct_answer": "Afternoon (12pm–6pm)",
            "difficulty": "very_hard",
        },
    ],
}


class GeminiQuestionGenerator:
    """
    Generates dynamic security questions using Google Gemini LLM.
    Uses RAG context to personalize multiple-choice questions to the user's data.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model = None
        self._setup_gemini()

    def _setup_gemini(self):
        """Initialize Gemini model if API key is available."""
        if not self.api_key:
            print("ℹ️  No GOOGLE_API_KEY found. Using static fallback questions.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self.model = "gemini-2.0-flash"
            print("✅ Gemini question generator initialized")
        except Exception as e:
            print(f"⚠️  Gemini initialization failed ({e}). Using fallback.")
            self.model = None

    def generate_questions(
        self,
        user_id: str,
        security_level: str,
        num_questions: int,
        rag_context: str = "",
        user_profile: Optional[dict] = None,
    ) -> List[dict]:
        """
        Generate multiple-choice security questions for authentication.
        """
        # Hard cap at 3 questions as requested
        num_questions = min(num_questions, 3)

        if self.model and rag_context:
            try:
                return self._generate_with_gemini(
                    user_id, security_level, num_questions, rag_context, user_profile
                )
            except Exception as e:
                print(f"⚠️  Gemini generation failed ({e}). Falling back to static.")

        return self._get_static_questions(security_level, num_questions)

    def _generate_with_gemini(
        self,
        user_id: str,
        security_level: str,
        num_questions: int,
        rag_context: str,
        user_profile: Optional[dict],
    ) -> List[dict]:
        """Generate meaningful multiple-choice questions about stores, locations, and categories using Gemini."""
        difficulty_map = {
            "LOW": "easy — ask about general shopping habits and favorite store types",
            "MEDIUM": "moderate — ask about specific store locations, categories, and recent days",
            "HIGH": "difficult — ask about specific purchases, exact cities, and amounts from the last 5 days",
            "CRITICAL": "very difficult — ask about exact transaction details, specific store visits, and purchase amounts",
        }
        difficulty = difficulty_map.get(security_level, "moderate")

        prompt = f"""You are a bank security system verifying a customer's identity through their recent shopping activity.
Generate exactly {num_questions} multiple-choice questions for customer {user_id}.

Difficulty: {difficulty}

Recent transaction data for this customer:
{rag_context}

IMPORTANT RULES — READ CAREFULLY:
1. Ask ONLY about real-world shopping activity: store types, locations (cities), categories, amounts spent, days of the week, time of day.
2. NEVER ask about risk scores, z-scores, security levels, fraud flags, or any internal system metrics.
3. Questions should feel natural — like a bank calling to verify recent purchases.
4. Focus on the LAST 5 DAYS of transactions whenever possible.
5. Provide exactly 4 plausible options. EXACTLY ONE must be correct based on the data. The other three must be realistic but wrong.
6. Mix question types across these categories:
   - "Which store/category did you shop at?" (e.g., Jewelry, Electronics, Grocery, Restaurants)
   - "In which city was your recent purchase?" (e.g., Miami, FL; Seattle, WA)
   - "What was the approximate amount of your purchase at [store type]?"
   - "On which day did you make a purchase at [location]?"
   - "Did you visit a [store type] in the last 5 days?"

Example question styles:
- "In which city did you make your most recent purchase?"
- "Which type of store did you visit most in the last 5 days?"
- "What was the approximate amount you spent at a Jewelry store recently?"
- "Did you shop at a Gas Station in the last few days?"

Return ONLY a JSON array (no markdown, no code fences) with exactly {num_questions} objects:
[
  {{
    "question": "In which city did you make your most recent Jewelry purchase?",
    "options": ["Seattle, WA", "Chicago, IL", "Miami, FL", "Denver, CO"],
    "correct_answer": "Miami, FL",
    "difficulty": "medium"
  }}
]
"""

        response = self._client.models.generate_content(
            model=self.model, contents=prompt
        )
        text = response.text.strip()

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        questions = json.loads(text)

        validated = []
        for q in questions[:num_questions]:
            options = q.get("options", [])
            # Shuffle options to avoid correct answer bias
            random.shuffle(options)
            validated.append({
                "question": q.get("question", "Verify your identity."),
                "options": options,
                "correct_answer": q.get("correct_answer", options[0] if options else ""),
                "difficulty": q.get("difficulty", "medium"),
            })
        return validated

    def _get_static_questions(self, security_level: str, num_questions: int) -> List[dict]:
        """Return a requested number of fallback multiple-choice questions."""
        pool = STATIC_QUESTIONS.get(security_level, STATIC_QUESTIONS["MEDIUM"])
        if num_questions > len(pool):
            num_questions = len(pool)
        
        selected = random.sample(pool, num_questions)
        for s in selected:
            # Shuffle options for static questions too
            random.shuffle(s["options"])
        return selected

    def _generate_fallback_analysis(self, txn: dict) -> str:
        """Generate a structured, rule-based fraud analysis when the LLM is unavailable."""
        amt = txn.get('AMOUNT', 0)
        user_avg = txn.get('USER_AVG_AMOUNT', 0)
        cat = txn.get('CATEGORY', 'Unknown')
        loc = txn.get('LOCATION', 'Unknown')
        is_new_loc = txn.get('IS_NEW_LOCATION', False)
        
        zscore_flag = txn.get('ZSCORE_FLAG', 0)
        vel_flag = txn.get('VELOCITY_FLAG', 0)
        cat_score = txn.get('CATEGORY_RISK_SCORE', 0)
        iso_score = txn.get('ISOLATION_FOREST_SCORE', 0)
        risk_level = txn.get('RISK_LEVEL', 'UNKNOWN')
        combined = txn.get('COMBINED_RISK_SCORE', 0)
        
        # Count active risk factors for severity
        risk_factors = []
        severity_count = 0
        if is_new_loc:
            risk_factors.append(f"- 🌍 **Geographic Anomaly** (severity: HIGH) — Transaction in **{loc}**, a new/unusual location for this account.")
            severity_count += 1
        if zscore_flag > 0.5:
            deviation = ((amt - user_avg) / user_avg * 100) if user_avg > 0 else 0
            risk_factors.append(f"- 💰 **Unusual Amount** (severity: {'HIGH' if zscore_flag > 0.7 else 'MEDIUM'}) — **${amt:.2f}** is {deviation:.0f}% above the user's average of ${user_avg:.2f}.")
            severity_count += 1
        if vel_flag > 0.5:
            risk_factors.append(f"- ⚡ **High Velocity** (severity: {'HIGH' if vel_flag > 0.7 else 'MEDIUM'}) — Unusual transaction frequency detected in recent 1h/24h window.")
            severity_count += 1
        if cat_score > 0.5:
            risk_factors.append(f"- 🏷️ **High-Risk Category** (severity: MEDIUM) — **{cat}** is statistically associated with elevated fraud rates.")
            severity_count += 1
        if iso_score > 0.6:
            risk_factors.append(f"- 🤖 **ML Anomaly Detection** (severity: {'HIGH' if iso_score > 0.8 else 'MEDIUM'}) — Isolation Forest detected multivariate anomalies across transaction features.")
            severity_count += 1
            
        if not risk_factors:
            risk_factors.append("- ℹ️ **Minor Deviation** (severity: LOW) — Slight deviation from normal behavior; standard monitoring applies.")

        risk_factors_text = "\n".join(risk_factors)
        
        # Recommendations based on level
        recommendations = {
            "CRITICAL": "🚫 **Block** transaction immediately. Require MFA and manual review before release.",
            "HIGH": "⚠️ **Flag** for review. Require step-up authentication before approval.",
            "MEDIUM": "📋 **Monitor** with enhanced tracking. Send user notification for verification.",
            "LOW": "✅ **Allow** with standard monitoring. No immediate action required.",
        }
        rec = recommendations.get(risk_level, recommendations["MEDIUM"])
        
        return f"""### 📊 Key Findings
- Risk Level: **{risk_level}** (combined score: {combined:.3f})
- Active Risk Factors: **{severity_count}** detected
- Transaction: **${amt:.2f}** in **{cat}** at **{loc}**

### 📋 Risk Factor Analysis
{risk_factors_text}

### ✅ Recommendation
{rec}

### 📈 Risk Score Breakdown
| Component | Score | Weight |
|---|---|---|
| Z-Score Flag | {zscore_flag:.3f} | 25% |
| Velocity Flag | {vel_flag:.3f} | 20% |
| Category Risk | {cat_score:.3f} | 15% |
| Geographic Risk | {txn.get('GEOGRAPHIC_RISK_SCORE', 0):.3f} | 15% |
| Isolation Forest | {iso_score:.3f} | 25% |
| **Combined** | **{combined:.3f}** | **100%** |

---
*🔒 Confidence: Deterministic analysis via GraphGuard rule-based models + Isolation Forest. AI context engine unavailable.*
"""

    def analyze_fraud_with_context(self, transaction_data: dict, rag_context: str) -> str:
        """Analyze a fraudulent transaction using Gemini + RAG with structured output."""
        if not self.model or not rag_context:
            return self._generate_fallback_analysis(transaction_data)

        risk_level = transaction_data.get('RISK_LEVEL', 'UNKNOWN')
        combined = transaction_data.get('COMBINED_RISK_SCORE', 0)
        amt = transaction_data.get('AMOUNT', 0)

        prompt = f"""You are GraphGuard AI, an expert fraud analyst. Analyze this flagged transaction and provide a STRUCTURED report.

Transaction Data:
{json.dumps(transaction_data, indent=2, default=str)}

Historical Context for this User (from RAG):
{rag_context}

You MUST format your response EXACTLY as follows (use these exact section headers):

### 📊 Key Findings
- 3-4 bullet points summarizing the most important facts
- Include the risk level, key anomalies, and comparison to user baseline
- Be specific with numbers and percentages

### 📋 Risk Factor Analysis
- For each risk factor detected, explain WHY it was flagged
- Use severity ratings: LOW / MEDIUM / HIGH for each factor
- Compare current transaction against user's historical patterns from RAG context

### ✅ Recommendation
- One clear, actionable recommendation with specific next steps
- Include urgency level

### 📈 Data Points
- List 3-5 key data points used in the analysis as a bullet list
- Include both this transaction's values and the user's baseline values

IMPORTANT RULES:
- Be CONCISE — each section should be 2-4 bullet points max
- Use specific numbers, not vague language
- End with: "🔒 Confidence: [HIGH/MEDIUM/LOW] — Based on [N] data sources"
"""
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini Analysis Failed ({e}), using fallback heuristics.")
            return self._generate_fallback_analysis(transaction_data)

    def answer_question_with_rag(self, question: str, rag_context: str) -> str:
        """Answer a free-form question using RAG context with structured, summarized output."""
        if not self.model or not rag_context:
            return "🔒 **AI-powered Q&A requires a Google API key.**"

        prompt = f"""You are GraphGuard AI, a data analyst for a fraud detection platform.
Answer the user's question using ONLY the provided context data. If the answer is not in the context, say "I don't have enough data to answer that."

Context Data:
{rag_context}

Question: {question}

You MUST format your response in this EXACT structure:

### 📊 Key Findings
- Directly answer the user's question with specific data points.
- Use numbers, percentages, and metrics from the context.
- Be analytical—don't just list facts, explain what they mean.

### 📋 Detailed Analysis
- Provide a deeper look into the patterns or trends in the data.
- Compare values (e.g., this user vs. others, this category vs. average).
- Identify anomalies or consistent behaviors.

### ✅ Strategic Recommendations
- Offer 1-2 practical, actionable steps based on the data findings.
- Focus on business value, fraud prevention, or user experience.

IMPORTANT RULES:
- Provide MEANINGFUL, descriptive answers. Avoid one-word bullets if a sentence adds more value.
- Use specific data from the context. If the data is missing, state it clearly.
- Use a professional, analytical tone.
- End with: "🔒 Confidence: [HIGH/MEDIUM/LOW] — Based on [N] data sources"
- If data is insufficient, be honest about limitations
"""
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=prompt
            )
            return response.text
        except Exception as e:
            return f"⚠️ **Error generating answer:** {e}"
