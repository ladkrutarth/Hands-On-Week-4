"""Shared RAG relevance helpers: scoring, thresholds, rerank, grounded prompts."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

# Chroma default space for SentenceTransformerEmbeddingFunction is L2.
# Empirically, dist ~> 1.2 is usually off-topic for MiniLM on short docs.
DEFAULT_MAX_DISTANCE = 1.15
DEFAULT_MIN_CONFIDENCE = 0.28
CANDIDATE_MULTIPLIER = 8
CANDIDATE_FLOOR = 24

_SUMMARY_QUERY_RE = re.compile(
    r"\b("
    r"summarize|summary|overview|recap|"
    r"what\s+(did|have)\s+i\s+(upload|file|document)|"
    r"list\s+(my\s+)?(files?|documents?|uploads?)"
    r")\b",
    re.IGNORECASE,
)


def distance_to_confidence(distance: float, scale: float = 1.5) -> float:
    """Map L2 distance to a 0–1 confidence score."""
    try:
        dist = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - (dist / scale)))


def is_summary_query(query_text: str) -> bool:
    return bool(_SUMMARY_QUERY_RE.search(query_text or ""))


def candidate_n(n_results: int) -> int:
    return max(CANDIDATE_FLOOR, int(n_results) * CANDIDATE_MULTIPLIER)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy-load cross-encoder; returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"⚠️ Cross-encoder unavailable ({e}); using distance ranking only.")
        return None


def rerank_hits(
    query_text: str,
    hits: Sequence[Dict[str, Any]],
    top_k: int,
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Rerank candidates with a cross-encoder when available."""
    if not hits:
        return []
    if len(hits) == 1 or top_k <= 0:
        return list(hits)[: max(1, top_k)]

    model = _get_cross_encoder()
    if model is None:
        # Fall back to confidence / distance ordering
        return sorted(
            hits,
            key=lambda h: (
                float(h.get("confidence", 0.0)),
                -float(h.get("distance", 99.0)),
            ),
            reverse=True,
        )[:top_k]

    pairs = [[query_text, (h.get(text_key) or "")[:2000]] for h in hits]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        print(f"⚠️ Rerank failed ({e}); using distance ranking.")
        return sorted(
            hits,
            key=lambda h: float(h.get("confidence", 0.0)),
            reverse=True,
        )[:top_k]

    ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
    out: List[Dict[str, Any]] = []
    for hit, score in ranked[:top_k]:
        enriched = dict(hit)
        enriched["rerank_score"] = float(score)
        # Blend: keep distance confidence but prefer strong rerank
        base = float(enriched.get("confidence", 0.0))
        # Map typical cross-encoder logits roughly into [0,1]
        rr = max(0.0, min(1.0, (float(score) + 5.0) / 10.0))
        enriched["confidence"] = max(base, rr)
        out.append(enriched)
    return out


def filter_by_relevance(
    hits: Sequence[Dict[str, Any]],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    relax_if_empty: bool = True,
    relax_min_confidence: float = 0.18,
) -> List[Dict[str, Any]]:
    """Drop weak hits; optionally relax once if everything is filtered out."""
    kept = []
    for h in hits:
        conf = float(h.get("confidence", 0.0))
        dist = h.get("distance")
        if dist is not None and float(dist) > max_distance and conf < min_confidence:
            continue
        if conf < min_confidence:
            continue
        kept.append(h)

    if kept or not relax_if_empty or not hits:
        return kept

    # Soft fallback: take best few weak hits rather than inventing context
    soft = [
        h
        for h in hits
        if float(h.get("confidence", 0.0)) >= relax_min_confidence
    ]
    return soft[:3] if soft else list(hits)[:1]


def extract_lexical_tokens(query_text: str) -> List[str]:
    """Tokens useful for lexical boost: currency, amounts, filenames, keywords."""
    q = query_text or ""
    tokens: List[str] = []
    # $ amounts and bare numbers with decimals
    tokens.extend(re.findall(r"\$\s?\d[\d,]*(?:\.\d+)?", q))
    tokens.extend(re.findall(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", q))
    tokens.extend(re.findall(r"\b\d+\.\d{2}\b", q))
    # filename-like tokens
    tokens.extend(re.findall(r"\b[\w\-]+\.(?:pdf|png|jpg|jpeg|csv|txt|json)\b", q, flags=re.I))
    # significant words (length >= 4), skip stopwords
    stop = {
        "what", "when", "where", "which", "that", "this", "with", "from", "have",
        "does", "about", "please", "show", "tell", "give", "list", "into", "your",
        "document", "documents", "file", "files", "image", "images", "page",
    }
    for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", q):
        if w.lower() not in stop:
            tokens.append(w)
    # dedupe preserving order
    seen = set()
    out = []
    for t in tokens:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[:24]


def apply_lexical_boost(
    query_text: str,
    hits: Sequence[Dict[str, Any]],
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Bump confidence when chunk text contains query amounts/keywords/filenames."""
    tokens = extract_lexical_tokens(query_text)
    if not hits:
        return []
    if not tokens:
        return list(hits)

    boosted: List[Dict[str, Any]] = []
    for h in hits:
        item = dict(h)
        text = (item.get(text_key) or "").lower()
        meta = item.get("metadata") or {}
        fname = str(meta.get("filename") or "").lower()
        matches = 0
        for tok in tokens:
            t = tok.lower().lstrip("$").strip()
            if not t:
                continue
            if t in text or t in fname:
                matches += 1
        if matches:
            bump = min(0.35, 0.08 * matches)
            item["confidence"] = min(1.0, float(item.get("confidence", 0.0)) + bump)
            item["lexical_matches"] = matches
            # Prefer exact lexical hits in sort by slightly lowering distance
            if item.get("distance") is not None:
                item["distance"] = max(0.0, float(item["distance"]) - 0.05 * matches)
        boosted.append(item)
    return boosted


def prefer_content_types(
    hits: Sequence[Dict[str, Any]],
    query_text: str,
) -> List[Dict[str, Any]]:
    """Down-rank summary stubs unless the user asked for a summary/overview."""
    if not hits:
        return []
    if is_summary_query(query_text):
        return list(hits)
    content_types = {"pdf_doc", "image_doc", "csv_doc", "text_doc"}
    content = [h for h in hits if (h.get("type") or (h.get("metadata") or {}).get("type")) in content_types]
    summaries = [h for h in hits if h not in content]
    # Keep summaries only as filler if we have few content hits
    if len(content) >= 2:
        return content + summaries[:1]
    return content + summaries


def is_visual_query(query_text: str) -> bool:
    q = query_text or ""
    return bool(
        re.search(
            r"\b("
            r"image|screenshot|photo|picture|visual|ocr|"
            r"looks?\s+like|in\s+the\s+(image|photo|screenshot)|"
            r"tell\s+me\s+about\s+(this|the)|"
            r"describe\s+(this|the)|"
            r"what\s+(is|does)\s+(this|the)\s+(image|screenshot|photo|picture)|"
            r"what\s+do\s+you\s+see|"
            r"analyze\s+(this|the)\s+(image|screenshot|photo)"
            r")\b",
            q,
            re.I,
        )
    )


def format_grounded_context(hits: Sequence[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Format ranked hits for an LLM prompt with source labels."""
    if not hits:
        return ""
    parts: List[str] = []
    total = 0
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata") or {}
        fname = meta.get("filename") or h.get("type") or "source"
        page = meta.get("page")
        conf = float(h.get("confidence", 0.0))
        text = (h.get("text") or "").strip()
        page_bit = f" | page={page}" if page is not None else ""
        block = f"[CHUNK {i} | SOURCE: {fname}{page_bit} | conf={conf:.2f}]\n{text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


def is_unreadable_visual_evidence(text: str) -> bool:
    """True when text is a placeholder/stub, not real OCR/Vision content."""
    t = (text or "").strip().lower()
    if not t or t in {"(none)", "none"}:
        return True
    markers = (
        "ocr returned little",
        "ocr extracted little",
        "little or no text",
        "tesseract is not installed",
        "ocr not available",
        "vision llm recommended",
        "visual content uploaded for analysis",
        "could not read",
        "no usable ocr",
        "lack of ocr",
        "without ocr",
    )
    if any(m in t for m in markers):
        return True
    # Filename-only / near-empty visual blocks
    if t.startswith("[image ") and "ocr" in t and len(t) < 280:
        return True
    if "screenshot" in t and len(t) < 120 and "live ocr" not in t:
        return True
    return False


def grounded_system_prompt(context: str, visual_context: str = "") -> str:
    """Strict grounding instructions for RAG chat."""
    ctx = (context or "").strip()
    vis = (visual_context or "").strip()
    unread_ctx = is_unreadable_visual_evidence(ctx)
    unread_vis = is_unreadable_visual_evidence(vis)

    if (not ctx or unread_ctx) and (not vis or unread_vis):
        return (
            "You are a financial document assistant. No readable OCR or Vision transcript "
            "is available for the uploaded image/document.\n"
            "Reply with ONLY this message (no speculation):\n"
            "'I can see that an image was uploaded, but I cannot read its contents yet "
            "because OCR/Vision text is unavailable. Install Tesseract (`brew install tesseract`) "
            "or enable the Vision LLM, then re-index and ask again.'\n"
            "Do NOT invent tables, columns, dates, amounts, merchants, or layout details."
        )

    return (
        "You are a financial evidence assistant.\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using facts explicitly present in [DOCUMENT CONTEXT] and [VISUAL EVIDENCE].\n"
        "2. If those sections do not contain readable extracted text (only placeholders, "
        "filenames, or 'OCR failed' notes), say you cannot read the image yet. "
        "Do NOT guess that it is a spreadsheet, bank statement, or any other document type.\n"
        "3. NEVER invent column headers (Date/Description/Amount/Balance), transactions, "
        "or UI layout that is not explicitly transcribed in the evidence.\n"
        "4. If the context does not contain the answer, say exactly: "
        "'The provided evidence does not contain enough information to answer that.'\n"
        "5. Cite sources using the SOURCE labels when available.\n"
        "6. Prefer concise, factual answers tied to the user question.\n\n"
        f"[DOCUMENT CONTEXT]\n{ctx or '(none)'}\n\n"
        f"[VISUAL EVIDENCE]\n{vis or '(none)'}"
    )


def snippet_fallback_answer(query: str, hits: Sequence[Dict[str, Any]]) -> str:
    """Deterministic answer when LLM is unavailable — only high-confidence snippets."""
    if not hits:
        return (
            "No relevant content found in your uploaded documents for that question. "
            "Try a more specific query, or confirm the files were indexed."
        )
    # Refuse when evidence is weak / off-topic relative to query tokens
    best = max(float(h.get("confidence", 0.0)) for h in hits)
    tokens = [t.lower().lstrip("$") for t in extract_lexical_tokens(query)]
    blob = " ".join((h.get("text") or "").lower() for h in hits)
    lexical_hit = any(t in blob for t in tokens if len(t) >= 3) if tokens else True
    if best < 0.32 or (tokens and not lexical_hit and best < 0.55):
        return (
            "The provided evidence does not contain enough information to answer that. "
            "Try rephrasing, or upload documents that mention the topic."
        )
    lines = [f"Based on the most relevant passages for: **{query.strip()}**\n"]
    by_file: Dict[str, List[str]] = {}
    for h in hits:
        fname = (h.get("metadata") or {}).get("filename", h.get("filename") or "Unknown")
        by_file.setdefault(fname, []).append(h.get("text") or "")
    for fname, texts in by_file.items():
        lines.append(f"**From {fname}:**")
        for t in texts[:2]:
            clean = re.sub(r"^\[From (PDF|CSV|Image OCR|Text): [^\]]+\]\s*", "", t).strip()
            # strip page annotations
            clean = re.sub(r"^\[From PDF: [^\]]+\]\s*", "", clean).strip()
            if len(clean) > 450:
                clean = clean[:450] + "..."
            lines.append(f"> {clean}\n")
    return "\n".join(lines)


def improve_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Paragraph-aware chunking with character fallback."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = max(end - overlap, start + 1)
            continue
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= chunk_size:
            buf = f"{buf}\n\n{para}"
        else:
            chunks.append(buf)
            # soft overlap: keep tail of previous chunk
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = f"{tail}\n\n{para}".strip() if tail else para
    if buf:
        chunks.append(buf)
    return chunks or [text[:chunk_size]]
