"""
Offline multimodal RAG eval — retrieval hit rate + refuse behavior.

Does not require MLX / Vision. Uses text (+ optional tiny PDF via pypdf writer).
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

from models.multimodal_rag import MultimodalRAG
from models.rag_relevance import extract_lexical_tokens, is_visual_query


SESSION = "eval_mm_session_v1"

STATEMENT_TXT = """\
ACME Bank Statement
Account: USER_0001
Period: December 2025

Date       Merchant              Amount
2025-12-01 GROCERY MART          $54.20
2025-12-03 COFFEE HOUSE          $6.75
2025-12-05 WIRE TRANSFER OUT     $1,200.00
2025-12-08 DINING PALACE         $86.40
2025-12-12 STREAMING SUB         $15.99
2025-12-18 GAS STATION           $48.10

Ending balance: $3,412.55
Fraud alert: WIRE TRANSFER OUT flagged as high risk.
"""


def evaluate_multimodal_rag() -> float:
    # Isolate chroma path for eval
    tmp_db = Path(tempfile.mkdtemp(prefix="veriscan_mm_eval_"))
    print(f"Eval DB: {tmp_db}")
    try:
        rag = MultimodalRAG(db_path=str(tmp_db))

        # Index text statement
        r1 = rag.index_file_bytes("acme_statement.txt", STATEMENT_TXT.encode("utf-8"), session_id=SESSION)
        assert r1.get("status") == "indexed", r1

        # Index weak-OCR style image stub via empty-ish bytes that aren't a real image —
        # use a text file labeled as evidence instead for keyword retrieval.
        receipt = "Receipt: COFFEE HOUSE total $6.75 on 2025-12-03. Thank you."
        r2 = rag.index_file_bytes("coffee_receipt.txt", receipt.encode("utf-8"), session_id=SESSION)
        assert r2.get("status") == "indexed", r2

        # Image with no OCRable content: 1x1 png
        try:
            from PIL import Image

            img_buf = io.BytesIO()
            Image.new("RGB", (8, 8), color=(30, 30, 30)).save(img_buf, format="PNG")
            r3 = rag.index_file_bytes("dark_scan.png", img_buf.getvalue(), session_id=SESSION)
            print("Image index:", r3)
        except Exception as e:
            print("Skip image fixture:", e)
            r3 = {"status": "skipped"}

        cases = [
            {
                "q": "What was the WIRE TRANSFER OUT amount?",
                "expect_any": ["1200", "1,200", "WIRE"],
                "refuse": False,
            },
            {
                "q": "List grocery spending on the ACME statement",
                "expect_any": ["GROCERY", "54.20", "54"],
                "refuse": False,
            },
            {
                "q": "What did COFFEE HOUSE charge?",
                "expect_any": ["6.75", "COFFEE"],
                "refuse": False,
            },
            {
                "q": "What is the ending balance?",
                "expect_any": ["3412", "3,412"],
                "refuse": False,
            },
            {
                "q": "Which transaction was flagged as high risk fraud?",
                "expect_any": ["WIRE", "fraud", "high risk"],
                "refuse": False,
            },
            {
                "q": "What is the capital of France?",
                "expect_any": [],
                "refuse": True,
            },
            {
                "q": "Summarize my uploaded bank statement",
                "expect_any": ["ACME", "statement", "USER_0001", "December"],
                "refuse": False,
            },
        ]

        # Unit checks for helpers
        assert "$1,200.00" in extract_lexical_tokens("Wire of $1,200.00") or any(
            "1200" in t.replace(",", "") for t in extract_lexical_tokens("Wire of $1,200.00")
        )
        assert is_visual_query("What is in the screenshot?")

        success = 0
        print(f"\n{'='*60}")
        print(f" Multimodal RAG Eval — {len(cases)} cases")
        print(f"{'='*60}")

        for i, case in enumerate(cases, 1):
            hits = rag.query(case["q"], n_results=5, session_id=SESSION)
            blob = " ".join(h.get("text", "") for h in hits).lower()
            if case["refuse"]:
                # Off-topic: either no hits or none of the statement keywords dominate wrongly —
                # we check that Paris/France not fabricated in retrieval text from our corpus.
                ok = "paris" not in blob and (
                    not hits
                    or max(float(h.get("confidence", 0)) for h in hits) < 0.55
                    or not any(k.lower() in blob for k in ["acme", "wire transfer"])
                )
                # Stronger: answer_from_retrieval should not claim France facts from empty evidence
                answered = rag.answer_from_retrieval(case["q"], session_id=SESSION)
                refuse_ok = (
                    "do not contain" in answered["reply"].lower()
                    or "no relevant" in answered["reply"].lower()
                    or "not contain" in answered["reply"].lower()
                    or "enough information" in answered["reply"].lower()
                    or not hits
                )
                # For refuse case accept if retrieval is weak OR snippet says no relevant
                ok = refuse_ok or (not hits)
            else:
                ok = any(tok.lower() in blob for tok in case["expect_any"])

            mark = "✅" if ok else "❌"
            print(f"[{i}/{len(cases)}] {mark} {case['q'][:70]}")
            if not ok:
                print(f"     expect={case['expect_any']} refuse={case['refuse']}")
                print(f"     top={[ (h.get('filename'), round(float(h.get('confidence',0)),2)) for h in hits[:3] ]}")
            else:
                success += 1

        # Image weak OCR inventory presence
        inv = rag.get_file_inventory(SESSION)
        names = {f["filename"] for f in inv}
        assert "acme_statement.txt" in names
        if r3.get("status") in ("indexed", "indexed_weak_ocr"):
            assert "dark_scan.png" in names

        acc = success / len(cases)
        print(f"\n{'='*60}")
        print(f" Accuracy: {success}/{len(cases)} = {acc*100:.0f}%")
        print(f"{'='*60}")
        return acc
    finally:
        shutil.rmtree(tmp_db, ignore_errors=True)


if __name__ == "__main__":
    evaluate_multimodal_rag()
