"""
Multimodal RAG — session-isolated indexing and retrieval for user uploads.

Improvements vs naive path:
- Page-aware PDF extract with OCR fallback for scanned pages
- Image OCR chunking with honest empty-OCR metadata
- Delete-before-reindex per file+session
- Lexical boost + content-type preference in query
"""

from __future__ import annotations

import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import pandas as pd
import pypdf
from chromadb.utils import embedding_functions

from models.rag_relevance import (
    apply_lexical_boost,
    candidate_n,
    distance_to_confidence,
    filter_by_relevance,
    format_grounded_context,
    improve_chunks,
    is_summary_query,
    is_unreadable_visual_evidence,
    is_visual_query,
    prefer_content_types,
    rerank_hits,
    snippet_fallback_answer,
)
from paths import CHROMA_MULTIMODAL_DIR, PROJECT_ROOT, UPLOADS_DIR

DB_PATH = CHROMA_MULTIMODAL_DIR
MIN_PAGE_CHARS = 40
MIN_OCR_CHARS = 20


class MultimodalRAG:
    """Dedicated RAG engine for user-uploaded multimodal evidence."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._client = chromadb.PersistentClient(path=self.db_path)
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name="veriscan_multimodal",
            embedding_function=self._embedding_function,
        )
        print(f"✅ Multimodal RAG Engine initialized with {self._collection.count()} documents.")

    # ── Delete prior chunks for a file ─────────────────────────────────────

    def _delete_file_chunks(self, filename: str, session_id: str) -> None:
        """Remove stale vectors for filename+session before re-index."""
        try:
            existing = self._collection.get(
                where={
                    "$and": [
                        {"filename": filename},
                        {"session_id": session_id},
                    ]
                },
                include=[],
            )
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
        except Exception as e:
            # Older Chroma may not like $and — try sequential soft cleanup by id prefix
            print(f"⚠️ delete_file_chunks soft-fail ({e}); continuing upsert.")

    # ── OCR / raster helpers ───────────────────────────────────────────────

    def _configure_tesseract(self) -> None:
        """Point pytesseract at common Homebrew / MacTeX binary locations."""
        if getattr(self, "_tesseract_configured", False):
            return
        self._tesseract_configured = True
        try:
            import shutil
            import pytesseract

            existing = getattr(pytesseract.pytesseract, "tesseract_cmd", None)
            if existing and Path(existing).exists():
                return
            for candidate in (
                shutil.which("tesseract"),
                "/opt/homebrew/bin/tesseract",
                "/usr/local/bin/tesseract",
            ):
                if candidate and Path(candidate).exists():
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    return
        except Exception:
            pass

    def _ocr_with_rapidocr(self, img) -> str:
        """Pure-Python OCR fallback when system Tesseract is missing."""
        try:
            import numpy as np
            from rapidocr_onnxruntime import RapidOCR

            engine = getattr(self, "_rapid_ocr", None)
            if engine is None:
                engine = RapidOCR()
                self._rapid_ocr = engine
            arr = np.array(img.convert("RGB") if hasattr(img, "convert") else img)
            result, _ = engine(arr)
            if not result:
                return ""
            # RapidOCR returns list of [box, text, score]
            lines = []
            for item in result:
                if not item or len(item) < 2:
                    continue
                text = item[1]
                if text:
                    lines.append(str(text).strip())
            return "\n".join(lines).strip()
        except Exception as e:
            print(f"⚠️ RapidOCR unavailable: {e}")
            return ""

    def _ocr_pil_image(self, img) -> str:
        self._configure_tesseract()
        # Prefer Tesseract when installed; fall back to RapidOCR (no brew needed).
        try:
            import pytesseract

            text = (pytesseract.image_to_string(img) or "").strip()
            if len(text) >= MIN_OCR_CHARS:
                return text
        except Exception as e:
            print(f"⚠️ Tesseract OCR unavailable: {e}")
            text = ""

        rapid = self._ocr_with_rapidocr(img)
        if len(rapid) > len(text or ""):
            return rapid
        return text or rapid

    def _rasterize_pdf_page_from_path(self, pdf_path: Path, page_index: int):
        """Return a PIL Image for page_index, or None."""
        try:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(str(pdf_path))
            if page_index < 0 or page_index >= len(doc):
                return None
            page = doc[page_index]
            bitmap = page.render(scale=2.0)
            return bitmap.to_pil()
        except Exception:
            pass
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                str(pdf_path),
                first_page=page_index + 1,
                last_page=page_index + 1,
                dpi=200,
            )
            return images[0] if images else None
        except Exception:
            return None

    def _rasterize_pdf_page_from_bytes(self, pdf_bytes: bytes, page_index: int):
        try:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(pdf_bytes)
            if page_index < 0 or page_index >= len(doc):
                return None
            page = doc[page_index]
            bitmap = page.render(scale=2.0)
            return bitmap.to_pil()
        except Exception:
            pass
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = Path(tmp.name)
            img = self._rasterize_pdf_page_from_path(tmp_path, page_index)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return img
        except Exception:
            return None

    def _extract_pages_from_pdf_path(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Return list of {page, text, ocr_used}."""
        pages: List[Dict[str, Any]] = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                ocr_used = False
                if len(text) < MIN_PAGE_CHARS:
                    img = self._rasterize_pdf_page_from_path(pdf_path, i)
                    if img is not None:
                        ocr_text = self._ocr_pil_image(img)
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            ocr_used = True
                pages.append({"page": i + 1, "text": text, "ocr_used": ocr_used})
        return pages

    def _extract_pages_from_pdf_bytes(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            ocr_used = False
            if len(text) < MIN_PAGE_CHARS:
                img = self._rasterize_pdf_page_from_bytes(pdf_bytes, i)
                if img is not None:
                    ocr_text = self._ocr_pil_image(img)
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        ocr_used = True
            pages.append({"page": i + 1, "text": text, "ocr_used": ocr_used})
        return pages

    def _extract_text_from_image(self, image_path: Path) -> str:
        try:
            from PIL import Image

            img = Image.open(image_path)
            return self._ocr_pil_image(img)
        except Exception as e:
            print(f"⚠️ Image OCR failed for {image_path.name}: {e}")
            return ""

    def _extract_text_from_image_bytes(self, file_bytes: bytes) -> Tuple[str, Optional[str]]:
        """Returns (ocr_text, error)."""
        try:
            from PIL import Image, ImageOps, ImageEnhance

            img = Image.open(io.BytesIO(file_bytes))
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            # Light preprocessing helps screenshots
            img = ImageOps.exif_transpose(img)
            if max(img.size) > 2200:
                img.thumbnail((2200, 2200))
            gray = ImageOps.grayscale(img)
            gray = ImageEnhance.Contrast(gray).enhance(1.6)
            text = self._ocr_pil_image(gray)
            if len(text) < MIN_OCR_CHARS:
                # retry on color RGB
                text2 = self._ocr_pil_image(img)
                if len(text2) > len(text):
                    text = text2
            return text, None
        except Exception as e:
            return "", str(e)

    def live_analyze_session_images(self, session_id: str, query_text: str = "") -> Dict[str, Any]:
        """
        Query-time image analysis without Vision LLM:
        re-OCR files on disk and return text context + diagnostics.
        """
        paths = self.resolve_session_image_paths(session_id)
        if not paths:
            return {"text": "", "files": [], "error": "no_session_images"}

        parts: List[str] = []
        files_info = []
        last_err = None
        for p in paths[:3]:
            try:
                raw = p.read_bytes()
                text, err = self._extract_text_from_image_bytes(raw)
                last_err = err or last_err
                files_info.append(
                    {"filename": p.name, "ocr_chars": len(text or ""), "path": str(p)}
                )
                if text and len(text.strip()) >= MIN_OCR_CHARS:
                    parts.append(f"[Live OCR from {p.name}]\n{text.strip()}")
                else:
                    parts.append(
                        f"[Image {p.name}] OCR extracted little text"
                        + (f" ({err})" if err else "")
                        + ". Vision LLM recommended for this screenshot."
                    )
            except Exception as e:
                last_err = str(e)
                files_info.append({"filename": p.name, "error": str(e)})

        return {
            "text": "\n\n".join(parts),
            "files": files_info,
            "error": last_err,
            "has_useful_ocr": any(f.get("ocr_chars", 0) >= MIN_OCR_CHARS for f in files_info),
        }

    def _hits_from_get(
        self,
        got: Dict[str, Any],
        *,
        type_allow: Optional[set] = None,
        default_type: str = "doc",
        confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        hits = []
        for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
            meta = meta or {}
            ftype = meta.get("type", default_type)
            if type_allow is not None and ftype not in type_allow:
                continue
            # Skip honest empty/weak stubs for document force-include
            if meta.get("empty") == "true" or meta.get("ocr_weak") == "true":
                continue
            if is_unreadable_visual_evidence(doc or ""):
                continue
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "type": ftype,
                    "distance": 0.4,
                    "confidence": confidence,
                    "filename": meta.get("filename"),
                    "page": meta.get("page"),
                }
            )
        return hits

    def force_image_hits(self, session_id: str) -> List[Dict[str, Any]]:
        """Return indexed image_doc chunks for a session (bypass relevance filter)."""
        try:
            got = self._collection.get(
                where={
                    "$and": [
                        {"session_id": session_id},
                        {"type": "image_doc"},
                    ]
                },
                include=["documents", "metadatas"],
            )
            return self._hits_from_get(got, type_allow={"image_doc"}, default_type="image_doc")
        except Exception:
            try:
                got = self._collection.get(
                    where={"session_id": session_id},
                    include=["documents", "metadatas"],
                )
                return self._hits_from_get(got, type_allow={"image_doc"}, default_type="image_doc")
            except Exception:
                return []

    def force_document_hits(self, session_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Return readable PDF/CSV/text chunks when dense retrieval returns nothing."""
        allow = {"pdf_doc", "pdf_summary", "csv_doc", "csv_summary", "text_doc"}
        try:
            got = self._collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"],
            )
            hits = self._hits_from_get(got, type_allow=allow, default_type="pdf_doc", confidence=0.55)
            # Prefer page chunks over summaries
            hits.sort(key=lambda h: 0 if h.get("type") == "pdf_doc" else 1)
            return hits[:limit]
        except Exception:
            return []

    def needs_vision(self, query_text: str, hits: List[Dict[str, Any]], session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        if is_visual_query(query_text):
            return bool(self.session_image_files(session_id) or self.resolve_session_image_paths(session_id))
        if not hits:
            return bool(self.session_image_files(session_id) or self.resolve_session_image_paths(session_id))
        avg_conf = sum(float(h.get("confidence", 0)) for h in hits) / max(len(hits), 1)
        weak = avg_conf < 0.35 or all(
            (h.get("metadata") or {}).get("ocr_weak") == "true"
            for h in hits
            if h.get("type") == "image_doc"
        )
        return weak and bool(self.resolve_session_image_paths(session_id))

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        return improve_chunks(text, chunk_size=chunk_size, overlap=overlap)

    # ── Index from session folder ──────────────────────────────────────────

    def index_data(self, session_id: Optional[str] = None, force: bool = False):
        """Index user-uploaded data in the session folder (isolated)."""
        sess_id = session_id or "global"
        session_dir = UPLOADS_DIR / sess_id

        if not session_dir.exists():
            print(f"DEBUG: No upload directory found for session {sess_id}")
            return

        print(f"⚡ Indexing user evidence for session: {sess_id}...")
        for file_path in session_dir.glob("*"):
            if not file_path.is_file():
                continue
            try:
                file_bytes = file_path.read_bytes()
                result = self.index_file_bytes(file_path.name, file_bytes, session_id=sess_id)
                print(f"  → {file_path.name}: {result.get('status')} {result}")
            except Exception as e:
                print(f"Error indexing {file_path.name}: {e}")

        print(f"✅ Multimodal Index for session {sess_id} complete. Total DB size: {self._collection.count()}")

    # ── Core indexing from bytes ───────────────────────────────────────────

    def index_file_bytes(self, filename: str, file_bytes: bytes, session_id: str = "global") -> dict:
        """Index a file from raw bytes (Streamlit / API)."""
        ext = Path(filename).suffix.lower()
        stem = Path(filename).stem
        ts = pd.Timestamp.now().isoformat()

        try:
            self._delete_file_chunks(filename, session_id)

            if ext == ".pdf":
                return self._index_pdf_bytes(filename, stem, file_bytes, session_id, ts)
            if ext == ".csv":
                return self._index_csv_bytes(filename, stem, file_bytes, session_id, ts)
            if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
                return self._index_image_bytes(filename, stem, file_bytes, session_id, ts)
            if ext in [".txt", ".json"]:
                text = file_bytes.decode("utf-8", errors="ignore")
                self._delete_file_chunks(filename, session_id)
                chunks = self._chunk_text(text)
                docs, metas, ids = [], [], []
                for i, chunk in enumerate(chunks or [text[:5000]]):
                    docs.append(f"[From Text: {filename}] {chunk}")
                    metas.append(
                        {
                            "type": "text_doc",
                            "is_user": True,
                            "filename": filename,
                            "session_id": session_id,
                            "chunk_index": i,
                            "timestamp": ts,
                            "char_count": str(len(chunk)),
                        }
                    )
                    ids.append(f"text_{stem}_{session_id}_{i}")
                if docs:
                    self._collection.upsert(documents=docs, metadatas=metas, ids=ids)
                return {"filename": filename, "status": "indexed", "chars": len(text), "chunks": len(docs)}
            return {"filename": filename, "status": "unsupported"}
        except Exception as e:
            return {"filename": filename, "status": "error", "error": str(e)}

    def _index_pdf_bytes(
        self, filename: str, stem: str, file_bytes: bytes, session_id: str, ts: str
    ) -> dict:
        pages = self._extract_pages_from_pdf_bytes(file_bytes)
        ocr_pages = sum(1 for p in pages if p.get("ocr_used"))
        nonempty = [p for p in pages if (p.get("text") or "").strip()]
        if not nonempty:
            # Honest empty index record so inventory still lists the file
            self._collection.upsert(
                documents=[
                    f"PDF Document: {filename}. No extractable text "
                    f"({len(pages)} pages). File may be scanned; OCR unavailable or failed."
                ],
                metadatas=[
                    {
                        "type": "pdf_summary",
                        "is_user": True,
                        "filename": filename,
                        "session_id": session_id,
                        "timestamp": ts,
                        "ocr_used": "false",
                        "empty": "true",
                        "pages": str(len(pages)),
                    }
                ],
                ids=[f"pdf_summary_{stem}_{session_id}"],
            )
            return {
                "filename": filename,
                "status": "empty",
                "error": "No text extracted",
                "pages": len(pages),
                "ocr_pages": ocr_pages,
                "chunks": 0,
            }

        full_preview = "\n".join(p["text"][:400] for p in nonempty[:3])
        self._collection.upsert(
            documents=[f"PDF Document: {filename}. Preview:\n{full_preview[:800]}"],
            metadatas=[
                {
                    "type": "pdf_summary",
                    "is_user": True,
                    "filename": filename,
                    "session_id": session_id,
                    "timestamp": ts,
                    "pages": str(len(pages)),
                    "ocr_pages": str(ocr_pages),
                }
            ],
            ids=[f"pdf_summary_{stem}_{session_id}"],
        )

        docs, metas, ids = [], [], []
        chunk_i = 0
        for p in nonempty:
            page_no = int(p["page"])
            page_text = p["text"]
            ocr_used = bool(p.get("ocr_used"))
            for chunk in self._chunk_text(page_text):
                docs.append(f"[From PDF: {filename} | page {page_no}] {chunk}")
                metas.append(
                    {
                        "type": "pdf_doc",
                        "is_user": True,
                        "filename": filename,
                        "session_id": session_id,
                        "page": page_no,
                        "chunk_index": chunk_i,
                        "timestamp": ts,
                        "char_count": str(len(chunk)),
                        "ocr_used": "true" if ocr_used else "false",
                    }
                )
                ids.append(f"pdf_{stem}_{session_id}_p{page_no}_{chunk_i}")
                chunk_i += 1
        if docs:
            self._collection.upsert(documents=docs, metadatas=metas, ids=ids)
        return {
            "filename": filename,
            "status": "indexed",
            "pages": len(pages),
            "ocr_pages": ocr_pages,
            "chunks": len(docs),
        }

    def _index_csv_bytes(
        self, filename: str, stem: str, file_bytes: bytes, session_id: str, ts: str
    ) -> dict:
        df = pd.read_csv(io.BytesIO(file_bytes))
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        sample = df.head(5).to_csv(index=False)
        summary = (
            f"CSV Dataset: {filename}. Columns: {', '.join(df.columns.tolist())}. "
            f"Total rows: {len(df)}. Numeric: {', '.join(num_cols)}. "
            f"Categorical: {', '.join(cat_cols)}.\nSample data:\n{sample}"
        )
        self._collection.upsert(
            documents=[summary],
            metadatas=[
                {
                    "type": "csv_summary",
                    "is_user": True,
                    "filename": filename,
                    "session_id": session_id,
                    "timestamp": ts,
                    "columns": json.dumps(df.columns.tolist()),
                    "row_count": str(len(df)),
                }
            ],
            ids=[f"csv_summary_{stem}_{session_id}"],
        )
        docs, metas, ids = [], [], []
        for i in range(0, len(df), 5):
            chunk = df.iloc[i : i + 5]
            content = f"[From CSV: {filename}] Rows {i}-{i + len(chunk) - 1}:\n{chunk.to_csv(index=False)}"
            docs.append(content)
            metas.append(
                {
                    "type": "csv_doc",
                    "is_user": True,
                    "filename": filename,
                    "session_id": session_id,
                    "chunk_start": i,
                    "timestamp": ts,
                }
            )
            ids.append(f"csv_{stem}_{session_id}_{i // 5}")
        if docs:
            self._collection.upsert(documents=docs, metadatas=metas, ids=ids)
        return {
            "filename": filename,
            "status": "indexed",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "chunks": len(docs),
        }

    def _index_image_bytes(
        self, filename: str, stem: str, file_bytes: bytes, session_id: str, ts: str
    ) -> dict:
        text, err = self._extract_text_from_image_bytes(file_bytes)
        size_kb = round(len(file_bytes) / 1024.0, 1)
        docs, metas, ids = [], [], []

        if len(text.strip()) >= MIN_OCR_CHARS:
            chunks = self._chunk_text(text) or [text]
            for i, chunk in enumerate(chunks):
                docs.append(f"[From Image OCR: {filename}] {chunk}")
                metas.append(
                    {
                        "type": "image_doc",
                        "is_user": True,
                        "filename": filename,
                        "session_id": session_id,
                        "chunk_index": i,
                        "timestamp": ts,
                        "ocr_used": "true",
                        "char_count": str(len(chunk)),
                        "size_kb": str(size_kb),
                    }
                )
                ids.append(f"img_{stem}_{session_id}_{i}")
            status = "indexed"
            ocr_preview = text[:200]
        else:
            # Honest stub — never invent visual content
            reason = err or "OCR returned little or no text"
            stub = (
                f"Image file: {filename} ({size_kb} KB). {reason}. "
                "Ask a visual question so Vision analysis can inspect the image, "
                "or upload a clearer scan."
            )
            docs.append(stub)
            metas.append(
                {
                    "type": "image_doc",
                    "is_user": True,
                    "filename": filename,
                    "session_id": session_id,
                    "timestamp": ts,
                    "ocr_used": "false",
                    "ocr_weak": "true",
                    "size_kb": str(size_kb),
                    "char_count": "0",
                }
            )
            ids.append(f"img_{stem}_{session_id}_0")
            status = "indexed_weak_ocr"
            ocr_preview = ""

        self._collection.upsert(documents=docs, metadatas=metas, ids=ids)
        return {
            "filename": filename,
            "status": status,
            "ocr_text": ocr_preview,
            "chunks": len(docs),
            "size_kb": size_kb,
        }

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        include_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query user evidence with relevance filtering, lexical boost, rerank."""
        if not self._collection or self._collection.count() == 0:
            return []

        conditions = []
        if include_types:
            if len(include_types) == 1:
                conditions.append({"type": include_types[0]})
            else:
                conditions.append({"type": {"$in": include_types}})
        if session_id:
            conditions.append({"session_id": session_id})

        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        fetch_n = min(candidate_n(n_results), max(1, self._collection.count()))
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=fetch_n,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"⚠️ Multimodal query error ({e}); retrying with session filter only...")
            try:
                self._collection = self._client.get_or_create_collection(
                    name="veriscan_multimodal",
                    embedding_function=self._embedding_function,
                )
                where_retry = {"session_id": session_id} if session_id else None
                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=min(fetch_n, max(1, self._collection.count())),
                    where=where_retry,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e2:
                print(f"⚠️ Multimodal query failed: {e2}")
                return []

        parsed: List[Dict[str, Any]] = []
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        if not dists or len(dists) != len(docs):
            dists = [0.5] * len(docs)

        seen = set()
        for doc, meta, dist in zip(docs, metas, dists):
            if not doc:
                continue
            signature = doc.strip().lower()[:160]
            if signature in seen:
                continue
            seen.add(signature)
            meta = meta or {}
            conf = distance_to_confidence(dist)
            parsed.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "type": meta.get("type", "unknown"),
                    "distance": float(dist),
                    "confidence": conf,
                    "filename": meta.get("filename"),
                    "page": meta.get("page"),
                }
            )

        if session_id and is_summary_query(query_text):
            try:
                recent = self._collection.get(
                    where={"session_id": session_id},
                    limit=5,
                    include=["documents", "metadatas"],
                )
                if recent.get("documents"):
                    existing = {p["text"] for p in parsed}
                    for i, doc in enumerate(recent["documents"]):
                        if doc in existing:
                            continue
                        meta = recent["metadatas"][i] or {}
                        parsed.append(
                            {
                                "text": doc,
                                "metadata": meta,
                                "type": meta.get("type", "recent_context"),
                                "distance": 0.9,
                                "confidence": 0.4,
                                "filename": meta.get("filename"),
                                "page": meta.get("page"),
                            }
                        )
            except Exception as e:
                print(f"Recency fallback error: {e}")

        boosted = apply_lexical_boost(query_text, parsed)
        preferred = prefer_content_types(boosted, query_text)
        filtered = filter_by_relevance(preferred)
        ranked = rerank_hits(query_text, filtered, top_k=n_results)
        # Ensure citation fields on output
        for h in ranked:
            meta = h.get("metadata") or {}
            h.setdefault("filename", meta.get("filename"))
            h.setdefault("page", meta.get("page"))
        return ranked

    def get_context_for_query(
        self,
        query_text: str,
        n_results: int = 6,
        include_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        hits = self.query(
            query_text,
            n_results=n_results,
            include_types=include_types,
            session_id=session_id,
        )
        return format_grounded_context(hits) or "No relevant context found."

    def answer_from_retrieval(
        self,
        query_text: str,
        n_results: int = 6,
        include_types: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return grounded snippet answer + sources (LLM-free fallback)."""
        hits = self.query(
            query_text,
            n_results=n_results,
            include_types=include_types,
            session_id=session_id,
        )
        return {
            "reply": snippet_fallback_answer(query_text, hits),
            "sources": [
                {
                    "text": h["text"],
                    "metadata": h.get("metadata", {}),
                    "confidence": h.get("confidence"),
                    "filename": h.get("filename"),
                    "page": h.get("page"),
                }
                for h in hits
            ],
            "hits": hits,
        }

    def session_image_files(self, session_id: str) -> List[Dict[str, Any]]:
        """List image docs in session (for query-time Vision)."""
        inv = self.get_file_inventory(session_id)
        return [
            f
            for f in inv
            if str(f.get("type", "")).startswith("image")
            or str(f.get("filename", "")).lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

    def resolve_session_image_paths(self, session_id: str) -> List[Path]:
        folder = UPLOADS_DIR / session_id
        if not folder.exists():
            return []
        out = []
        for p in folder.iterdir():
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                out.append(p)
        return out

    def get_file_inventory(self, session_id: str) -> List[Dict[str, Any]]:
        """Return a list of files indexed for a given session."""
        try:
            results = self._collection.get(
                where={"session_id": session_id},
                include=["metadatas"],
            )
            files: Dict[str, Dict[str, Any]] = {}
            if results.get("metadatas"):
                for meta in results["metadatas"]:
                    fname = meta.get("filename", "unknown")
                    ftype = meta.get("type", "unknown")
                    if fname not in files:
                        files[fname] = {"filename": fname, "type": ftype, "chunks": 0}
                    files[fname]["chunks"] += 1
            return list(files.values())
        except Exception:
            return []

    # Back-compat helpers used by older call sites
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        pages = self._extract_pages_from_pdf_path(pdf_path)
        return "\n".join(p["text"] for p in pages if p.get("text")).strip()
