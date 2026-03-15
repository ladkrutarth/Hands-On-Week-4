import os
import pandas as pd
import chromadb
import json
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Optional, Any
import pypdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / ".chroma_db_multimodal"
PDF_DATA_PATH = PROJECT_ROOT / "dataset" / "pdf_data"
IMAGE_DATA_PATH = PROJECT_ROOT / "dataset" / "image_data"
CSV_DATA_PATH = PROJECT_ROOT / "dataset" / "csv_data"

class MultimodalRAG:
    """
    Dedicated RAG Engine for user-uploaded multimodal evidence.
    Operates on a separate collection to ensure strict isolation from system data.
    """
    
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._client = chromadb.PersistentClient(path=self.db_path)
        
        # Use a high-performance local embedding function
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self._collection = self._client.get_or_create_collection(
            name="veriscan_multimodal",
            embedding_function=self._embedding_function
        )
        print(f"✅ Multimodal RAG Engine initialized with {self._collection.count()} documents.")

    def index_data(self, session_id: Optional[str] = None, force: bool = False):
        """Index user-uploaded data in the session folder (isolated)."""
        SESS_ID = session_id or "global"
        SESSION_DIR = PROJECT_ROOT / "dataset" / "user_uploads" / SESS_ID
        
        if not SESSION_DIR.exists():
            print(f"DEBUG: No upload directory found for session {SESS_ID}")
            return

        print(f"⚡ Indexing user evidence for session: {SESS_ID}...")
        
        # 1. Scan the session folder
        for file_path in SESSION_DIR.glob("*"):
            ext = file_path.suffix.lower()
            fname = file_path.name
            
            try:
                # 1.A PDFs
                if ext == ".pdf":
                    print(f"Processing PDF: {fname}...")
                    text_content = self._extract_text_from_pdf(file_path)
                    if not text_content: continue
                    
                    # Summary Chunk
                    self._collection.upsert(
                        documents=[f"PDF Document Summary: {fname}. This document contains extracted text for analysis."],
                        metadatas=[{
                            "type": "pdf_summary", "is_user": True, "filename": fname, 
                            "session_id": SESS_ID, "timestamp": pd.Timestamp.now().isoformat()
                        }],
                        ids=[f"pdf_summary_{file_path.stem}_{SESS_ID}"]
                    )
                    
                    chunks = self._chunk_text(text_content)
                    docs, metas, ids = [], [], []
                    for i, chunk in enumerate(chunks):
                        docs.append(chunk)
                        metas.append({
                            "type": "pdf_doc", "is_user": True, "filename": fname, 
                            "chunk_index": i, "session_id": SESS_ID, "timestamp": pd.Timestamp.now().isoformat()
                        })
                        ids.append(f"pdf_{file_path.stem}_{SESS_ID}_{i}")
                    if docs: self._collection.upsert(documents=docs, metadatas=metas, ids=ids)

                # 1.B CSVs
                elif ext == ".csv":
                    print(f"Processing CSV: {fname}...")
                    df = pd.read_csv(file_path)
                    summary = f"CSV Dataset Summary: {fname}. Columns: {', '.join(df.columns)}. Total rows: {len(df)}."
                    self._collection.upsert(
                        documents=[summary],
                        metadatas=[{
                            "type": "csv_summary", "is_user": True, "filename": fname, 
                            "session_id": SESS_ID, "timestamp": pd.Timestamp.now().isoformat()
                        }],
                        ids=[f"csv_summary_{file_path.stem}_{SESS_ID}"]
                    )
                    
                    docs, metas, ids = [], [], []
                    for i in range(0, len(df), 5):
                        chunk = df.iloc[i:i+5]
                        content = f"CSV Evidence ({fname}) - Rows {i} to {i+len(chunk)-1}:\n{chunk.to_csv(index=False)}"
                        docs.append(content)
                        metas.append({
                            "type": "csv_doc", "is_user": True, "filename": fname, 
                            "session_id": SESS_ID, "chunk_start": i, "timestamp": pd.Timestamp.now().isoformat()
                        })
                        ids.append(f"csv_{file_path.stem}_{SESS_ID}_{i//5}")
                    if docs: self._collection.upsert(documents=docs, metadatas=metas, ids=ids)

                # 1.C Images (OCR)
                elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                    print(f"Processing Image: {fname} (OCR)...")
                    text = self._extract_text_from_image(file_path)
                    if text:
                        self._collection.upsert(
                            documents=[f"Visual Evidence Content ({fname}): {text}"],
                            metadatas=[{
                                "type": "image_doc", "is_user": True, "filename": fname, 
                                "session_id": SESS_ID, "timestamp": pd.Timestamp.now().isoformat()
                            }],
                            ids=[f"img_{file_path.stem}_{SESS_ID}"]
                        )

                # 1.D Plain Text / JSON
                elif ext in [".txt", ".json"]:
                    print(f"Processing Text/JSON: {fname}...")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    if text:
                        self._collection.upsert(
                            documents=[f"Plain Text Evidence ({fname}):\n{text[:5000]}"],
                            metadatas=[{
                                "type": "text_doc", "is_user": True, "filename": fname, 
                                "session_id": SESS_ID, "timestamp": pd.Timestamp.now().isoformat()
                            }],
                            ids=[f"text_{file_path.stem}_{SESS_ID}"]
                        )

            except Exception as e:
                print(f"Error indexing {fname}: {e}")

        print(f"✅ Multimodal Index for session {SESS_ID} complete. Total DB size: {self._collection.count()}")

    def query(self, query_text: str, n_results: int = 5, include_types: Optional[List[str]] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query user evidence with strict isolation."""
        if not self._collection: return []

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

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
        except Exception:
            # If collection is lost (e.g. during concurrent re-index), try to reload it once
            print("⚠️ Multimodal collection lost; attempting to reload...")
            self._collection = self._client.get_or_create_collection(
                name="veriscan_multimodal",
                embedding_function=self._embedding_function
            )
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
        
        parsed = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                parsed.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "type": results["metadatas"][0][i].get("type", "unknown")
                })
        
        # 4. Mandatory Recency Fallback: If querying session data, always include the most recent 3 items
        # to ensure that "summarize this" or general recent context is present.
        if session_id:
            try:
                recent = self._collection.get(
                    where={"session_id": session_id},
                    limit=3,
                    include=["documents", "metadatas"]
                )
                if recent["documents"]:
                    # Avoid duplicates by checking IDs
                    existing_texts = {p["text"] for p in parsed}
                    for i in range(len(recent["documents"])):
                        if recent["documents"][i] not in existing_texts:
                            parsed.insert(0, {
                                "text": recent["documents"][i],
                                "metadata": recent["metadatas"][i],
                                "type": recent["metadatas"][i].get("type", "recent_context")
                            })
            except Exception as e:
                print(f"Recency fallback error: {e}")

        return parsed

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        chunks = []
        if len(text) <= chunk_size: return [text]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _extract_text_from_image(self, image_path: Path) -> str:
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(image_path)
            return pytesseract.image_to_string(img)
        except: return ""
