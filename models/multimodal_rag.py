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

    def index_data(self, force: bool = False):
        """Index user-uploaded PDFs, Images, and CSVs."""
        if self._collection.count() > 0 and not force:
            return

        if force:
            print("🗑️ Clearing multimodal collection...")
            self._client.delete_collection(name="veriscan_multimodal")
            self._collection = self._client.create_collection(
                name="veriscan_multimodal",
                embedding_function=self._embedding_function
            )

        print("⚡ Indexing user evidence...")
        
        # 1. Index PDF Documents
        if PDF_DATA_PATH.exists():
            for pdf_file in PDF_DATA_PATH.glob("*.pdf"):
                # Skip system PDFs if they are in the same folder (architectural cleanup will eventually move them)
                if pdf_file.name in ["2024_IC3Report.pdf", "The-Scam-Economy_The-True-Cost-of-Online-Scams.pdf"]:
                    continue
                    
                print(f"Processing user PDF: {pdf_file.name}...")
                try:
                    text_content = self._extract_text_from_pdf(pdf_file)
                    if not text_content: continue
                    
                    chunks = self._chunk_text(text_content)
                    documents, metadatas, ids = [], [], []
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "type": "pdf_doc",
                            "is_user": True,
                            "filename": pdf_file.name,
                            "chunk_index": i
                        })
                        ids.append(f"pdf_{pdf_file.stem}_{i}")
                    
                    if documents:
                        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                except Exception as e:
                    print(f"Error processing {pdf_file.name}: {e}")
            
        # 2. Index Image Documents
        if IMAGE_DATA_PATH.exists():
            for img_file in IMAGE_DATA_PATH.glob("*"):
                if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
                    continue
                print(f"Processing user Image: {img_file.name} (OCR)...")
                try:
                    text_content = self._extract_text_from_image(img_file)
                    if not text_content: continue
                    
                    self._collection.add(
                        documents=[f"Visual Evidence Content ({img_file.name}): {text_content}"],
                        metadatas=[{
                            "type": "image_doc", 
                            "is_user": True,
                            "filename": img_file.name,
                            "source": "ocr_extraction"
                        }],
                        ids=[f"img_{img_file.stem}_{hash(text_content) % 10000}"]
                    )
                except Exception as e:
                    print(f"Error processing image {img_file.name}: {e}")

        # 3. Index CSV Documents
        if CSV_DATA_PATH.exists():
            for csv_file in CSV_DATA_PATH.glob("*.csv"):
                # Skip system CSVs
                if csv_file.name in ["cfpb_credit_card.csv", "top10_scam_types_by_losses.csv", "fraud_detection_qa_dataset.json", "financial_advisor_dataset.csv", "spending_dna_dataset.csv"]:
                    continue
                    
                print(f"Processing user CSV: {csv_file.name}...")
                try:
                    df = pd.read_csv(csv_file, nrows=50)
                    summary = f"User Dataset {csv_file.name} contains columns: {', '.join(df.columns)}. Sample data: {df.head(5).to_json()}"
                    self._collection.add(
                        documents=[summary],
                        metadatas=[{"type": "csv_doc", "is_user": True, "filename": csv_file.name}],
                        ids=[f"csv_{csv_file.stem}"]
                    )
                except Exception as e:
                    print(f"Error processing CSV {csv_file.name}: {e}")

        print(f"✅ Multimodal Index complete. Total: {self._collection.count()} items.")

    def query(self, query_text: str, n_results: int = 5, include_types: Optional[List[str]] = None, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query user evidence with strict isolation."""
        if not self._collection: return []

        conditions = []
        if include_types:
            if len(include_types) == 1:
                conditions.append({"type": include_types[0]})
            else:
                conditions.append({"type": {"$in": include_types}})
        
        # In the future, we can add session_id filtering here if session_id is indexed per document
        
        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

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
