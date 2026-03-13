import os
import pandas as pd
import chromadb
import json
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Optional, Any
import pypdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / ".chroma_db_local"
PDF_DATA_PATH = PROJECT_ROOT / "dataset" / "pdf_data"

class RAGEngineLocal:
    """
    RAG Engine using local embeddings (sentence-transformers) and ChromaDB.
    No external API calls are made during indexing or retrieval.
    """
    
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._client = chromadb.PersistentClient(path=self.db_path)
        
        # Use a high-performance local embedding function
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self._collection = self._client.get_or_create_collection(
            name="veriscan_intel_local",
            embedding_function=self._embedding_function
        )
        print(f"✅ Local RAG Engine initialized with {self._collection.count()} documents.")

    def index_data(self, force: bool = False):
        """Load and index transaction, complaint, and expert QA data."""
        if self._collection.count() > 0 and not force:
            return

        if force:
            print("🗑️ Clearing existing collection for re-indexing...")
            self._client.delete_collection(name="veriscan_intel_local")
            self._collection = self._client.create_collection(
                name="veriscan_intel_local",
                embedding_function=self._embedding_function
            )

        print("⚡ Indexing local knowledge base...")
        

        # 2. Index CFPB Complaints (the contextual knowledge base)
        cfpb_path = PROJECT_ROOT / "dataset" / "csv_data" / "cfpb_credit_card.csv"
        if cfpb_path.exists():
            print(f"Indexing CFPB complaints from {cfpb_path}...")
            # Load a larger chunk for better coverage
            df_cfpb = pd.read_csv(cfpb_path, nrows=1000)
            documents, metadatas, ids = [], [], []
            seen_texts = set()
            
            for i, row in df_cfpb.iterrows():
                issue = row.get('Issue', 'Financial Issue')
                sub_issue = row.get('Sub-issue', '')
                company = row.get('Company', 'a financial institution')
                complaint = row.get('Consumer complaint narrative', '')
                
                # Contextual fallback if narrative is missing
                if pd.isna(complaint) or complaint == '':
                    detail = f" specifically relating to {sub_issue}" if not pd.isna(sub_issue) and sub_issue != '' else ""
                    complaint = f"Consumer reported a problem regarding {issue}{detail} with {company}."
                
                text = f"CFPB Complaint [{issue}]: {complaint}"
                
                # Index-time deduplication: Skip if we've already generated this exact text
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                
                documents.append(text)
                metadatas.append({
                    "type": "complaint", 
                    "is_system": True,
                    "company": str(company), 
                    "issue": str(issue),
                    "sub_issue": str(sub_issue) if not pd.isna(sub_issue) else "",
                    "state": str(row.get("State", "Unknown"))
                })
                ids.append(f"cfpb_{i}")
            
            # Batch add
            batch_size = 100
            for j in range(0, len(documents), batch_size):
                self._collection.add(
                    documents=documents[j:j+batch_size],
                    metadatas=metadatas[j:j+batch_size],
                    ids=ids[j:j+batch_size]
                )

        # 3. Index Fraud Expert Q&A (the expert intelligence base)
        qa_path = PROJECT_ROOT / "dataset" / "csv_data" / "fraud_detection_qa_dataset.json"
        if qa_path.exists():
            print(f"Indexing expert fraud intelligence from {qa_path}...")
            with open(qa_path, 'r') as f:
                qa_data = json.load(f)
            
            documents, metadatas, ids = [], [], []
            for qa in qa_data.get("qa_pairs", []):
                category = qa.get("category", "General")
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                
                text = f"Expert Intelligence ({category}): {question} - {answer}"
                documents.append(text)
                metadatas.append({
                    "type": "expert_qa", 
                    "is_system": True,
                    "category": category, 
                    "difficulty": qa.get("difficulty", "N/A")
                })
                ids.append(f"qa_{qa.get('id', 'unknown')}")
            
            if documents:
                self._collection.add(documents=documents, metadatas=metadatas, ids=ids)

        # 3b. Index Top Scam Types (newly added context)
        scam_path = PROJECT_ROOT / "dataset" / "csv_data" / "top10_scam_types_by_losses.csv"
        if scam_path.exists():
            print(f"Indexing top scam types from {scam_path}...")
            df_scam = pd.read_csv(scam_path)
            documents, metadatas, ids = [], [], []
            for i, row in df_scam.iterrows():
                scam_name = row.get('Scam Type', 'Unknown Scam')
                desc = row.get('Description', '')
                losses = row.get('Victim Losses 2024', '$0')
                
                text = f"Expert Intelligence (Scam Profile): {scam_name}. Details: {desc}. Estimated annual victim losses: {losses}."
                documents.append(text)
                metadatas.append({
                    "type": "scam_profile",
                    "is_system": True,
                    "scam_name": str(scam_name),
                    "losses": str(losses)
                })
                ids.append(f"scam_{i}")
            
            if documents:
                self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
        # 4. Index System PDF Documents (built-in knowledge)
        if PDF_DATA_PATH.exists():
            print(f"Indexing system knowledge from {PDF_DATA_PATH}...")
            # We specifically index known system documents to avoid user-data crosstalk
            system_pdfs = ["2024_IC3Report.pdf", "The-Scam-Economy_The-True-Cost-of-Online-Scams.pdf"]
            for pdf_file_name in system_pdfs:
                pdf_file = PDF_DATA_PATH / pdf_file_name
                if not pdf_file.exists(): continue
                print(f"Processing system PDF: {pdf_file.name}...")
                try:
                    text_content = self._extract_text_from_pdf(pdf_file)
                    if not text_content:
                        continue
                    
                    chunks = self._chunk_text(text_content, chunk_size=1000, overlap=100)
                    documents, metadatas, ids = [], [], []
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "type": "pdf_doc",
                            "is_system": True,
                            "filename": pdf_file.name,
                            "chunk_index": i
                        })
                        ids.append(f"sys_pdf_{pdf_file.stem}_{i}")
                    
                    if documents:
                        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                except Exception as e:
                    print(f"Error processing system PDF {pdf_file.name}: {e}")

        print(f"✅ Indexed {self._collection.count()} items locally in System RAG.")

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file."""
        text = ""
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        if len(text) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def _extract_text_from_image(self, image_path: Path) -> str:
        """Extract text from an image using Tesseract OCR."""
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(image_path)
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""


    def query(self, query_text: str, n_results: int = 5, include_types: Optional[List[str]] = None, only_user_data: bool = False, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Semantic search with strict isolation and type filtering."""
        if not self._collection:
            return []

        # Prepare filter logic
        conditions = []
        if include_types:
            if len(include_types) == 1:
                conditions.append({"type": include_types[0]})
            else:
                conditions.append({"type": {"$in": include_types}})
        
        if only_user_data:
            conditions.append({"is_user": True})
            # Note: session_id filtering can be added here if session_id is indexed
            # For now, we use is_user as a strict barrier against is_system
            
        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Fetch results
        results = self._collection.query(
            query_texts=[query_text],
            n_results=max(100, n_results * 8),
            where=where_filter
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        parsed = []
        seen_fuzzy = set()
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            # Fuzzy Deduplication:
            # 1. Normalize text (lower, strip, remove common redundant parts)
            norm = doc.lower().strip()
            # Remove company names from complaints for similarity check if possible
            # or just take the first 150 chars as a "signature"
            signature = norm[:150] 
            
            if signature in seen_fuzzy:
                continue
            seen_fuzzy.add(signature)
            
            conf = max(0, 1 - (dist / 1.5))
            parsed.append({
                "text": doc,
                "metadata": meta,
                "confidence": conf,
                "type": meta.get("type", "unknown")
            })

        # 2. Group by type for diversity
        by_type = {}
        for r in parsed:
            t = r["type"]
            if t not in by_type: by_type[t] = []
            by_type[t].append(r)
            
        expert_qa = by_type.get("expert_qa", [])
        scam_profiles = by_type.get("scam_profile", [])
        pdf_docs = by_type.get("pdf_doc", [])
        image_docs = by_type.get("image_doc", [])
        csv_docs = by_type.get("csv_doc", [])
        complaints = by_type.get("complaint", [])
        
        # Expert/Visual content gets a boost
        for r in expert_qa + scam_profiles + image_docs:
            if r["confidence"] > 0.4:
                r["confidence"] += 0.1
                
        # Interleave sources
        sources = [expert_qa, scam_profiles, image_docs, pdf_docs, csv_docs, complaints]
        merged = []
        max_depth = max(len(src) for src in sources) if sources else 0
        
        for i in range(max_depth):
            # Prioritize Expert/Scam/PDF in each round
            for src in sources:
                if i < len(src):
                    merged.append(src[i])
            if len(merged) >= n_results:
                break
            
        return merged[:n_results]

    def get_context_for_query(self, query_text: str, include_types: Optional[List[str]] = None) -> str:
        """Returns a formatted context string for the LLM."""
        results = self.query(query_text, n_results=3, include_types=include_types)
        if not results:
            return "No relevant context found."
        
        context = []
        for i, res in enumerate(results, 1):
            source = res["type"].upper()
            context.append(f"[{source} {i}]: {res['text']}")
        
        return "\n\n".join(context)

if __name__ == "__main__":
    engine = RAGEngineLocal()
    engine.index_data(force=True)
    
    test_q = "Are there any high value transactions in travel?"
    print(f"\nQuery: {test_q}")
    results = engine.query(test_q)
    for r in results:
        print(f"[{r['confidence']}] {r['text']}")
