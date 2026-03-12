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

        print("⚡ Indexing local knowledge base...")
        

        # 2. Index CFPB Complaints (the contextual knowledge base)
        cfpb_path = PROJECT_ROOT / "dataset" / "csv_data" / "cfpb_credit_card.csv"
        if cfpb_path.exists():
            print(f"Indexing CFPB complaints from {cfpb_path}...")
            # Load a larger chunk for better coverage
            df_cfpb = pd.read_csv(cfpb_path, nrows=1000)
            documents, metadatas, ids = [], [], []
            for i, row in df_cfpb.iterrows():
                issue = row.get('Issue', 'Financial Issue')
                complaint = row.get('Consumer complaint narrative', '')
                if pd.isna(complaint) or complaint == '':
                    complaint = f"Consumer reported an issue regarding {issue} with {row.get('Company', 'a financial institution')}."
                
                text = f"CFPB Complaint: {issue}. Narrative: {complaint}"
                documents.append(text)
                metadatas.append({"type": "complaint", "company": str(row.get("Company", "Unknown")), "state": str(row.get("State", "Unknown"))})
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
                    "category": category, 
                    "difficulty": qa.get("difficulty", "N/A")
                })
                ids.append(f"qa_{qa.get('id', 'unknown')}")
            
            if documents:
                self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        # 4. Index PDF Documents (the custom knowledge base)
        if PDF_DATA_PATH.exists():
            print(f"Indexing PDF documents from {PDF_DATA_PATH}...")
            for pdf_file in PDF_DATA_PATH.glob("*.pdf"):
                print(f"Processing {pdf_file.name}...")
                try:
                    text_content = self._extract_text_from_pdf(pdf_file)
                    if not text_content:
                        continue
                    
                    # Split into chunks of ~1000 characters with 100 char overlap
                    chunks = self._chunk_text(text_content, chunk_size=1000, overlap=100)
                    
                    documents, metadatas, ids = [], [], []
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "type": "pdf_doc",
                            "filename": pdf_file.name,
                            "chunk_index": i
                        })
                        ids.append(f"pdf_{pdf_file.stem}_{i}")
                    
                    if documents:
                        self._collection.add(documents=documents, metadatas=metadatas, ids=ids)
                except Exception as e:
                    print(f"Error processing {pdf_file.name}: {e}")
            
        print(f"✅ Indexed {self._collection.count()} items locally.")

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


    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Semantic search with multi-stage re-ranking.
        
        Prioritizes Expert QA data over raw transaction context if confidence is high.
        """
        if not self._collection:
            return []

        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results * 2  # Fetch double for re-ranking
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Convert distances to confidence (rough inverse sigmoid)
        parsed = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            conf = max(0, 1 - (dist / 1.5))
            parsed.append({
                "text": doc,
                "metadata": meta,
                "confidence": conf,
                "type": meta.get("type", "unknown")
            })

        # Multi-stage Re-ranking: Move Expert QA to the top if confidence > 40%
        expert_qa = [r for r in parsed if r["type"] == "expert_qa" and r["confidence"] > 0.4]
        other_data = [r for r in parsed if r not in expert_qa]

        # Sort within groups by confidence
        expert_qa.sort(key=lambda x: x["confidence"], reverse=True)
        other_data.sort(key=lambda x: x["confidence"], reverse=True)

        merged = expert_qa + other_data
        return merged[:n_results]

    def get_context_for_query(self, query_text: str) -> str:
        """Returns a formatted context string for the LLM."""
        results = self.query(query_text, n_results=3)
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
