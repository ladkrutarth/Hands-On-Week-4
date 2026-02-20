import traceback
import sys
try:
    from models.rag_engine import RAGEngine
    import os
    # We will pass a dummy API key to force it to use GeminiEmbedder
    r = RAGEngine(api_key="dummy_api_key_to_force_gemini")
    
    if r._collection:
        print("Got collection")
        r._collection.add(ids=['test1'], documents=['hello world'])
        print("Added successfully")
except Exception as e:
    traceback.print_exc()
