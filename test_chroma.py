import traceback
try:
    from google import genai
    from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
    import chromadb

    client = genai.Client()
    class GeminiEmbedder(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            return [[0.1, 0.2]]
    
    db_client = chromadb.Client()
    db_client.get_or_create_collection("test", embedding_function=GeminiEmbedder())
    print("Success")
except Exception as e:
    traceback.print_exc()
