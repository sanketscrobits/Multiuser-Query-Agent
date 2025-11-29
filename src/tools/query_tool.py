from langchain.tools import tool
from src.utils.vector_db.vector_store_singleton import VectorStoreSingleton
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.vector_db.loader_strategies.local_loader import LocalLoader
from src.utils.vector_db.index_strategies.pinecone_vector_index import PineconeVectorIndex
from langchain_core.runnables import RunnableConfig


@tool
def get_context(query_text: str, config: RunnableConfig) -> str:
    """
    This function helps to answer user question by retrieving relevant context from documents.
    
    Args: 
        query_text: User question in string format
        config: Configuration dictionary (injected)
        
    Returns: 
        Context related to user's question in string format
    """

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    document_loader_strategy = LocalLoader()
    vector_index_strategy = PineconeVectorIndex(embeddings=embeddings_model)

    vector_store = VectorStoreSingleton(
        embeddings_model=embeddings_model,
        document_loader_strategy=document_loader_strategy,
        vector_index_strategy=vector_index_strategy,
    )

    if config is None:
        config = {}
    
    print(f"DEBUG: get_context tool config: {config}")
        
    # Try to get namespace from config first, then fallback to context var
    namespace = config.get("configurable", {}).get("namespace")
    
    if namespace is None:
        from src.utils.request_context import get_namespace
        namespace = get_namespace()
        print(f"DEBUG: Retrieved namespace from ContextVar: {namespace}")
    else:
        print(f"DEBUG: Retrieved namespace from Config: {namespace}")
        
    result = vector_store.query(query_text = query_text, namespace=namespace)
    return result

if __name__ == "__main__":
    print(get_context("What initiative did the federal government announce regarding AI?"))