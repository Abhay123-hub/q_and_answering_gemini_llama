## here in this .py file i am going to generate word embeddings of my input text,along with srive context
## vector store index,query engine and will be returning the query engine
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
import traceback

def download_gemini_embedding(model, documents):
    """
    Creates a vector store index with Gemini embeddings and returns a query engine.

    Parameters:
    - model: The LLM model to use with the service context.
    - documents: The documents to be embedded in the vector store index.

    Returns:
    - query_engine: The query engine for querying the vector store index.
    - None: If an error occurs during the process.
    """
    try:
        if not model:
            raise ValueError("The model parameter is required but was not provided.")
        if not documents:
            raise ValueError("The documents parameter is required but was not provided.")

        # Initialize the embedding model and service context
        gemini_embed_model = GeminiEmbedding(model_name='models/text-embedding-004')
        service_context = ServiceContext(
            llm=model,
            embed_model=gemini_embed_model,
            chunk_size=800,
            chunk_overlap=20
        )

        # Create the index from the documents and persist it
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()

        # Return the query engine for the created index
        query_engine = index.as_query_engine()
        print("Query engine created successfully.")
        return query_engine

    except Exception as e:
        # Print the traceback for debugging
        print("Failed to create the query engine. See error details below:")
        traceback.print_exception(type(e), e, e.__traceback__)
        return None

        
