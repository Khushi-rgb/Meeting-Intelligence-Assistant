
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def build_vector_store(transcript: str) -> Chroma:

    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    # Open/create the same persistent Chroma collection
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    # Get all existing document IDs
    existing_data = vector_store.get()

    existing_ids = existing_data.get("ids", [])

    # Delete old transcript documents
    if existing_ids:
        vector_store.delete(ids=existing_ids)

    # Add new transcript documents
    vector_store.add_documents(docs)

    print("Total chunks created:", len(docs))
    print("Documents in vector DB:", vector_store._collection.count())

    return vector_store


def load_vector_store() -> Chroma:

    embeddings = get_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )


def get_retriever(vector_store: Chroma, k: int = 4):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )