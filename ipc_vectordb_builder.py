import json
import os
from dotenv import load_dotenv

from langchain_community.docstore.document import Document
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings



# Shared Embeddings Configuration

def get_embeddings():
    """Return Azure OpenAI embedding function."""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )



# Vector DB BUILDER

def build_ipc_vectordb():
    """
    Build and persist a Chroma vector database from IPC JSON data.
    This function should be run only once.
    """
    load_dotenv()
    print(">>> Starting IPC Vector DB build")

    ipc_json_path = os.getenv("IPC_JSON_PATH")
    persist_dir = os.getenv("PERSIST_DIRECTORY_PATH")
    collection_name = os.getenv("IPC_COLLECTION_NAME")

    if not all([ipc_json_path, persist_dir, collection_name]):
        raise EnvironmentError("❌ Missing required environment variables")

    # Load IPC JSON
    with open(ipc_json_path, "r", encoding="utf-8") as f:
        ipc_data = json.load(f)

    print(f">>> Loaded {len(ipc_data)} IPC sections")

    # Prepare documents
    documents = [
        Document(
            page_content=(
                f"Section {entry['Section']}: {entry['section_title']}\n\n"
                f"{entry['section_desc']}"
            ),
            metadata={
                "chapter": entry["chapter"],
                "chapter_title": entry["chapter_title"],
                "section": entry["Section"],
                "section_title": entry["section_title"],
            }
        )
        for entry in ipc_data
    ]

    print(f">>> Prepared {len(documents)} documents")

    # Create vector store
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    print("✅ IPC Vector DB build completed successfully")



# Vector DB RETRIEVER (Runtime)

def load_ipc_retriever(k: int = 3):
    """
    Load the persisted IPC vector DB and return a retriever.
    Used at runtime by ipc_section_agent.
    """
    load_dotenv()

    persist_dir = os.getenv("PERSIST_DIRECTORY_PATH")
    collection_name = os.getenv("IPC_COLLECTION_NAME")

    if not persist_dir or not collection_name:
        raise EnvironmentError("❌ Missing vector DB environment variables")

    embeddings = get_embeddings()

    vectordb = Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings
    )

    return vectordb.as_retriever(search_kwargs={"k": k})


# Entry Point

if __name__ == "__main__":
    build_ipc_vectordb()
