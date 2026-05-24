import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS_PATH = "data/documents"
VECTOR_STORE_PATH = "data/vector_store"


def load_documents():
    loader = DirectoryLoader(
        DOCUMENTS_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs


def build_vector_store():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    print("Vector store built and saved!")
    return vector_store


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    return vector_store


def get_vector_store():
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        print("Loading existing vector store...")
        return load_vector_store()
    else:
        print("Building new vector store...")
        return build_vector_store()


def retrieve(query: str) -> str:
    vector_store = get_vector_store()
    results = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    return context