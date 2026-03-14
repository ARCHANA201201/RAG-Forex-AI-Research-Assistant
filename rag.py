from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from newspaper import Article
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
load_dotenv()

# ---------------- CONFIG ---------------- #

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "fx_trade"

llm = None
vector_store = None


# ---------------- INITIALIZE ---------------- #

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )

    if vector_store is None:
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            persist_directory=str(VECTORSTORE_DIR)
        )


# ---------------- LOAD ARTICLES ---------------- #

def load_articles(urls):
    docs = []

    for url in urls:
        try:
            print(f"\nDownloading article: {url}")

            article = Article(url)
            article.download()
            article.parse()

            text = article.text.strip()

            if len(text) < 200:
                print("⚠ Article too short. Skipping.")
                continue

            print("Article length:", len(text))

            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": url}
                )
            )

        except Exception as e:
            print(f"❌ Failed to load {url}: {e}")

    return docs


# ---------------- PROCESS URLS ---------------- #

def process_urls(urls):

    initialize_components()

    # vector_store.reset_collection()

    data = load_articles(urls)

    if len(data) == 0:
        print("⚠ No valid articles loaded.")
        return False

    print("\nSplitting articles...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )

    docs = splitter.split_documents(data)

    ids = [str(uuid4()) for _ in docs]

    print("Storing embeddings...")

    vector_store.add_documents(docs, ids=ids)

    print(f"✅ {len(docs)} chunks stored in vector database\n")

    return True


########################################################


def get_qa_chain():

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return qa_chain
# ---------------- MAIN PROGRAM ---------------- #
if __name__ == "__main__":

    print("\nEnter article URLs (type 'done' when finished):")

    urls = []

    while True:
        url = input("URL: ")

        if url.lower() == "done":
            break

        urls.append(url)

    success = process_urls(urls)

    if not success:
        print("No data available. Exiting.")
        exit()

    # ---------------- RAG CHATBOT ---------------- #

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    print("\n🤖 RAG Chatbot Ready")
    print("Type 'exit' to stop\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        result = qa_chain.invoke({"question": query})

        print("\nBot:", result["answer"])
        print("\n------------------------------------\n")