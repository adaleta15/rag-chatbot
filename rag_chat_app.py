import streamlit as st
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Set environment variables correctly
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")  # from .env
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # from .env

# ✅ LangChain imports (updated)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ✅ UI
st.set_page_config(page_title="RAG Breast Cancer Assistant")
st.title("💬 Breast Cancer FAQ Assistant")

# ✅ Use local PDF (no upload for now)
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file is not None:
    loader = PyPDFLoader(uploaded_file.name)
else:
    loader = PyPDFLoader("breast_cancer_guidelines_demo.pdf")


# ✅ Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Load model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ✅ Load and split document
with st.spinner("🔍 Indexing document..."):
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # ✅ Embedding and FAISS index
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ✅ Memory-enabled chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

# ✅ Chat input interface
user_input = st.chat_input("Ask a question about breast cancer...")
if user_input:
    response = chain.run(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Assistant", response))

# ✅ Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:** {msg}")
