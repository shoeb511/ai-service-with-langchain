from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedder)
model = ChatGroq()

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that answers questions based on the provided context. if the \
    context does not contain the answer just say, I don't know. please be consise and quote facts \
    directly from the context."
    )


