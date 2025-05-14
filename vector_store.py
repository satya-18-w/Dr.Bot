from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma,FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4

load_dotenv()
api=os.getenv("CHATGEMINI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

def load_pdf()-> list:
    loader=PyPDFLoader(r"C:\Users\Satyajit Samal\OneDrive\Desktop\GenAI Projects\DRbot\Dr.Bot\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf")
    docs=loader.load()
    return docs




def split_docs(data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks=text_splitter.split_documents(data)
    return chunks

docs=load_pdf()
chunks=split_docs(docs)

    
    
    
    
def get_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    return embeddings
embedding=get_embedding()

# index_name = "drbot"  # change if desired

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=3072,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

# index = pc.Index(index_name)

# uuids = [str(uuid4()) for _ in range(len(chunks))]
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(chunks,embedding=embedding)
db.save_local(DB_FAISS_PATH)
#)
