from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from vector_store import get_embedding
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st



import os
from dotenv import load_dotenv

load_dotenv()
api=os.getenv("CHATGEMINI_API_KEY")
# if you want to use huggingface endpoint
# repo_id="mistralai/Mistral-7B-Instruct-v0.2"
# llm=HuggingFaceEndpoint(repo_id,temprature=0.8model_kwargs={"token":"","max_length":2344})


class retriever:
    def __init__(self):
        self.api=api
        

    def get_llm(self):
        self.llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.75,max_output_tokens=2000,api_key=api)
        return self.llm
    
    def get_prompt(self):
        
            
        
        prompt=ChatPromptTemplate.from_template(
            """ Use the pieces of information provided in the context to answer the user's question with an accurate and human language fromat.
            If you donot know the answer just say that you donot know or try your4 own knowledge to give the answer.
            Donot provide anything out of context If you do this job i will give you $4000
            Context: {context}
            Question: {input}
            """
        )
        return prompt
    
        # self.db=get_vectorstore()
        # retriever=self.db.as_retriever()
        # document_chain=create_stuff_documents_chain(self.llm,prompt)
        # retrieval_chain=create_retrieval_chain(retriever,document_chain)
        # return retrieval_chain

        # Creating QA chain
        # Another is through RetrievalQA

