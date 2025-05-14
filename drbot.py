import streamlit as st
from llm import retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from vector_store import get_embedding
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


rt=retriever()
llm=rt.get_llm()
prompt=rt.get_prompt()
embedding=get_embedding()



def vectotstore(embedding):
    DB_FAISS_PATH="vectorstore/db_faiss"
    db=FAISS.load_local(DB_FAISS_PATH,embedding,allow_dangerous_deserialization=True)
    return db


db=vectotstore(embedding)
retriever=db.as_retriever()
document_chain=create_stuff_documents_chain(llm,prompt)
retrieval_chain=create_retrieval_chain(retriever,document_chain)

    
    



# ret=retriever().get_retriever()
def main():
    st.title("Ask Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages=[]
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    prompt=st.chat_input("Enter your Query :")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
        
        
        
        
        
        
        
        
        # response=ret.invoke({"input":prompt})
        response=retrieval_chain.invoke({"input":prompt})
        st.chat_message("assistant").markdown(response.answer)
        st.session_state.messages.append({"role":"assistant","content":response})
        
        
if __name__ == "__main__":
    main()