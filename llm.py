from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GEMINI_API_KEY"]=os.getenv("CHATGEMINI_API_KEY")



def get_llm():
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.75,max_output_tokens=2000)
    return llm
