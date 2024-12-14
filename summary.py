import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.8, api_key=GOOGLE_API_KEY)


# Stuff Chain 
def short_docs(short_text):
    document = Document(page_content=short_text)
    stuff_chain = load_summarize_chain(llm, chain_type = "stuff", verbose = True) 
    summary = stuff_chain.invoke([document])
    return summary


short_text = input("""Content: """)
output = short_docs(short_text)
print("Summary:\n",output["output_text"])


# Map Reduce Chain
def long_docs(long_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    docs = text_splitter.create_documents([long_text])
    map_reduce_chain = load_summarize_chain(llm, chain_type = "map_reduce", verbose = False)
    summary = map_reduce_chain.invoke(docs)
    return summary


long_text = input("""Content: """)
output = long_docs(long_text)
print("Summary:\n",output["output_text"])

    