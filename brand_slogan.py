import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")

# Initialize LLM 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, api_key=GOOGLE_API_KEY)


# Generate a company name
first_prompt = PromptTemplate.from_template("What is the best creative name to describe a company that makes {product}?")
first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="name")

# Generate a slogan for the company
second_prompt = ChatPromptTemplate.from_template("Write a creative slogan for the following company: {name}")
second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="slogan")

# Combine the chains into a SequentialChain
overall_chain = SequentialChain(
    chains=[first_chain, second_chain],
    input_variables=["product"], 
    output_variables=["name", "slogan"],
    verbose=True,
)

 
result = overall_chain.invoke({"product": "Gaming Laptop"})
print("Company Name:", result["name"])
print("Slogan:", result["slogan"])


