# Importing necessary libraries
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain

# Retrieve the API key from env file 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")

# Initializing the language model
llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.5, api_key=GOOGLE_API_KEY)

# Creating first chain
first_prompt = ChatPromptTemplate.from_template("Analyze the sentiment of the following review and provide just the sentiment (positive or negative or neutral):{review}")
first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="sentiment" )


# Creating second chain
second_prompt = ChatPromptTemplate.from_template("Generate response message for the given review based on sentiment provided:{review}{sentiment}")
second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="response")

# Using sequential chain 
overall_chain = SequentialChain(
    chains=[first_chain, second_chain],
    input_variables=["review"],
    output_variables=["sentiment", "response"],
    verbose=True
)

# Example positive review
result = overall_chain.invoke({"review": "This product is amazing! It exceeded all my expectations."})
print(result["sentiment"])
print(result["response"])

# Example negative review
result = overall_chain.invoke({"review": "This product is terrible. It didn't work as advertised and was a waste of money."})
print(result["sentiment"])
print(result["response"])

# Example neutral review
result = overall_chain.invoke({"review": "The product is okay. It does what it says, but nothing special."})
print(result["sentiment"])
print(result["response"])
