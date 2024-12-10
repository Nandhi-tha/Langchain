# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Retrieve the API key from env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")

# Initializing the language model with specific parameters
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.7, api_key = GOOGLE_API_KEY)

# Movie recommendation based on specified language and genre
def movie_recommender():
    # Defining the message prompt template
    messages = [
        ("system", "You are a movie recommendation system based on ratings."),
        ("user", "Top 5 {language} movie names in {genre} genre.")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)

    # Creating the chain
    chain = prompt_template|llm|StrOutputParser()

    # Invoking the chain with necessary inputs
    response = chain.invoke({"language" : "english", "genre" : "Kid"})
    return response


print("Recommended Movies:" , movie_recommender())


