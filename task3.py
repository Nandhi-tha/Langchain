#Install necessary packages
# pip install langchain-google-genai langchain langchain-community 

# Import necessary libraries
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Retrieve the API key from env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")

# Initializing the language model with specific parameters
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.5, api_key = GOOGLE_API_KEY)

# Prompt template to define given word
prompt_template = PromptTemplate.from_template("Define the term {word} briefly")
# Generate response by invoking the model
response = llm.invoke(prompt_template.format(word = "Energy"))

# Response formatter
parser = StrOutputParser()

answer = parser.parse(response.content)
print(answer)

# Function to generate proverb based on the given topic 
def proverb_generator():
    prompt_template = PromptTemplate.from_template("Write a proverb on the topic of {topic}")
    response = llm.invoke(prompt_template.format(topic = topic))
    proverb =  parser.parse(response.content)
    return proverb

# Example usage of the proverb generator
# topic = "Success"
topic = "Life"
print(f"Proverb on {topic}: \n",proverb_generator())

# Function to generate a study guide 
def study_guide(): 
    prompt_template = PromptTemplate.from_template("Provide a study plan for {topic} in {x} days for {user_type}")
    response = llm.invoke(prompt_template.format(topic = topic, x = x, user_type = user_type))
    guide =  parser.parse(response.content)
    return guide

# Example usage of the study guide
topic = "Python"
x = 5
user_type = "beginner"

print(f"Study Guide for {topic}: \n",study_guide())


# Function to translate the phrase into specified language
def translator():
    prompt_template = PromptTemplate.from_template("Translate the english phrase provided into {language}: {content}")
    response = llm.invoke(prompt_template.format(language = language, content = content))
    translated = parser.parse(response.content)
    return translated

# Example usage of the translator
language = "spanish"
content = "good morning"

# print(f"The translation for '{content}' in {language} is :\n",translator())
print(translator())

