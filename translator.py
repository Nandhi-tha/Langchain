import os
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")


# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, api_key=GOOGLE_API_KEY)

# Define the example prompt template
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\n Output: {output}"
)

# Few-shot examples
examples = [
    {"input": "Translate 'Hello' to French.", "output": "Bonjour"},
    {"input": "Translate 'Goodbye' to French.", "output": "Au revoir"}
]

# Define the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Translate the following English text to French:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

# Format the prompt
formatted_prompt = few_shot_prompt.format(input="Good morning")
print("Generated Prompt:\n", formatted_prompt)  
response = llm.invoke(formatted_prompt)
print("Response:\n", response.content)








