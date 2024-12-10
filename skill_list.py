import os
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

# Retrieve the API key from env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_KEY")

# Initializing the language model with specific parameters
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0.7, api_key = GOOGLE_API_KEY)

# Example for prompt 
examples = [
    {"input" : "Looking for web developer with experience in HTML, CSS, JavaScript, and responsive web design. Familiarity with front-end framework like React is preferred.", "output": "Required Skills: HTML, CSS, JavaScript, Responsive web design, React"},
    {"input" : "Looking for a data analyst proficient in Python, SQL, and Power BI. Must have experience with Excel and good communication skills.", "output": "Required skills: Python, SQL, Power BI, Excel, Communication"}
]

# Example prompt template
example_prompt = PromptTemplate(
    input_variables=["input","output"], 
    template = "Input : {input} \n Output : {output}"
)

# Fewshot prompt template 
few_shot_prompt = FewShotPromptTemplate(examples = examples,
                                        example_prompt = example_prompt,
                                        prefix = "Extract relevant skills from the following job description:",
                                        suffix = "Input: {input}\n Output:",
                                        input_variables=["input"]
)

# Sample input to prompt
job_description = ("Java, Spring framework and Spring Boot, Hands-on experience of development on Microservices using REST API, Good knowledge of Design Patterns, Experience with source versioning using GIT,TFS\n Knowledge of Database (RDBMS, NoSQL), Couchbase knowledge will be a plus.")
formatted_prompt = few_shot_prompt.format(input = job_description)

traceable(project_name="skill_list")

# Response formatter
parser = StrOutputParser()

def output():
    # Invoke the formatted prompt 
    response = llm.invoke(formatted_prompt)
    answer = parser.parse(response.content)
    return answer


print(output())
