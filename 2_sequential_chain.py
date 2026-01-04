import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGHAIN_PROJECT"] = "SEQUENTIAL_CHAIN_DEMO"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
    api_key=os.getenv("GOOGLE_API_KEY"),
)  


parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

config = {
    "tags": ["llm app", "report generation", "summary generation"],
    "metadata": {"model": "gemini-2.5-flash", "temperature": 0.8}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)