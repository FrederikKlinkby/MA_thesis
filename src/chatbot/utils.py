#utils
import os
from dotenv import find_dotenv, load_dotenv
from langchain_core.pydantic_v1 import Field
from pydantic import BaseModel
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment files and access OpenAI api key
#dotenv_path = find_dotenv()
#load_dotenv(dotenv_path)

#OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')

#Fra https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/expansion/
class ParaphrasedQuery(BaseModel):
    """Du har lavet en query expansion af et spørgsmål"""

    paraphrased_query: str = Field(
        ...,
        description="En unik omformulering af det originale spørgsmål.",
    )



# Function for query optimisation
def query_optim(question):
    system = """Du er ekspert i at konvertere brugerspørgsmål til database queries.
    Du har adgang til information om billetter, sæsonkort, FAQ og billetpriser på FC Midtjyllands hjemmside.
    
    Lav en query expansion. Hvis der er flere måde at formulere brugerens spørgsmål på \
    eller der er synonymer for nøgleord i spørgsmålet, returnér flere versioner af spørgsmålet \
    med forskellige fraseringer.

    Hvis der er forkortelser eller ord du ikke kender, forsøg ikke at omformulere dem.
    
    Returnér 3 versioner af spørgsmålet."""

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", question),
    ]
    )
    
    # Define LLM and query analyzer
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm_with_tools = llm.bind_tools([ParaphrasedQuery])
    query_analyzer = prompt | llm_with_tools | PydanticToolsParser(tools=[ParaphrasedQuery])

    # Invoke
    query_analyzer.invoke({"question": question})


query_optim("Hvordan får jeg billet?")