#utils for chatbot
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


#From https://python.langchain.com/v0.1/docs/use_cases/query_analysis/techniques/expansion/
class ParaphrasedQuery(BaseModel):
    """Du har lavet en query expansion af et spørgsmål"""

    paraphrased_query: list[str] = Field(
        ...,
        description="Liste af omformulerede spørgsmål, mindst 3 versioner",
    )


# Function for query optimisation
def query_optim(question, print_paraphrased_questions=False) -> list:
    system = """Du er ekspert i at konvertere brugerspørgsmål til database queries.
    Du har adgang til information om billetter, sæsonkort, FAQ og billetpriser på FC Midtjyllands hjemmeside.
    
    Lav en query expansion. Hvis der er flere måde at formulere brugerens spørgsmål på \
    eller der er synonymer for nøgleord i spørgsmålet, returnér flere versioner af spørgsmålet \
    med forskellige fraseringer.

    Hvis der er forkortelser eller ord du ikke kender, forsøg ikke at omformulere dem.
    
    Returnér mindst 3 versioner af spørgsmålet som en liste af strenge."""

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

    # Invoke and get paraphrased queries
    output = query_analyzer.invoke({"question": question})
    paraphrased_queries = output[0].paraphrased_query
    
    if print_paraphrased_questions:
        # Print the original and reformulated questions
        print("\nOriginal question:", question)
        print("Reformulated questions:")
        print(paraphrased_queries)
        for i, reformulation in enumerate(paraphrased_queries, 1):
            print(f"{i}. {reformulation}")

    paraphrased_queries = " ".join(paraphrased_queries)
    return paraphrased_queries