# main.py file for chatbot
import sys
import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import retrieval.indexing as indexing
import utils

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')
langchain_api_key = os.getenv('langchain_api_key')


# Define system prompt
system_prompt = (
    '''
    Du er en assistent der skal svare på spørgsmål fra fans af fodboldklubben FC Midtjylland.
    Brug følgende kontekst til at besvare spørgsmålet.
    Hvis du ikke kender svaret, sig du ikke kender svaret.
    Brug gerne lange svar. Svar præcist.


    Kontekst: {context}
    '''
    )

splits = indexing.split_data(chunk_size=600, chunk_overlap=150, num_splits=True)
vectorstore = indexing.store_data(splits)

question = "Køb af billetter og gavekort på billetsalg.fcm.dk"

# Build and use RAG
def chatbot(t=0.1, vectorstore=vectorstore, k=3, system_prompt=system_prompt, question=question, 
            search_type="similarity", openai_api_key=OPENAI_API_KEY, q_expand=False, show_retrieved=False):
    
    # Define LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=t)

    # Define retriever
    retriever = vectorstore.as_retriever(search_type=search_type, 
                                     search_kwargs={"k": k})
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # If query expansion is chosen
    if q_expand:
        # Get paraphrased questions
        optimised_question = utils.query_optim(question) # Returns a list of three questions
        print(f"Optimised question: {optimised_question}")

        # Generate response
        response = rag_chain.invoke({"input": optimised_question})
    
    else: #Otherwise
        response = rag_chain.invoke({"input": question})

    if show_retrieved:
        retrieved_docs = retriever.invoke(question)
        retrieved_docs = retrieved_docs[0].page_content #Retrieving only the page_content
        print(retrieved_docs)

    print(response["answer"])
    
    return response["answer"]


response = chatbot(show_retrieved=True)

# Fjern ubruglig tekst fra vector store. Evt finpuds billetpriser.txt for bedre retrieval! Måske mindre chunks for at forbedre svar på priser (Information overload ved for store chunks)