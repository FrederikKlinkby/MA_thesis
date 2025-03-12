# main.py file for chatbot
import sys
import os
from dotenv import find_dotenv, load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from retrieval import indexing
import utils

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')



system_prompt = (
    '''
    Du er en assistent der skal svare på spørgsmål fra fans af fodboldklubben FC Midtjylland.
    Brug følgende kontekst til at besvare spørgsmålet.
    Hvis du ikke kender svaret, sig du ikke kender svaret.
    Brug gerne lange svar. Svar præcist.


    Kontekst: {context}
    '''
    )

splits = indexing.split_data()
vectorstore = indexing.store_data(splits)

question = "Jeg har lagt mine billetter i kurven, men kan ikke gå til betaling."

# Build and use RAG
def chatbot(t=0.1, vectorstore=vectorstore, k=3, system_prompt=system_prompt, question=question, search_type="similarity", q_expand=False, show_retrieved=False):
    
    # Define LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=t)

    # Define retriever
    retriever = vectorstore.as_retriever(search_type=search_type, 
                                     search_kwargs={"k": k})


    # If query expansion
    if q_expand:
        # Get paraphrased question
        optimized_questions = utils.query_optim(question)
        # Retrieve documents for each optimized question
        all_retrieved_docs = []
        for opt_question in optimized_questions:
            retrieved_docs = retriever.invoke(opt_question)
            all_retrieved_docs.extend(retrieved_docs)
            if show_retrieved:
                print(f"Retrieved for optimized query '{opt_question}': {retrieved_docs}")
    else:
        # Standard retrieval without optimization
        all_retrieved_docs = retriever.invoke(question)
        if show_retrieved:
            print(f"Retrieved: {all_retrieved_docs}")


    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    # Generate response
    response = rag_chain.invoke({"input": question})
    return response["answer"]


response = chatbot(show_retrieved=True, q_expand=True)
print(response)

# Fjern ubruglig tekst fra vector store. Evt finpuds billetpriser.txt for bedre retrieval! Måske mindre chunks for at forbedre svar på priser (Information overload ved for store chunks)