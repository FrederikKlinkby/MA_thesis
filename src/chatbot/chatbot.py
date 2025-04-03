# .py file for chatbot
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils


# chatbot function
def chatbot(vectorstore, question, t, k, search_type, openai_api_key, 
            q_expand=False, show_retrieved=False, save_context=False):
    """
    Execute a RAG (Retrieval-Augmented Generation) chatbot for answering questions.

    This function uses OpenAI's language model to provide contextually relevant answers
    to user questions by retrieving and incorporating relevant documents from a vectorstore.

    Args:
        vectorstore (FAISS): A vector database containing indexed documents
        question (str): The user's input question to be answered
        t (float): Temperature setting for the language model (controls randomness)
        k (int): Number of top-k documents to retrieve from the vectorstore
        search_type (str): Type of document retrieval ('similarity' or 'mmr')
        openai_api_key (str): API key for OpenAI services
        q_expand (bool, optional): Whether to use query expansion. Defaults to False.
        show_retrieved (bool, optional): Whether to print retrieved documents. Defaults to False.

    Returns:
        str: The generated answer to the input question

    Behavior:
        - Uses GPT-4o-mini model for generating responses
        - Supports two retrieval methods: similarity search and MMR (Maximal Marginal Relevance)
        - Optional query expansion to generate multiple paraphrased questions
        - Optional display of retrieved documents
        - Answers in Danish, tailored for FC Midtjylland fans

    Example:
        response = chatbot(
            vectorstore=my_vectorstore, 
            question="Hvornår spiller FC Midtjylland næste kamp?", 
            t=0.1, 
            k=3, 
            search_type='mmr', 
            openai_api_key='your_api_key'
        )
    """
    
    # Define LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=t)

    if search_type == "similarity": 
        # Define retriever with similarity search
        retriever = vectorstore.as_retriever(search_type="similarity", 
                                        search_kwargs={"k": k})
    else: 
        # Define retriever with MMR search with default values for lambda and fetch_k
        retriever = vectorstore.as_retriever(search_type="mmr", 
                                        search_kwargs={"k": k,
                                                       "lambda_mult": 0.5,
                                                       "fetch_k": 20})
    
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
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build RAG chain
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
        retrieved_docs = [doc.page_content for doc in retrieved_docs] #Retrieving only the page_content
        retrieved_docs = '\n'.join(retrieved_docs)
        print(retrieved_docs)

    if save_context:
        retrieved_docs = retriever.invoke(question)
        context = [doc.page_content for doc in retrieved_docs] #Retrieving only the page_content 
        return {
            "answer": response["answer"],
            "context": context
        }

    print(response["answer"])
    
    return response["answer"]


#response = chatbot(show_retrieved=True)

# Fjern ubruglig tekst fra vector store. Evt finpuds billetpriser.txt for bedre retrieval! Måske mindre chunks for at forbedre svar på priser (Information overload ved for store chunks)