### File for creating chatbot

import os
from dotenv import find_dotenv, load_dotenv
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')

# Web pages to scrape and billetpriser file path
web_paths = ['https://www.fcm.dk/billetter/', 'https://www.fcm.dk/saesonkort/', 'https://billetsalg.fcm.dk/CMS?page=FAQ']
txt_file_path = r'billetpriser.txt'


def load_split_docs(web_paths=web_paths, txt_file_path=txt_file_path, chunk_size=1000, chunk_overlap=200, num_splits=False):
    # Load urls
    web_loader = WebBaseLoader(web_paths=web_paths) #Consider including argument that removes irrelevant text
    web_docs = web_loader.load() # Define docs

    # Load text file
    text_loader = TextLoader(txt_file_path)
    text_docs = text_loader.load()

    # Combine all docs
    all_docs = web_docs + text_docs

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # Split text

    splits = text_splitter.split_documents(all_docs)

    if num_splits:
        print(f"Number of splits: {len(splits)}")
        print(f"Number of splits: {len(web_docs)}")
        print(f"Number of splits: {len(text_docs)}")

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))
    return vectorstore


vectorstore = load_split_docs(chunk_size=1000, chunk_overlap=200)

system_prompt = (
    "Du er en assistent der skal svare på spørgsmål fra fans af fodboldklubben FC Midtjylland. "
    "Brug følgende kontekst til at besvare spørgsmålet. "
    "Hvis du ikke kender svaret, sig du ikke kender svaret "
    "Brug maks otte sætninger og svar præcist. "
    "\n\n"
    "Kontekst: {context}"
)

question = "Hvad koster en billet på faxe-kondi til en A-kamp?"

# Build and use RAG
def build_rag(t=0.5, vectorstore=vectorstore, k=3, system_prompt=system_prompt, question=question, search_type="similarity", show_retrieved=False):
    
    # Define LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=t)

    # Define retriever
    retriever = vectorstore.as_retriever(search_type=search_type, 
                                     search_kwargs={"k": k})

    if show_retrieved:
        retrieved_docs = retriever.invoke(question)
        print(retrieved_docs)

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question})
    return response["answer"]


response = build_rag(show_retrieved=True)
print(response)

# Fjern ubruglig tekst fra vector store. Evt finpuds billetpriser.txt for bedre retrieval! Måske mindre chunks for at forbedre svar på priser (Information overload ved for store chunks)