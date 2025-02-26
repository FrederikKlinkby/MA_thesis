### File for creating chatbot

import os
from dotenv import find_dotenv, load_dotenv
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')

# Web pages to scrape
web_paths = ['https://www.fcm.dk/billetter/', 'https://www.fcm.dk/saesonkort/', 'https://billetsalg.fcm.dk/CMS?page=FAQ']


def load_split_docs(web_paths=web_paths, chunk_size=1000, chunk_overlap=200):
    # Load urls
    loader = WebBaseLoader(
    web_paths=web_paths
    ) #Consider including argument that removes irrelevant text

    docs = loader.load() # Define docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # Split text

    splits = text_splitter.split_documents(docs)
    print(f"Number of splits: {len(splits)}")
    #print(splits[-2])  # Print the first few splits to inspect their content

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

question = "Er cc contractor ståtribune?"

#Temperature of LLM
t = 0.9

def build_rag(t=t, vectorstore=vectorstore, k=3, system_prompt=system_prompt, question=question):
    
    # Define LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=t)

    # Define retriever
    retriever = vectorstore.as_retriever(k=k)

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
    print(response["answer"])
    return response["answer"]


response = build_rag()