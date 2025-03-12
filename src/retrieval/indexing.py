#File for indexing documents
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

# Load environment files and access OpenAI api key
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')

# Web pages to scrape
web_paths = ['https://www.fcm.dk/billetter/', 'https://www.fcm.dk/saesonkort/', 'https://billetsalg.fcm.dk/CMS?page=FAQ']

#billetpriser.txt file path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
txt_file_path = os.path.join(project_root, "data", "billetpriser.txt")


# Function for splitting data
def split_data(web_paths=web_paths, txt_file_path=txt_file_path, chunk_size=1000, chunk_overlap=200, num_splits=False):
    # Load urls
    web_loader = WebBaseLoader(web_paths=web_paths) #Consider including argument that removes irrelevant text
    web_docs = web_loader.load() # Define docs

    # Load text file
    text_loader = TextLoader(txt_file_path)
    text_docs = text_loader.load()

    # Combine all docs
    all_docs = web_docs + text_docs

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    splits = text_splitter.split_documents(all_docs)

    if num_splits:
        print(f"Number of splits: {len(splits)}")
        print(f"Number of splits: {len(web_docs)}")
        print(f"Number of splits: {len(text_docs)}")
    return splits


# Function for storing splits in vector store
def store_data(splits):
     vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY)) #experiment with other embeddings
     return vectorstore


