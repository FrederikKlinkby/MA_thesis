#File for indexing documents
import os
from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader
import bs4
from langchain_openai import OpenAIEmbeddings
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
    web_loader = WebBaseLoader(web_paths=web_paths, 
                               #bs_kwargs={"parse_only": bs4.SoupStrainer(class_=["collapsible-body"])}
                               ) #Consider including argument that removes irrelevant text (fx '\n').
    
    web_docs = web_loader.load() # Define web docs

    # Load text file
    text_loader = TextLoader(txt_file_path)
    text_docs = text_loader.load()

    # Combine all docs
    all_docs = web_docs + text_docs

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    splits = text_splitter.split_documents(all_docs)

    if num_splits:
        print(f"Number of total splits: {len(splits)}")
        print(f"Number of web_docs: {len(text_splitter.split_documents(web_docs))}")
        print(f"Number of text_docs: {len(text_splitter.split_documents(text_docs))}")
    return splits

split_data(num_splits=True)

# Function for storing the splits made in split_data() in a Chroma vector store
def store_data(splits):
     vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-large'))

     if not vectorstore:
         print('Vectorstore not created')
     return vectorstore

