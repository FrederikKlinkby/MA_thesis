#File for indexing documents
import os
from dotenv import find_dotenv, load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# Load environment files and access OpenAI api key
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')

# Web pages to scrape
web_paths = ['https://www.fcm.dk/billetter/', 'https://www.fcm.dk/saesonkort/', 'https://billetsalg.fcm.dk/CMS?page=FAQ']

# Function for splitting data
def split_data(web_paths=web_paths, chunk_size=1000, chunk_overlap=200, num_splits=False):

    # Custom parsing function for complex websites
    def custom_parse_website(url):
        # Send request to website
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # For this structure, we'll extract text from collapsible sections
        content = []
        
        collapsible_bodies = soup.find_all('div', class_='collapsible-body')
        for body in collapsible_bodies:
            body_text = body.get_text(strip=True)
            if body_text:
                content.append(body_text)
        
        return ' '.join(content)

    # Custom document loader that uses the custom parsing function
    class CustomWebLoader(WebBaseLoader):
        def parse(self, html, **kwargs):
            # Override the default parsing method
            return [custom_parse_website(url) for url in self.web_paths]

    # Use the custom web loader
    web_loader = CustomWebLoader(web_paths=web_paths)
    web_docs = web_loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(web_docs)

    if num_splits:
        print(f"Number of web_docs: {len(text_splitter.split_documents(web_docs))}")
    
    return splits


# Function for storing the splits made in split_data() in a Chroma vector store
def store_data(splits):
     vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-large'))

     if not vectorstore:
         print('Vectorstore not created')
     return vectorstore

