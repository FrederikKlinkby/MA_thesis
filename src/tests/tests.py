import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.indexing import web_paths, txt_file_path, split_data, store_data, retrieve_docs


question = 'Hvordan får jeg billet?'

splits = split_data(web_paths, txt_file_path, chunk_size=1000, chunk_overlap=200)
vectorstore = store_data(splits)


#Function from 28-03-2025
#def split_data(web_paths=web_paths, txt_file_path=txt_file_path, chunk_size=1000, chunk_overlap=200, num_splits=False):
#    # Load urls
#    web_loader = WebBaseLoader(web_paths=web_paths, 
#                               #bs_kwargs={"parse_only": bs4.SoupStrainer(class_=["collapsible-body"])}
#                               ) #Consider including argument that removes irrelevant text (fx '\n').
#    
#    web_docs = web_loader.load() # Define web docs
#
#    # Load text file
#    text_loader = TextLoader(txt_file_path)
#    text_docs = text_loader.load()
#
#    # Combine all docs
#    all_docs = web_docs + text_docs
#
#    # Split text
#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#
#    splits = text_splitter.split_documents(all_docs)
#
#    if num_splits:
#        print(f"Number of total splits: {len(splits)}")
#        print(f"Number of web_docs: {len(text_splitter.split_documents(web_docs))}")
#        print(f"Number of text_docs: {len(text_splitter.split_documents(text_docs))}")
#    return splits