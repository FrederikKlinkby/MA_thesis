import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.indexing import web_paths, txt_file_path, split_data, store_data, retrieve_docs


question = 'Hvordan får jeg billet?'

splits = split_data(web_paths, txt_file_path, chunk_size=1000, chunk_overlap=200)
vectorstore = store_data(splits)
retrieve_docs(vectorstore, question, q_expand=False, show_retrieved=True)