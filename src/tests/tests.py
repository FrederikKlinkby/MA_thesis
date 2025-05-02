from dotenv import find_dotenv, load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import retrieval.indexing as indexing
import chatbot.chatbot as bot


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def test():
    question = 'Hvem er næste kamp imod?'
    splits = indexing.split_data(chunk_size=1000, chunk_overlap=150)
    vectorstore = indexing.store_data(splits)
    bot.chatbot(vectorstore, question, 0.1, 5, 'mmr', OPENAI_API_KEY, show_retrieved=True)

test()