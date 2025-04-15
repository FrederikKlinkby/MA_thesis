import sys
import os
import asyncio
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chatbot import chatbot
import src.retrieval.indexing as indexing
import utils
from main import OPENAI_API_KEY

# Create df
def create_metrics_df() -> pd.DataFrame:
    headers = ['Cosine Similarity', 'Faithfulness', 'Context Relevance', 'Answer Relevance']
    df = pd.DataFrame(index=range(10), columns=headers)
    return df

#Test ragas
async def test_ragas():
    # Creating vector store with optimised parameters
    splits = indexing.split_data(chunk_size=1000, chunk_overlap=150)
    vectorstore = indexing.store_data(splits)

    # Defining question and reference
    question = "Jeg har lagt mine billetter i kurven, men kan ikke gå til betaling."
    reference = "I kurven skal du personliggøre dine billetter ved at tildele hver billet en bruger. Det gør du ved at indtaste dit brugernavn (e-mailadresse) på hver billet i kurven. Du kan også klikke på 'Tildel bruger' og derefter klikke på dit navn i dropdown-menuen. \nNår du har personliggjort alle billetter i kurven, kan du gå videre til betalingen. \nPå dette link finder du en videoguide til hvordan du køber billetter - Klik her for at åbne videoguiden (https://www.youtube.com/watch?v=W7QxKP21vrE)"
    
    # Defining metrics df
    metrics = create_metrics_df()

    # Get response
    response = chatbot.chatbot(vectorstore, question, 0.1, 5, 'mmr', OPENAI_API_KEY, save_context=True)
    print(response['answer'])

    #For loop for testing reliability of RAGAS
    for i in range(10):
        #Calculate metrics
        eval_metrics = await utils.calculate_metrics(
            reference=reference,
            candidate=response['answer'],
            question=question,
            context=response['context'])
        
        #Store metrics in df
        metrics.loc[i, ['Cosine Similarity', 'Faithfulness', 'Context Relevance', 'Answer Relevance']] = [
            eval_metrics['Cosine Similarity'],
            eval_metrics['Faithfulness'],
            eval_metrics['Context Relevance'],
            eval_metrics['Answer Relevance']
        ]
    metrics.to_excel('test_ragas.xlsx')
    return

asyncio.run(test_ragas())