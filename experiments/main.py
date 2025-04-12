# Main file for experimentation
import sys
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import inquirer
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chatbot import chatbot
import src.retrieval.indexing as indexing
import utils

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Dictionary of settings to loop through
settings = {'chunk sizes':      [100, 200, 400, 600, 1000],
            'temperature':      [0, 0.1, 0.25],
            'k':                [3, 5],
            'search types':     ['similarity', 'mmr'],
            'query expansion':  [True, False]
            }


# Count number of combinations in settings
def create_metrics_df() -> pd.DataFrame:
    total_combinations = (
        len(settings['chunk sizes']) * 
        len(settings['temperature']) * 
        len(settings['k']) * 
        len(settings['search types']) * 
        len(settings['query expansion'])
    )

    headers = ['Cosine Similarity', 'Faithfulness', 'Context Relevance']
    df = pd.DataFrame(index=range(total_combinations), columns=headers)
    return df


# Function for various testing
def test():
    splits = indexing.split_data(chunk_size=800, chunk_overlap=200)
    vectorstore = indexing.store_data(splits)
    question = "Hvornår starter billetsalget til næste kamp?"
    chatbot.chatbot(vectorstore, question, 0.1, 3, 'similarity', OPENAI_API_KEY)


# Full experiment function
async def full_experiment():
    print('Conducting full experiment. This takes a while...')

    question = "Hvornår starter billetsalget til næste kamp?"
    reference = "Billetsalget til FC Midtjyllands hjemmebanekampe åbner som regel 14 dage før kampene. Du kan se den specifikke åbningsdato for billetsalget på www.fcm.dk."

    #Create empty dataframe for eval metrics
    metrics = create_metrics_df()

    # Counter to track current row in metrics DataFrame
    row_counter = 0

    for chunk_size in settings['chunk sizes']:
        chunk_overlap = int(chunk_size*0.15) #Fixed chunk overlap to 15% of chunk size
        splits = indexing.split_data(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        vectorstore = indexing.store_data(splits)

        for t in settings['temperature']:
            for k in settings['k']:
                for search_type in settings['search types']:
                    for query_expansion in settings['query expansion']:
                        print(f'''Runs with following settings:
                                chunk size: {chunk_size}
                                chunk overlap: {chunk_overlap}
                                temp: {t}
                                k: {k}
                                search type: {search_type}
                                query expansion: {query_expansion}''')
                        
                        #Running chatbot
                        response = chatbot.chatbot(vectorstore, 
                                        question,
                                        openai_api_key=OPENAI_API_KEY,
                                        t=t,
                                        k=k,
                                        search_type=search_type, 
                                        q_expand=query_expansion,
                                        save_context=True)
                        
                        #Calculate metrics
                        eval_metrics = await utils.calculate_metrics(
                            reference=reference,
                            candidate=response['answer'],
                            question=question,
                            context=response['context'])

                        #Store metrics in df
                        metrics.loc[row_counter, ['Cosine Similarity', 'Faithfulness', 'Context Relevance']] = [
                            eval_metrics['Cosine Similarity'],
                            eval_metrics['Faithfulness'],
                            eval_metrics['Context Relevance']
                        ]

                        print('Run successful')

                        row_counter += 1
                metrics.to_excel('rag_experiment_metrics.xlsx')
                return

# Main menu
async def main():
    questions = [
        inquirer.List('Type of experiment',
                      message="What type of experiment would you like to run?",
                      choices=['Test', 'Full experiment', 'Exit'],
                      carousel=True),
    ]
    experiment_type = inquirer.prompt(questions)
    
    if experiment_type:
        experiment = experiment_type['Type of experiment']
        if experiment == 'Test':
            test()
        elif experiment == 'Full experiment':
            await full_experiment()
        else:
            print("Exit")
            exit()


if __name__ == '__main__':
    asyncio.run(main())