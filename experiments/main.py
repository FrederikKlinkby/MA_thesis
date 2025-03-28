# Main file for experimentation
import sys
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import openpyxl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chatbot import chatbot
import src.retrieval.indexing as indexing
import utils
import inquirer

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')

# Dictionary of settings to loop through
settings = {'chunk sizes': [400, 600, 800, 1000],
            'chunk overlaps': [150, 250],
            'temperature': [0.1, 0.25],
            'k': [3, 4],
            'search types': ['similarity', 'mmr'],
            'query expansion': [True, False]
            }


# Count number of combinations in settings
def create_metrics_df() -> pd.DataFrame:
    total_combinations = (
        len(settings['chunk sizes']) * 
        len(settings['chunk overlaps']) * 
        len(settings['temperature']) * 
        len(settings['k']) * 
        len(settings['search types']) * 
        len(settings['query expansion'])
    )

    headers = ['ROUGE-2', 'Cosine Similarity' 'Faithfulness', 'Answer Relevance', 'Context Relevance']
    df = pd.DataFrame(index=range(total_combinations), columns=headers)
    return df


# Test function
def test():
    splits = indexing.split_data(chunk_size=800)
    vectorstore = indexing.store_data(splits)
    question = "Hvornår starter billetsalget til næste kamp?"
    test = chatbot.chatbot(vectorstore, question, 0.1, 3, 'similarity', OPENAI_API_KEY, save_context=True)
    print(test['context'])


# Full experiment function
def full_experiment():
    print('Conducting full experiment. This takes a while...')

    question = "Hvornår starter billetsalget til næste kamp?"
    reference = "Billetsalget til FC Midtjyllands hjemmebanekampe åbner som regel 14 dage før kampene. Du kan se den specifikke åbningsdato for billetsalget på www.fcm.dk."

    #Create empty dataframe for eval metrics
    metrics = create_metrics_df()

    # Counter to track current row in metrics DataFrame
    row_counter = 0

    for chunk_size in settings['chunk sizes']:
        for chunk_overlap in settings['chunk overlaps']:
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
                            eval_metrics = utils.run_calculate_metrics(
                                reference=reference,
                                candidate=response['answer'],
                                question=question,
                                context=response['context'])

                            #Store metrics in df
                            metrics.loc[row_counter, ['ROUGE-2', 'Cosine Similarity', 'Faithfulness']] = [
                                eval_metrics['ROUGE-2'],
                                eval_metrics['Cosine Similarity'],
                                eval_metrics['Faithfulness']
                            ]

                            print('Run successful')

                            row_counter += 1
                    metrics.to_excel('rag_experiment_metrics.xlsx')
                    return

# Main menu
def main():
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
            full_experiment()
        else:
            print("Exit")
            exit()


if __name__ == '__main__':
    main()