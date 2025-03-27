# Main file for experimentation
import sys
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import openpyxl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chatbot import chatbot
import src.retrieval.indexing as indexing
import inquirer
import sacrebleu
from rouge_score import rouge_scorer
from ragas import evaluate #See following for available metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/#agents-or-tool-use-cases

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
def count_combinations_and_create_df() -> pd.DataFrame:
    total_combinations = (
        len(settings['chunk sizes']) * 
        len(settings['chunk overlaps']) * 
        len(settings['temperature']) * 
        len(settings['k']) * 
        len(settings['search types']) * 
        len(settings['query expansion'])
    )

    headers = ['BLEU', 'ROUGE-2', 'Faithfulness', 'Answer Relevance', 'Context Relevance']
    df = pd.DataFrame(index=range(total_combinations), columns=headers)
    return df


# Evaluation function
def calculate_metrics(reference, candidate):
    # BLEU Score
    bleu = sacrebleu.metrics.BLEU(effective_order=True) # effective_order = TRUE -> Prevents penalizing the score for n-grams longer than those in the reference
    bleu_score = bleu.sentence_score(candidate, [reference]).score
    
    # ROUGE Scores
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(reference, candidate)
    
    return {
        'BLEU': bleu_score,
        #'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        #'ROUGE-L': rouge_scores['rougeL'].fmeasure
    }

# Test function
def Test():
    df = count_combinations_and_create_df()
    df.to_excel('test.xlsx')


# Full experiment function
def Full_experiment():
    print('Conduncting full experiment. This takes a while...')

    question = "Hvornår åbner billetsalget til næste kamp?"
    reference = "Billetsalget til FC Midtjyllands hjemmebanekampe åbner som regel 14 dage før kampene. Du kan se den specifikke åbningsdato for billetsalget på www.fcm.dk."

    #Create empty dataframe for eval metrics
    metrics = count_combinations_and_create_df()

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
                                            q_expand=query_expansion
                                            )
                            
                            #Calculate metrics
                            eval_metrics = calculate_metrics(reference, response)

                            #Store metrics in df
                            metrics.loc[row_counter, ['BLEU', 'ROUGE-2']] = [
                                eval_metrics['BLEU'], 
                                eval_metrics['ROUGE-2']
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
            Test()
        elif experiment == 'Full experiment':
            Full_experiment()
        else:
            print("Exit")
            exit()


if __name__ == '__main__':
    main()