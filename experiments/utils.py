import os
import numpy as np
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import asyncio
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference #See following for available metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/#agents-or-tool-use-cases

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI-API-KEY')


# Function for Cosine Similarity 
def get_embedding(text):
    # Get client
    client = OpenAI()

    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-large"
    )
    embedding = response.data[0].embedding
    return np.array(embedding)


def cosine_similarity(vec1, vec2):
    # Ensure vectors are numpy arrays
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Calculate cosine similarity
    if magnitude1 and magnitude2:
        return dot_product / (magnitude1 * magnitude2)
    else:
        return 0.0


def calculate_text_similarity(text1, text2):
    # Generate embeddings
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    
    # Calculate and return cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity


# Evaluation function
def calculate_metrics(reference, candidate, question, context) -> dict:
    # ROUGE Scores
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_instance.score(reference, candidate)

    # Cosine Similarity
    cosine_sim = calculate_text_similarity(reference, candidate)

    # RAGAs metrics
    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,  # low temperature for more deterministic responses
        api_key=OPENAI_API_KEY
    )

    sample = SingleTurnSample(
        user_input=question,
        response = candidate,
        retrieved_contexts = context
    )
    
    ## Faithfulness
    F = Faithfulness(llm=evaluator_llm)
    faithfulness_score = F.score(sample)


    ## Answer Relevance

    ## Context Relevance
    
    return {
        #'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        #'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'Cosine Similarity': cosine_sim,
        'Faithfulness': faithfulness_score
    }

def run_calculate_metrics(reference, candidate, question, context):
    return asyncio.run(calculate_metrics(reference, candidate, question, context))