import os
import numpy as np
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
import asyncio
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference #See following for available metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/#agents-or-tool-use-cases
from ragas.integrations.langchain import LangchainLLMWrapper

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Function for ROUGE Score
def calculate_rouge_score(reference, candidate):
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) #Perhaps remove rouge1 and rougeL
    rouge_scores = rouge_scorer_instance.score(reference, candidate)
    return rouge_scores

# Functions necessary for Cosine Similarity 
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

# Function for RAGAS scorers with Danish prompts
async def define_scorer_with_danish_prompts() -> dict:
    # Convert prompts into Danish
    # Create the base LLM
    base_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY
        )

    # Define evaluator llm
    evaluator_llm = LangchainLLMWrapper(langchain_llm=base_llm)

    # Faithfulness score
    scorer_f = Faithfulness(llm=evaluator_llm)

    # Context relevancy
    scorer_cr = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    #Translate prompts into Danish (Faithfulness)
    danish_prompts_f = await scorer_f.adapt_prompts(language='danish', llm=evaluator_llm)
    scorer_f.set_prompts(**danish_prompts_f)

    #Translate prompts into Danish (Faithfulness)
    danish_prompts_cr = await scorer_cr.adapt_prompts(language='danish', llm=evaluator_llm)
    scorer_cr.set_prompts(**danish_prompts_cr)
    return {
        "faithfulness_scorer": scorer_f,
        "context_relevance_scorer": scorer_cr
    }

# Get scorer objects in memory
scorer_dict = asyncio.run(define_scorer_with_danish_prompts())
scorer_f = scorer_dict["faithfulness_scorer"]
scorer_cr = scorer_dict["context_relevance_scorer"]

# Function for calculating RAGAs
async def calculate_ragas(candidate, question, context) -> dict:
    
    sample = SingleTurnSample(
        user_input=question,
        response=candidate,
        retrieved_contexts=context
    )
    
    # Faithfulness score
    faithfulness_score = await scorer_f.single_turn_ascore(sample)

    # Context relevancy
    context_relevance_score = await scorer_cr.single_turn_ascore(sample)

    return {
        "faithfulness_score": faithfulness_score,
        "context_relevance_score": context_relevance_score
    }

# Evaluation function
async def calculate_metrics(reference, candidate, question, context) -> dict:
    # Start RAGAs metrics early while ROUGE and Cosine similarity are calculated
    task = asyncio.create_task(calculate_ragas(candidate, question, context))
    
    # ROUGE Scores
    rouge_scores = calculate_rouge_score(reference, candidate)
    print(f'ROUGE-2 score: {rouge_scores['rouge2'].fmeasure}')

    # Cosine Similarity
    cosine_sim = calculate_text_similarity(reference, candidate)
    print(f'Cosine similarity: {cosine_sim}')

    # Await RAGAs metrics
    ragas_scores = await task

    #Faithfulness and Context Relevance
    faithfulness_score = ragas_scores['faithfulness_score']
    print(f'Faithfulness score: {faithfulness_score}')
    context_relevance_score = ragas_scores['context_relevance_score']
    print(f'Context relevance score: {context_relevance_score}')
    
    return {
        #'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        #'ROUGE-L': rouge_scores['rougeL'].fmeasure,
        'Cosine Similarity': cosine_sim,
        'Faithfulness': faithfulness_score,
        'Context Relevance': context_relevance_score
    }