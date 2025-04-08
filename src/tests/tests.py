from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import os
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import Faithfulness
import asyncio
from ragas.integrations.langchain import LangchainLLMWrapper

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

async def test():
    # Create the base LLM
    base_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY
    )
    
    # Define evaluator llm
    evaluator_llm = LangchainLLMWrapper(langchain_llm=base_llm)
    
    sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles.", "Johnny Madsen er født i Thyborøn engang i 1950'erne!!!"
        ]
    )
    
    scorer = Faithfulness(llm=evaluator_llm)
    score = await scorer.single_turn_ascore(sample)
    print(f"Faithfulness score: {score}")
    return score

asyncio.run(test())