from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import os
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import Faithfulness
import asyncio
from ragas.integrations.langchain import LangchainLLMWrapper
from ragas.llms import llm_factory

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
    scorer = Faithfulness(llm=evaluator_llm)

    #Translate prompts into Danish
    danish_prompts = await scorer.adapt_prompts(language='danish', llm=evaluator_llm)
    scorer.set_prompts(**danish_prompts)
    
    sample = SingleTurnSample(
        user_input="Hvornår starter billetsalget til næste kamp?",
        response="Billetsalget starter typisk 14 dage før kampen",
        retrieved_contexts=[
            "Billetsalget til FC Midtjyllands hjemmebanekampe åbner som regel 14 dage før kampene. Du kan se den specifikke åbningsdato for billetsalget på www.fcm.dk."
        ]
    )
    

    #score = await scorer.single_turn_ascore(sample)
    #print(f"Faithfulness score: {score}")
    print(danish_prompts)
    return #score

asyncio.run(test())