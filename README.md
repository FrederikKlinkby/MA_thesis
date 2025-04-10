# MA_thesis - Streamlining fan support: A case study on RAG implementation at FC Midtjylland
## This repo is used to test Datastax Astra databases and optimizing various parameter settings within the customized RAG

### Requirements
- python >= 3.11

### Clone repository and install dependencies:
git clone https://github.com/FrederikKlinkby/MA_thesis.git
pip install -r requirements.txt

### Create .env file and add API-keys and tokens
db_token = 'db_token'

OPENAI_API_KEY = 'OPENAI_API_KEY'

APPLICATION_TOKEN = 'APPLICATION_TOKEN'

langchain_api_key = 'langchain_api_key'

## Activate venv
.\venv\Scripts\Activate.ps1

### For full access
Set-ExecutionPolicy Unrestricted -Scope Process

## Use langflow API
python src\tests\test_flow.py "<query>"
