from astrapy import DataAPIClient
import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Get db-token and OpenAI key
db_token = os.getenv("db_token")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the client
client = DataAPIClient(db_token)
db = client.get_database_by_api_endpoint(
  "https://69bdc51f-37ed-492f-8423-1c7f943410ec-westus3.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db.list_collection_names()}")


# Query collection
collection = db.get_collection("fcm_rag_collection")

# Query to fetch only '_id' fields
documents = collection.find({}, projection={"_id": 1})  # Only fetch '_id' field

# Generate list of all ids
ids = []
for doc in documents:
    ids.append(doc["_id"])

""" To find page_content of single ids
doc = collection.find_one({'_id': ids[0]}) # Change to find other docs

if doc:
  print("Page content:", doc["page_content"])

else:
  print("Page content not found")

 """

###If a text string can be converted to a vector embedding, it is perhaps possible to perform similarity searches in the db:
query_text = "Hvad koster et sæsonkort?"
response = openai.embeddings.create(model='text-embedding-3-large', input=query_text)
query_embedding = response.data[0].embedding

print(query_embedding)
