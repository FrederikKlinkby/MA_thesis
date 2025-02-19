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


# Converting text query to vector embedding
query_text = "Hvad er aldersgrænsen for børn?"
response = openai.embeddings.create(model='text-embedding-3-large', input=query_text)
query_embedding = response.data[0].embedding

# Perform similarity search
results = collection.find(
    sort={"$vector": query_embedding},
    limit=3, #query top-k results
)

# Print results of similarity search
for doc in results:
   content = doc.get("page_content") or doc.get('content', 'Content not found') #In some JSON data points, the key is called 'page_content', in others just 'content'
   print(content)
   print(100*"_")
   
