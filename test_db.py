from astrapy import DataAPIClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the value of 'db-token'
db_token = os.getenv("db_token")
# Initialize the client

client = DataAPIClient(db_token)
db = client.get_database_by_api_endpoint(
  "https://69bdc51f-37ed-492f-8423-1c7f943410ec-westus3.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db.list_collection_names()}")


## Query collection
collection = db.get_collection("fcm_rag_collection")
id = "79c12811634c4ee5a8f492c259ddb9f4" #change to see other chunks
query = {"_id": id}

# Find document
document = collection.find_one(query)

# Extract and print `page_content`
if document:
    print(document.get("page_content"))
else:
    print("Document not found")