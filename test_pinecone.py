import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv() # Load environment variables from .env

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
index_name = os.getenv("PINECONE_INDEX_NAME")

print(f"Attempting to connect with API Key: {pinecone_api_key[:5]}... (first 5 chars)")
print(f"Environment: {pinecone_env}")
print(f"Expected Index Name: {index_name}")

try:
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

    all_indexes = pc.list_indexes()
    print(f"Indexes found: {all_indexes}")

    if index_name in all_indexes:
        print(f"Success! Index '{index_name}' found in list_indexes().")
    else:
        print(f"Error: Index '{index_name}' NOT found in list_indexes().")

    # Try to describe the index directly
    try:
        desc = pc.describe_index(index_name)
        print(f"Successfully described index '{index_name}': {desc}")
    except Exception as e:
        print(f"Could not describe index '{index_name}': {e}")

except Exception as e:
    print(f"Failed to initialize Pinecone client or list indexes: {e}")