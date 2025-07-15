from pinecone import Pinecone, ServerlessSpec
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if "arxiv-search" in pc.list_indexes().names():
    pc.delete_index("arxiv-search")
pc.create_index(
    name="arxiv-search",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["arxiv_db"]
db["papers"].drop()
mongo_client.close()
