import os
import re
import requests
import arxiv
from pinecone import Pinecone
from pymongo import MongoClient, operations
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import time

MAX_ITEMS = 30000
RECENT_ITEMS = 27000
CLASSIC_ITEMS = 3000
BATCH_SIZE = 200

CATEGORIES = (
    "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV "
    "OR cat:stat.ML OR cat:cs.IR OR cat:cs.NE OR cat:cs.RO"
)


def cleanup_old_items(collection, index, max_items=MAX_ITEMS):
    current_count = collection.count_documents({})
    if current_count > max_items:
        docs_to_delete = list(
            collection.find({"type": {"$ne": "classic"}})
            .sort("ingested_at", 1)
            .limit(current_count - max_items)
        )
        delete_ids = [doc["_id"] for doc in docs_to_delete]
        if delete_ids:
            print(f"Deleting {len(delete_ids)} old non-classic papers.")
            pinecone_batch_size = 1000
            print(f"Deleting from Pinecone in batches of {pinecone_batch_size}...")
            for i in tqdm(
                range(0, len(delete_ids), pinecone_batch_size),
                desc="Pinecone delete batches",
            ):
                batch = delete_ids[i : i + pinecone_batch_size]
                index.delete(ids=batch)
            print("Deleting from MongoDB...")
            collection.delete_many({"_id": {"$in": delete_ids}})
        else:
            print("No old non-classic papers to delete.")


def fetch_classic_ids_from_github(n):
    awesome_lists = [
        "https://raw.githubusercontent.com/terryum/awesome-deep-learning-papers/master/README.md",
        "https://raw.githubusercontent.com/ChristosChristofidis/awesome-deep-learning/master/README.md",
        "https://raw.githubusercontent.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code/master/README.md",
        "https://raw.githubusercontent.com/keon/awesome-nlp/master/README.md",
        "https://raw.githubusercontent.com/jtoy/awesome-tensorflow/master/README.md",
        "https://raw.githubusercontent.com/kjw0612/awesome-deep-vision/master/README.md",
    ]
    arxiv_pattern = re.compile(
        r"https://arxiv.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)"
    )
    classic_ids = set()
    print("Fetching classic paper IDs from GitHub 'Awesome' lists...")
    for url in tqdm(awesome_lists, desc="Scanning Awesome Lists"):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            text = response.text
            found_ids = arxiv_pattern.findall(text)
            classic_ids.update(found_ids)
        except requests.RequestException as e:
            print(f"\nCould not fetch {url}: {e}")
    print(f"Found {len(classic_ids)} unique IDs from GitHub.")
    return list(classic_ids)[:n]


def fetch_classic_ids_from_semantic_scholar(n):
    print("Fetching classic paper IDs from Semantic Scholar by citation count...")
    classic_ids = set()
    offset = 0
    limit = 100
    max_retries = 3
    query = f"({CATEGORIES.replace('cat:', '').replace(' OR ', ' OR ')})"
    with tqdm(total=n, desc="Scanning Semantic Scholar") as pbar:
        while len(classic_ids) < n:
            url = (
                "https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={query}&fields=externalIds,citationCount"
                f"&limit={limit}&offset={offset}&sort=citationCount"
            )
            retries = 0
            data = {}
            while retries < max_retries:
                try:
                    resp = requests.get(url)
                    if resp.status_code == 429:
                        print("Rate limit hit, sleeping for 60 seconds...")
                        time.sleep(60)
                        retries += 1
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    print(f"Error fetching from Semantic Scholar: {e}. Retrying...")
                    retries += 1
                    time.sleep(5)
            if not data or "data" not in data or not data["data"]:
                print("No more data from Semantic Scholar.")
                break
            for item in data["data"]:
                arxiv_id = item.get("externalIds", {}).get("ArXiv")
                if arxiv_id and arxiv_id not in classic_ids:
                    classic_ids.add(arxiv_id)
                    pbar.update(1)
                    if len(classic_ids) >= n:
                        break
            if len(classic_ids) >= n:
                break
            offset += limit
            if offset > 9900:
                print("Reached Semantic Scholar API result limit.")
                break
            time.sleep(2)
    print(f"Found {len(classic_ids)} unique IDs from Semantic Scholar.")
    return list(classic_ids)


def fetch_arxiv_metadata(arxiv_id):
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(arxiv.Client().results(search))
        clean_id = paper.entry_id.split("/")[-1]
        return {
            "_id": clean_id,
            "title": paper.title,
            "summary": paper.summary,
            "authors": ", ".join([a.name for a in paper.authors]),
            "pdf_url": paper.pdf_url,
        }
    except Exception as e:
        print(f"\nError fetching metadata for {arxiv_id}: {e}")
        return None


def ingest_classic_papers(collection, index, processed_ids, limit=CLASSIC_ITEMS):
    print("Starting hybrid ingestion of classic papers...")
    github_ids = fetch_classic_ids_from_github(limit)
    scholar_ids = fetch_classic_ids_from_semantic_scholar(limit // 2)
    combined_ids = set(github_ids) | set(scholar_ids)
    print(f"Found a combined total of {len(combined_ids)} unique classic paper IDs.")
    classic_ids_to_fetch = list(combined_ids)
    new_classic_ids = [cid for cid in classic_ids_to_fetch if cid not in processed_ids]
    if not new_classic_ids:
        print("No new classic papers to ingest.")
        return processed_ids
    print(f"Ingesting {len(new_classic_ids)} new classic papers...")
    classic_ingested = 0
    vectors_batch, mongo_batch = [], []
    for arxiv_id in tqdm(new_classic_ids, desc="Ingesting classic papers"):
        meta = fetch_arxiv_metadata(arxiv_id)
        if not meta:
            continue
        try:
            title = meta.get("title", "")
            summary = meta.get("summary", "")
            cleaned_summary = summary.replace("\n", " ")
            text_to_embed = f"Title: {title}. Abstract: {cleaned_summary}"
            embedding = model.encode(text_to_embed).tolist()
            doc_id = meta["_id"]
            vectors_batch.append((doc_id, embedding))
            mongo_op = operations.UpdateOne(
                {"_id": doc_id},
                {
                    "$set": {
                        "title": title,
                        "summary": cleaned_summary,
                        "authors": meta.get("authors", ""),
                        "pdf_url": meta.get("pdf_url", ""),
                        "ingested_at": datetime.now(timezone.utc),
                        "type": "classic",
                    }
                },
                upsert=True,
            )
            mongo_batch.append(mongo_op)
            classic_ingested += 1
            processed_ids.add(doc_id)
            if len(vectors_batch) >= BATCH_SIZE:
                index.upsert(vectors=vectors_batch)
                collection.bulk_write(mongo_batch)
                vectors_batch, mongo_batch = [], []
        except Exception as e:
            print(f"\nFailed to process classic paper {arxiv_id}: {e}")
    if vectors_batch:
        index.upsert(vectors=vectors_batch)
        collection.bulk_write(mongo_batch)
    print(
        f"Classic papers ingestion complete! Processed {classic_ingested} new papers."
    )
    return processed_ids


print("Initializing connections and models...")
load_dotenv()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("arxiv-search")
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["arxiv_db"]
collection = db["papers"]
print("Ensuring database index exists for 'ingested_at'...")
collection.create_index([("ingested_at", 1)])
print("Initialization complete.")
print("-" * 50)

cleanup_old_items(collection, index, MAX_ITEMS)
processed_ids = set(doc["_id"] for doc in collection.find({}, {"_id": 1}))

processed_ids = ingest_classic_papers(collection, index, processed_ids, CLASSIC_ITEMS)
print("-" * 50)

client = arxiv.Client(page_size=500, delay_seconds=5, num_retries=5)
vectors_batch = []
mongo_batch = []
papers_ingested_count = 0
weeks_to_go_back = 0
print("Starting nightly ingestion of recent papers...")
with tqdm(total=RECENT_ITEMS, desc="Ingesting recent papers") as pbar:
    while papers_ingested_count < RECENT_ITEMS:
        end_date = datetime.now(timezone.utc) - timedelta(weeks=weeks_to_go_back)
        start_date = end_date - timedelta(days=7)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")
        print(f"\nFetching papers from {start_date_str} to {end_date_str}...")
        query_for_week = (
            f"({CATEGORIES}) AND submittedDate:[{start_date_str} TO {end_date_str}]"
        )
        search = arxiv.Search(
            query=query_for_week,
            max_results=5000,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        try:
            results = client.results(search)
            found_in_chunk = 0
            for paper in results:
                doc_id = paper.entry_id.split("/")[-1]
                if doc_id in processed_ids:
                    continue
                title = paper.title or ""
                summary = paper.summary or ""
                cleaned_summary = summary.replace("\n", " ")
                text_to_embed = f"Title: {title}. Abstract: {cleaned_summary}"
                embedding = model.encode(text_to_embed).tolist()
                vectors_batch.append((doc_id, embedding))
                mongo_op = operations.UpdateOne(
                    {"_id": doc_id},
                    {
                        "$set": {
                            "title": title,
                            "summary": cleaned_summary,
                            "authors": ", ".join([a.name for a in paper.authors]),
                            "pdf_url": paper.pdf_url,
                            "ingested_at": datetime.now(timezone.utc),
                            "type": "recent",
                        }
                    },
                    upsert=True,
                )
                mongo_batch.append(mongo_op)
                processed_ids.add(doc_id)
                papers_ingested_count += 1
                found_in_chunk += 1
                pbar.update(1)
                if len(vectors_batch) >= BATCH_SIZE:
                    index.upsert(vectors=vectors_batch)
                    collection.bulk_write(mongo_batch)
                    vectors_batch, mongo_batch = [], []
                if papers_ingested_count >= RECENT_ITEMS:
                    break
            if found_in_chunk == 0:
                print("Found no new papers in this period. Searching further back.")
        except Exception as e:
            print(
                f"\nCould not process results for week {start_date_str}-{end_date_str}: {e}"
            )
        weeks_to_go_back += 1
        if weeks_to_go_back > 156:
            print("\nSearched 3 years back, stopping.")
            break
if vectors_batch:
    index.upsert(vectors=vectors_batch)
    collection.bulk_write(mongo_batch)
print(f"\nRecent papers ingestion complete! Processed {papers_ingested_count} papers.")
print("-" * 50)

cleanup_old_items(collection, index, MAX_ITEMS)
print(
    f"\nData ingestion complete! Total papers in DB: {collection.count_documents({})}"
)
mongo_client.close()
