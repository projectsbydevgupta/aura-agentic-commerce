import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
import os
import sys
import json
import csv
import argparse
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.search.embeddings import get_embedding_engine
from src.search.indexing import setup_indexes

def sanitize(row: dict) -> dict:
    """Cleans and casts catalog document types."""
    return {
        "product_id": str(row.get("product_id", "")).strip(),
        "original_id": int(float(row.get("original_id", 0) or 0)),
        "name": str(row.get("name", "")).strip(),
        "brand": str(row.get("brand", "")).strip(),
        "gender": str(row.get("gender", "")).strip(),
        "master_category": str(row.get("master_category", "")).strip(),
        "sub_category": str(row.get("sub_category", "")).strip(),
        "article_type": str(row.get("article_type", "")).strip(),
        "base_color": str(row.get("base_color", "")).strip(),
        "season": str(row.get("season", "")).strip(),
        "year": int(float(row.get("year", 0) or 0)),
        "usage": str(row.get("usage", "")).strip(),
        "image_url": str(row.get("image_url", "")).strip(),
        "price": round(float(row.get("price", 0.0) or 0.0), 2),
        "stock": int(float(row.get("stock", 0) or 0)),
        "rating": round(float(row.get("rating", 0.0) or 0.0), 2),
        "review_count": int(float(row.get("review_count", 0) or 0))
    }

def load_data(filepath: str, limit: int = None) -> list:
    """Loads records using Python standard library."""
    if not os.path.exists(filepath):
        # Fallback to check in data/ directory
        alt_path = os.path.join(os.path.dirname(__file__), "..", "data", os.path.basename(filepath))
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Catalog file not found: {filepath}")

    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [sanitize(r) for r in (data[:limit] if limit else data)]
    elif filepath.endswith(".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = []
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                records.append(sanitize(row))
            return records
    raise ValueError("Supported formats: .json, .csv")

try:
    import certifi
    CA_FILE = certifi.where()
except Exception:
    CA_FILE = None

def ingest(filepath: str = "data/products_catalog.json", batch_size: int = 64, limit: int = None):
    print(f"Connecting to MongoDB Atlas...")
    client_kwargs = {}
    if CA_FILE:
        client_kwargs["tlsCAFile"] = CA_FILE
    client = MongoClient(config.MONGODB_URI, **client_kwargs)
    collection = client[config.DB_NAME][config.COLLECTION_NAME]

    setup_indexes(collection)

    records = load_data(filepath, limit=limit)
    print(f"Loaded {len(records):,} catalog records from file.")

    # Resumable Ingestion: Identify products already in Atlas
    existing_ids = set(collection.distinct("product_id"))
    print(f"Already ingested in Atlas: {len(existing_ids):,} products.")
    records_to_process = [r for r in records if r["product_id"] not in existing_ids]
    print(f"Remaining products to embed and ingest: {len(records_to_process):,} products...")

    if not records_to_process:
        print(f"All {len(records):,} products are already ingested in MongoDB Atlas!")
        return

    engine = get_embedding_engine()
    total_to_do = len(records_to_process)

    for i in range(0, total_to_do, batch_size):
        batch = records_to_process[i : i + batch_size]
        texts = [engine.build_search_text(doc) for doc in batch]
        embeddings = engine.generate_embeddings_batch(texts, batch_size=len(batch))

        ops = []
        for doc, st, emb in zip(batch, texts, embeddings):
            doc["search_text"] = st
            doc["embedding"] = emb
            ops.append(UpdateOne({"product_id": doc["product_id"]}, {"$set": doc}, upsert=True))

        if ops:
            collection.bulk_write(ops, ordered=False)

        current_done = min(i + len(batch), total_to_do)
        pct = (current_done / total_to_do) * 100
        print(f"[{current_done:,}/{total_to_do:,}] ({pct:.1f}%) Ingested batch of {len(batch)} items. Total in DB: {collection.count_documents({}):,}", flush=True)

    print(f"Done! Final Atlas Collection total: {collection.count_documents({}):,} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest product catalog to MongoDB Atlas")
    parser.add_argument("--file", default="data/products_catalog.json", help="Path to JSON or CSV file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default 64)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    args = parser.parse_args()

    ingest(filepath=args.file, batch_size=args.batch_size, limit=args.limit)
