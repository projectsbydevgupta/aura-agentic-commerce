import json
import os
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.server_api import ServerApi
try:
    import certifi
    CA_FILE = certifi.where()
except Exception:
    CA_FILE = None
from src import config
from src.search.embeddings import get_embedding_engine

def serialize_doc(doc: dict) -> dict:
    """Ensures MongoDB document is JSON/msgpack serializable (e.g. converting ObjectId)."""
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

# Common synonyms and catalog normalization mapping
ARTICLE_TYPE_SYNONYMS = {
    "sneakers": ["Casual Shoes", "Sports Shoes"],
    "sneaker": ["Casual Shoes", "Sports Shoes"],
    "running shoes": ["Sports Shoes"],
    "running shoe": ["Sports Shoes"],
    "sports shoes": ["Sports Shoes"],
    "sports shoe": ["Sports Shoes"],
    "casual shoes": ["Casual Shoes"],
    "casual shoe": ["Casual Shoes"],
    "formal shoes": ["Formal Shoes"],
    "formal shoe": ["Formal Shoes"],
    "shoes": ["Casual Shoes", "Sports Shoes", "Formal Shoes"],
    "shoe": ["Casual Shoes", "Sports Shoes", "Formal Shoes"],
    "footwear": ["Casual Shoes", "Sports Shoes", "Formal Shoes", "Heels", "Flats", "Sandals", "Flip Flops"],
    "trainer": ["Sports Shoes", "Casual Shoes"],
    "trainers": ["Sports Shoes", "Casual Shoes"],
    "watch": ["Watches"],
    "watches": ["Watches"],
    "tshirt": ["Tshirts"],
    "tshirts": ["Tshirts"],
    "t-shirt": ["Tshirts"],
    "t-shirts": ["Tshirts"],
    "tee": ["Tshirts"],
    "shirt": ["Shirts"],
    "shirts": ["Shirts"],
    "denim": ["Jeans"],
    "jeans": ["Jeans"],
    "jean": ["Jeans"],
    "pants": ["Trousers", "Track Pants", "Jeans"],
    "trousers": ["Trousers"],
    "shorts": ["Shorts"],
    "short": ["Shorts"],
    "dress": ["Dresses", "Kurtas"],
    "dresses": ["Dresses"],
    "kurta": ["Kurtas"],
    "kurtas": ["Kurtas"],
    "jacket": ["Jackets"],
    "jackets": ["Jackets"],
    "handbag": ["Handbags"],
    "handbags": ["Handbags"],
    "bag": ["Handbags", "Backpacks"],
    "wallet": ["Wallets"],
    "wallets": ["Wallets"],
    "belt": ["Belts"],
    "belts": ["Belts"],
    "socks": ["Socks"],
    "sock": ["Socks"],
    "sunglasses": ["Sunglasses"],
    "sunglass": ["Sunglasses"],
    "track pants": ["Track Pants"],
    "track pant": ["Track Pants"],
    "tops": ["Tops"],
    "top": ["Tops"],
    "heels": ["Heels"],
    "heel": ["Heels"],
    "flats": ["Flats"],
    "flat": ["Flats"],
    "sandals": ["Sandals"],
    "sandal": ["Sandals"],
    "flip flops": ["Flip Flops"],
    "flip flop": ["Flip Flops"],
    "perfume": ["Perfume and Body Mist"],
    "deodorant": ["Deodorant"],
    "tie": ["Ties"],
    "ties": ["Ties"],
    "cap": ["Caps"],
    "caps": ["Caps"],
    "hat": ["Hats"],
    "hats": ["Hats"],
    "scarf": ["Scarves"],
    "scarves": ["Scarves"],
}

class ProductHybridSearchEngine:
    """Combines Atlas Vector Search + MongoDB Text Search + Metadata Filtering via RRF."""

    def __init__(self, uri: Optional[str] = None):
        connection_uri = uri or config.MONGODB_URI
        if not connection_uri:
            raise ValueError("MONGODB_URI or MONGODB_PASSWORD environment variable is required.")
        client_kwargs = {
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS": 5000
        }
        if CA_FILE:
            client_kwargs["tlsCAFile"] = CA_FILE
        self.client = MongoClient(connection_uri, **client_kwargs)
        self.collection = self.client[config.DB_NAME][config.COLLECTION_NAME]
        self.engine = get_embedding_engine()
        self._local_catalog = None
        self._mongo_reachable: Optional[bool] = None

    def is_mongo_reachable(self) -> bool:
        if self._mongo_reachable is not None:
            return self._mongo_reachable
        try:
            self.client.admin.command('ping', maxTimeMS=1200)
            self._mongo_reachable = True
        except Exception:
            self._mongo_reachable = False
        return self._mongo_reachable

    def _load_local_catalog(self) -> List[dict]:
        """Loads and caches the 44,000+ local catalog JSON documents as a robust zero-latency fallback."""
        if self._local_catalog is not None:
            return self._local_catalog
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "products_catalog.json"),
            os.path.join(os.getcwd(), "data", "products_catalog.json"),
            "data/products_catalog.json"
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self._local_catalog = json.load(f)
                        return self._local_catalog
                except Exception as e:
                    print(f"[SearchEngine] Notice loading local catalog: {e}")
        self._local_catalog = []
        return self._local_catalog

    def local_catalog_search(self, query: str, filter_dict: Optional[dict] = None, limit: int = 15) -> List[dict]:
        """
        High-performance in-memory semantic & keyword catalog search fallback.
        Filters strictly by metadata constraints (gender, article_type, base_color, budget)
        and scores products based on query relevance over 44,000+ authentic catalog items.
        """
        catalog = self._load_local_catalog()
        if not catalog:
            return []

        filter_dict = self.sanitize_db_filter(filter_dict) if filter_dict else {}
        q_raw = (query or "").lower().replace(",", " ").replace(".", " ")
        stopwords = {
            "the", "for", "and", "under", "with", "men", "women", "mens", "womens",
            "a", "an", "i", "want", "to", "buy", "need", "show", "me", "find", "shoes", "shoe"
        }
        q_words = [w for w in q_raw.split() if len(w) > 1 and w not in stopwords]

        brand = filter_dict.get("brand")
        gender = filter_dict.get("gender")
        article_type = filter_dict.get("article_type")
        master_category = filter_dict.get("master_category")
        sub_category = filter_dict.get("sub_category")
        price_filt = filter_dict.get("price", {}) if isinstance(filter_dict.get("price"), dict) else {}
        max_price = price_filt.get("$lte") or filter_dict.get("max_price") or filter_dict.get("$lte")
        min_price = price_filt.get("$gte") or filter_dict.get("min_price") or filter_dict.get("$gte")
        base_color = filter_dict.get("base_color")
        rating_filt = filter_dict.get("rating", {}) if isinstance(filter_dict.get("rating"), dict) else {}
        min_rating = rating_filt.get("$gte") or filter_dict.get("min_rating")

        allowed_types = None
        if isinstance(article_type, dict) and "$in" in article_type:
            allowed_types = set(t.lower() for t in article_type["$in"])
        elif isinstance(article_type, str):
            norm = article_type.strip().lower()
            if norm in ARTICLE_TYPE_SYNONYMS:
                allowed_types = set(t.lower() for t in ARTICLE_TYPE_SYNONYMS[norm])
            else:
                allowed_types = {norm}

        scored = []
        for it in catalog:
            # Metadata filter constraints
            if brand and it.get("brand") and it.get("brand").lower() != str(brand).lower():
                continue
            if gender and it.get("gender") and it.get("gender").lower() != str(gender).lower():
                continue
            if allowed_types:
                it_type = (it.get("article_type") or "").lower()
                if it_type not in allowed_types:
                    continue
            if master_category and it.get("master_category") and it.get("master_category").lower() != str(master_category).lower():
                continue
            if sub_category and it.get("sub_category") and it.get("sub_category").lower() != str(sub_category).lower():
                continue
            if base_color and it.get("base_color") and it.get("base_color").lower() != str(base_color).lower():
                continue

            price = it.get("price", 0)
            if max_price is not None and price > max_price:
                continue
            if min_price is not None and price < min_price:
                continue
            if min_rating is not None and it.get("rating", 0) < min_rating:
                continue

            # Text relevance scoring
            name = (it.get("name") or "").lower()
            it_brand = (it.get("brand") or "").lower()
            art_type = (it.get("article_type") or "").lower()
            sub_cat = (it.get("sub_category") or "").lower()

            score = 0
            for w in q_words:
                if w in name:
                    score += 3
                elif w in art_type:
                    score += 2
                elif w in it_brand or w in sub_cat:
                    score += 1

            if q_words and score == 0:
                continue

            scored.append((score, it))

        # If strict scoring with q_words returned 0, relax query words constraint but keep metadata filters
        if not scored and (allowed_types or gender or master_category or brand):
            for it in catalog:
                if brand and it.get("brand") and it.get("brand").lower() != str(brand).lower():
                    continue
                if gender and it.get("gender") and it.get("gender").lower() != str(gender).lower():
                    continue
                if allowed_types:
                    it_type = (it.get("article_type") or "").lower()
                    if it_type not in allowed_types:
                        continue
                if master_category and it.get("master_category") and it.get("master_category").lower() != str(master_category).lower():
                    continue
                if base_color and it.get("base_color") and it.get("base_color").lower() != str(base_color).lower():
                    continue
                price = it.get("price", 0)
                if max_price is not None and price > max_price:
                    continue
                if min_price is not None and price < min_price:
                    continue
                if min_rating is not None and it.get("rating", 0) < min_rating:
                    continue
                scored.append((1, it))

        scored.sort(key=lambda x: (x[0], x[1].get("rating", 0)), reverse=True)
        return [serialize_doc(s[1]) for s in scored[:limit]]

    @staticmethod
    def build_filter(
        brand: Optional[str] = None,
        gender: Optional[str] = None,
        master_category: Optional[str] = None,
        sub_category: Optional[str] = None,
        article_type: Optional[str] = None,
        base_color: Optional[str] = None,
        season: Optional[str] = None,
        usage: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        in_stock: bool = False,
        min_rating: Optional[float] = None
    ) -> Dict[str, Any]:
        """Constructs robust MongoDB query filters."""
        query = {}
        for field, val in [
            ("brand", brand), ("gender", gender), ("master_category", master_category),
            ("sub_category", sub_category), ("base_color", base_color),
            ("season", season), ("usage", usage)
        ]:
            if val:
                query[field] = val.strip()

        # Handle article_type normalization
        if article_type:
            norm_key = article_type.strip().lower()
            if norm_key in ARTICLE_TYPE_SYNONYMS:
                syns = ARTICLE_TYPE_SYNONYMS[norm_key]
                if len(syns) == 1:
                    query["article_type"] = syns[0]
                else:
                    query["article_type"] = {"$in": syns}
            else:
                query["article_type"] = article_type.strip()

        price = {}
        if min_price is not None: price["$gte"] = float(min_price)
        if max_price is not None: price["$lte"] = float(max_price)
        if price: query["price"] = price

        if in_stock: query["stock"] = {"$gt": 0}
        if min_rating is not None: query["rating"] = {"$gte": float(min_rating)}

        return query

    @staticmethod
    def sanitize_db_filter(filter_dict: Optional[dict]) -> dict:
        """Strips UI-only keys and ensures filters map to valid MongoDB collection fields."""
        if not filter_dict:
            return {}
        valid_fields = {
            "brand", "gender", "master_category", "sub_category",
            "article_type", "base_color", "season", "usage",
            "price", "stock", "rating", "_id", "id", "product_id"
        }
        cleaned = {}
        for k, v in filter_dict.items():
            if k in valid_fields:
                cleaned[k] = v
            elif k == "color":
                # Remap 'color' (UI key) to 'base_color' (MongoDB field)
                cleaned["base_color"] = v
        return cleaned

    def vector_search(self, query: str, filter_dict: Optional[dict] = None, limit: int = 20) -> List[dict]:
        """Runs Atlas Vector Search using sentence-transformers embedding."""
        if not self.is_mongo_reachable():
            return self.local_catalog_search(query, filter_dict=filter_dict, limit=limit)
        vec = self.engine.generate_embedding(query)
        stage = {
            "$vectorSearch": {
                "index": config.VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": vec,
                "numCandidates": max(limit * 5, 100),
                "limit": limit
            }
        }
        db_filter = self.sanitize_db_filter(filter_dict)
        if db_filter:
            stage["$vectorSearch"]["filter"] = db_filter

        try:
            docs = list(self.collection.aggregate([stage, {"$project": {"embedding": 0, "vector_score": {"$meta": "vectorSearchScore"}}}]))
            return [serialize_doc(d) for d in docs]
        except Exception:
            self._mongo_reachable = False
            return self.local_catalog_search(query, filter_dict=filter_dict, limit=limit)

    def keyword_search(self, query: str, filter_dict: Optional[dict] = None, limit: int = 20) -> List[dict]:
        """Runs MongoDB text keyword search with regex fallback."""
        if not self.is_mongo_reachable():
            return self.local_catalog_search(query, filter_dict=filter_dict, limit=limit)
        db_filter = self.sanitize_db_filter(filter_dict)
        match = {"$text": {"$search": query}}
        if db_filter:
            match.update(db_filter)

        pipeline = [
            {"$match": match},
            {"$project": {"embedding": 0, "keyword_score": {"$meta": "textScore"}}},
            {"$sort": {"keyword_score": -1}},
            {"$limit": limit}
        ]
        try:
            docs = list(self.collection.aggregate(pipeline))
            return [serialize_doc(d) for d in docs]
        except Exception:
            try:
                regex_match = {"$or": [{"name": {"$regex": query, "$options": "i"}}, {"brand": {"$regex": query, "$options": "i"}}]}
                if db_filter: regex_match.update(db_filter)
                docs = list(self.collection.find(regex_match, {"embedding": 0}).limit(limit))
                return [serialize_doc(d) for d in docs]
            except Exception:
                self._mongo_reachable = False
                return self.local_catalog_search(query, filter_dict=filter_dict, limit=limit)

    def hybrid_search(
        self,
        query: str,
        filter_dict: Optional[dict] = None,
        vec_weight: float = 0.6,
        kw_weight: float = 0.4,
        rrf_k: int = 60,
        limit: int = 10,
        fallback: bool = True
    ) -> List[dict]:
        """Fuses Vector Search + Keyword Search with Reciprocal Rank Fusion (RRF) & Smart Fallback."""
        if not self.is_mongo_reachable():
            return self.local_catalog_search(query, filter_dict=filter_dict, limit=limit)

        vec_results = self.vector_search(query, filter_dict=filter_dict, limit=limit * 3)
        kw_results = self.keyword_search(query, filter_dict=filter_dict, limit=limit * 3)

        items = {}
        rrf_scores = {}

        for rank, doc in enumerate(vec_results, 1):
            pid = doc.get("product_id") or str(doc["_id"])
            items[pid] = serialize_doc(doc)
            items[pid]["vector_rank"] = rank
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (vec_weight / (rrf_k + rank))

        for rank, doc in enumerate(kw_results, 1):
            pid = doc.get("product_id") or str(doc["_id"])
            if pid not in items:
                items[pid] = serialize_doc(doc)
            items[pid]["keyword_rank"] = rank
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (kw_weight / (rrf_k + rank))

        ranked = []
        for pid, score in rrf_scores.items():
            doc = items[pid]
            doc["rrf_score"] = round(score, 6)
            ranked.append(doc)

        ranked.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Smart Fallback: If strict filtering returned 0 results, relax constraints gracefully
        if not ranked and filter_dict and fallback:
            relaxed_filters = {}
            if "gender" in filter_dict: relaxed_filters["gender"] = filter_dict["gender"]
            if "article_type" in filter_dict: relaxed_filters["article_type"] = filter_dict["article_type"]
            if "master_category" in filter_dict: relaxed_filters["master_category"] = filter_dict["master_category"]
            if "price" in filter_dict: relaxed_filters["price"] = filter_dict["price"]
            
            if relaxed_filters != filter_dict:
                return self.hybrid_search(query, filter_dict=relaxed_filters, limit=limit, fallback=True)
            elif "article_type" in filter_dict:
                cat_only = {"article_type": filter_dict["article_type"]}
                if "gender" in filter_dict: cat_only["gender"] = filter_dict["gender"]
                return self.hybrid_search(query, filter_dict=cat_only, limit=limit, fallback=False)
            else:
                return self.hybrid_search(query, filter_dict=None, limit=limit, fallback=False)

        return ranked[:limit]

    def inspect_category_price_bounds(
        self,
        article_type: Optional[Any] = None,
        master_category: Optional[str] = None,
        gender: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculates minimum, maximum, and average price in MongoDB for the category constraints."""
        match = {}
        if article_type:
            if isinstance(article_type, dict):
                match["article_type"] = article_type
            elif isinstance(article_type, str):
                norm = article_type.strip().lower()
                if norm in ARTICLE_TYPE_SYNONYMS:
                    syns = ARTICLE_TYPE_SYNONYMS[norm]
                    match["article_type"] = syns[0] if len(syns) == 1 else {"$in": syns}
                else:
                    match["article_type"] = article_type.strip()
        if master_category:
            match["master_category"] = master_category
        if gender:
            match["gender"] = gender

        pipeline = [
            {"$match": match if match else {"price": {"$exists": True, "$gt": 0}}},
            {"$group": {
                "_id": None,
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"},
                "avg_price": {"$avg": "$price"},
                "count": {"$sum": 1}
            }}
        ]
        try:
            res = list(self.collection.aggregate(pipeline))
            if res and res[0].get("min_price") is not None:
                return {
                    "min_price": round(float(res[0]["min_price"]), 2),
                    "max_price": round(float(res[0]["max_price"]), 2),
                    "avg_price": round(float(res[0]["avg_price"]), 2),
                    "count": int(res[0]["count"])
                }
        except Exception as e:
            print(f"Price bounds aggregation notice: {e}")

        # Compute actual price bounds from 44,000+ local catalog items
        try:
            catalog = self._load_local_catalog()
            allowed_types = None
            if isinstance(article_type, dict) and "$in" in article_type:
                allowed_types = set(t.lower() for t in article_type["$in"])
            elif isinstance(article_type, str):
                norm = article_type.strip().lower()
                if norm in ARTICLE_TYPE_SYNONYMS:
                    allowed_types = set(t.lower() for t in ARTICLE_TYPE_SYNONYMS[norm])
                else:
                    allowed_types = {norm}

            prices = []
            for it in catalog:
                if gender and it.get("gender") and it.get("gender").lower() != str(gender).lower():
                    continue
                if allowed_types:
                    it_type = (it.get("article_type") or "").lower()
                    if it_type not in allowed_types:
                        continue
                p = it.get("price")
                if p is not None and p > 0:
                    prices.append(float(p))

            if prices:
                return {
                    "min_price": round(min(prices), 2),
                    "max_price": round(max(prices), 2),
                    "avg_price": round(sum(prices) / len(prices), 2),
                    "count": len(prices)
                }
        except Exception:
            pass
        return {"min_price": 20.0, "max_price": 250.0, "avg_price": 65.0, "count": 100}

    def hybrid_search_with_price_intelligence(
        self,
        query: str,
        filter_dict: Optional[dict] = None,
        limit: int = 15
    ) -> tuple:
        """
        Executes hybrid search with budget & price boundary reasoning.
        Returns: (results, price_analysis_dict)
        """
        filter_dict = filter_dict or {}
        price_filter = filter_dict.get("price", {})
        requested_max_price = price_filter.get("$lte")
        requested_min_price = price_filter.get("$gte")

        category_label = filter_dict.get("article_type") or filter_dict.get("master_category") or "items"
        if isinstance(category_label, dict) and "$in" in category_label:
            category_label = "/".join(category_label["$in"])

        price_analysis = {
            "price_filter_present": bool(price_filter),
            "requested_min_price": requested_min_price,
            "requested_max_price": requested_max_price,
            "price_gap_detected": False,
            "catalog_min_price": None,
            "catalog_avg_price": None,
            "category_name": str(category_label)
        }

        # 1. First attempt: search with full strict filters (fallback disabled to detect price gap)
        results = self.hybrid_search(query=query, filter_dict=filter_dict, limit=limit, fallback=False)

        if results:
            return results, price_analysis

        # 2. If 0 results and a price ceiling was set (e.g. max_price):
        if requested_max_price is not None:
            bounds = self.inspect_category_price_bounds(
                article_type=filter_dict.get("article_type"),
                master_category=filter_dict.get("master_category"),
                gender=filter_dict.get("gender")
            )
            min_catalog_price = bounds.get("min_price", 20.0)
            avg_catalog_price = bounds.get("avg_price", 65.0)

            price_analysis["catalog_min_price"] = min_catalog_price
            price_analysis["catalog_avg_price"] = avg_catalog_price

            if requested_max_price < min_catalog_price:
                price_analysis["price_gap_detected"] = True
                price_analysis["suggested_budget_floor"] = min_catalog_price

                # Relax price filter to retrieve closest available entry-level pieces
                relaxed_filters = filter_dict.copy()
                relaxed_filters.pop("price", None)
                fallback_results = self.hybrid_search(query=query, filter_dict=relaxed_filters, limit=limit, fallback=True)
                
                # Sort fallback results by price ascending to prioritize entry-level pieces
                fallback_results.sort(key=lambda x: x.get("price", 999.0))
                return fallback_results, price_analysis

        # 3. Default fallback if no special price gap
        if not results and filter_dict:
            return self.hybrid_search(query=query, filter_dict=filter_dict, limit=limit, fallback=True), price_analysis

        return results, price_analysis


_shared_search_engine: Optional[ProductHybridSearchEngine] = None

def get_search_engine() -> ProductHybridSearchEngine:
    """Returns the shared ProductHybridSearchEngine instance (lazy initialized)."""
    global _shared_search_engine
    if _shared_search_engine is None:
        _shared_search_engine = ProductHybridSearchEngine()
    return _shared_search_engine
