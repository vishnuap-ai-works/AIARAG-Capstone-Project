"""
Utility script for manually cleaning up Vector Store databases.
Can be used to delete specific files to prevent duplication.
"""

import json
import os
import chromadb
from qdrant_client import QdrantClient
from qdrant_client import models
from config.settings import settings

def delete_from_chroma(filename: str):
    """Deletes all chunks associated with a specific filename in ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        col = client.get_or_create_collection(name=settings.CHROMA_COLLECTION)
        col.delete(where={"file_name": filename})
        print(f"✅ Successfully deleted {filename} from ChromaDB.")
    except Exception as e:
        print(f"❌ Failed to delete {filename} from ChromaDB: {e}")

def delete_from_qdrant(filename: str):
    """Deletes all chunks associated with a specific filename in Qdrant."""
    try:
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_name",
                        match=models.MatchValue(value=filename),
                    )
                ]
            )
        )
        print(f"✅ Successfully deleted {filename} from Qdrant.")
    except Exception as e:
        print(f"❌ Failed to delete {filename} from Qdrant: {e}")

def delete_from_json(filename: str):
    """Deletes all chunks associated with a specific filename in JSON store."""
    try:
        filepath = settings.VECTOR_STORE_PATH
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if filename in data:
                del data[filename]
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"✅ Successfully deleted {filename} from JSON Store.")
            else:
                print(f"⚠️ {filename} not found in JSON Store.")
        else:
            print(f"⚠️ JSON Vector Store file not found at {filepath}.")
    except Exception as e:
        print(f"❌ Failed to delete {filename} from JSON Store: {e}")

if __name__ == "__main__":
    # Example Usage:
    test_filename = "test"
    print(f"Attempting to delete '{test_filename}' from all vector stores...")
    
    delete_from_chroma(test_filename)
    delete_from_qdrant(test_filename)
    delete_from_json(test_filename)
