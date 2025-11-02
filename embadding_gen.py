import os
import json
import asyncio
from pathlib import Path
import aiofiles
import numpy as np
from groq import AsyncGroq
from typing import List, Dict

# CONFIG
EMBEDDING_MODEL = "text-embedding-3-small"  # For OpenAI, or use sentence-transformers locally
INPUT_JSON = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\image_discriiption\\image_descriptions.json")
OUTPUT_EMBEDDINGS = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\embeddings\\image_embeddings.npz")
OUTPUT_MAPPING = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\embeddings\\filename_mapping.json")
BATCH_SIZE = 10
USE_LOCAL_MODEL = True  # Set to False if using OpenAI API

async def load_descriptions(json_path: Path) -> Dict[str, str]:
    """Load image descriptions and create text representations."""
    async with aiofiles.open(json_path, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)
    
    descriptions = {}
    for filename, desc_data in data.items():
        if "error" in desc_data:
            print(f"⚠ Skipping {filename} (has error)")
            continue
        
        # Create comprehensive text representation
        text_parts = []
        
        # Add scene summary
        if "scene_summary" in desc_data:
            text_parts.append(f"Scene: {desc_data['scene_summary']}")
        
        # Add objects with details
        if "objects" in desc_data and isinstance(desc_data["objects"], list):
            for obj in desc_data["objects"]:
                if isinstance(obj, dict):
                    obj_text = " ".join([
                        str(v) for v in obj.values() if v
                    ])
                    text_parts.append(obj_text)
        
        # Fallback to raw if structured data not available
        if not text_parts and "raw" in desc_data:
            text_parts.append(desc_data["raw"])
        
        descriptions[filename] = " ".join(text_parts)
    
    return descriptions

async def generate_embeddings_local(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings using local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    
    print("Loading local embedding model (sentence-transformers)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
    # Alternative models:
    # - 'all-mpnet-base-v2' (better quality, slower)
    # - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
    
    print(f"Generating embeddings for {len(texts)} descriptions...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    
    return embeddings

async def generate_embeddings_openai(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Generate embeddings using OpenAI API."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = []
    
    print(f"Generating embeddings for {len(texts)} descriptions using OpenAI...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    await client.close()
    return np.array(embeddings)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

async def search_similar_images(
    query: str,
    embeddings: np.ndarray,
    filename_list: List[str],
    top_k: int = 5,
    use_local: bool = True
) -> List[tuple]:
    """Search for similar images based on text query."""
    if use_local:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)
        await client.close()
    
    # Calculate similarities
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((filename_list[i], sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

async def main():
    print("🚀 Starting embedding generation process...\n")
    
    # Create output directories
    OUTPUT_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    
    # Load descriptions
    print("📖 Loading image descriptions...")
    descriptions = await load_descriptions(INPUT_JSON)
    print(f"✓ Loaded {len(descriptions)} valid descriptions\n")
    
    # Prepare data
    filenames = list(descriptions.keys())
    texts = list(descriptions.values())
    
    # Generate embeddings
    if USE_LOCAL_MODEL:
        embeddings = await generate_embeddings_local(texts, batch_size=BATCH_SIZE)
    else:
        embeddings = await generate_embeddings_openai(texts, batch_size=BATCH_SIZE)
    
    print(f"✓ Generated embeddings with shape: {embeddings.shape}\n")
    
    # Save embeddings (compressed numpy format)
    print("💾 Saving embeddings...")
    np.savez_compressed(
        OUTPUT_EMBEDDINGS,
        embeddings=embeddings,
        filenames=filenames
    )
    
    # Save mapping (for easy lookup)
    mapping_data = {
        "filenames": filenames,
        "embedding_dim": int(embeddings.shape[1]),
        "num_images": len(filenames),
        "model_used": "all-MiniLM-L6-v2" if USE_LOCAL_MODEL else "text-embedding-3-small"
    }
    
    async with aiofiles.open(OUTPUT_MAPPING, "w", encoding="utf-8") as f:
        await f.write(json.dumps(mapping_data, indent=2))
    
    print(f"✓ Saved embeddings to: {OUTPUT_EMBEDDINGS}")
    print(f"✓ Saved mapping to: {OUTPUT_MAPPING}\n")
    
    # Example search
    print("🔍 Example search: 'red flowers on table'")
    results = await search_similar_images(
        "red flowers on table",
        embeddings,
        filenames,
        top_k=5,
        use_local=USE_LOCAL_MODEL
    )
    
    print("\nTop 5 similar images:")
    for filename, score in results:
        print(f"  {filename}: {score:.4f}")
    
    print("\n✅ Complete!")

# Utility functions for loading and searching later
def load_embeddings(embeddings_path: Path):
    """Load saved embeddings."""
    data = np.load(embeddings_path)
    return data['embeddings'], data['filenames']

async def search_images_by_text(
    query: str,
    embeddings_path: Path,
    top_k: int = 5,
    use_local: bool = True
):
    """Standalone function to search images by text query."""
    embeddings, filenames = load_embeddings(embeddings_path)
    results = await search_similar_images(
        query,
        embeddings,
        list(filenames),
        top_k=top_k,
        use_local=use_local
    )
    return results

if __name__ == "__main__":
    # Check dependencies
    try:
        if USE_LOCAL_MODEL:
            import sentence_transformers
    except ImportError:
        print("❌ sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        exit(1)
    
    asyncio.run(main())