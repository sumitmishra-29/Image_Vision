import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# CONFIG
EMBEDDINGS_FILE = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\embeddings\\image_embeddings.npz")
MAPPING_FILE = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\embeddings\\filename_mapping.json")
DESCRIPTIONS_FILE = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\image_discriiption\\image_descriptions.json")

class ImageSearchEngine:
    """Semantic search engine for images using embeddings."""
    
    def __init__(self, embeddings_path: Path, mapping_path: Path, descriptions_path: Path = None):
        """Initialize search engine with embeddings and mapping."""
        print("Loading embeddings...")
        data = np.load(embeddings_path)
        self.embeddings = data['embeddings']
        self.filenames = list(data['filenames'])
        
        print("Loading metadata...")
        with open(mapping_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load descriptions if available
        self.descriptions = {}
        if descriptions_path and descriptions_path.exists():
            with open(descriptions_path, 'r', encoding='utf-8') as f:
                self.descriptions = json.load(f)
        
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✓ Loaded {len(self.filenames)} images with {self.metadata['embedding_dim']}-dim embeddings\n")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for images similar to the query.
        
        Returns:
            List of (filename, similarity_score, description_data) tuples
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        
        # Calculate cosine similarities (dot product since normalized)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            filename = self.filenames[idx]
            score = float(similarities[idx])
            desc_data = self.descriptions.get(filename, {})
            results.append((filename, score, desc_data))
        
        return results
    
    def search_by_filename(self, filename: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find images similar to a specific image by filename."""
        if filename not in self.filenames:
            raise ValueError(f"Filename {filename} not found in embeddings")
        
        idx = self.filenames.index(filename)
        query_embedding = self.embeddings[idx]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k (excluding the query image itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            results.append((self.filenames[idx], float(similarities[idx])))
        
        return results
    
    def get_description(self, filename: str) -> dict:
        """Get description data for a specific image."""
        return self.descriptions.get(filename, {})
    
    def print_results(self, results: List[Tuple[str, float, dict]], show_descriptions: bool = True):
        """Pretty print search results."""
        for i, (filename, score, desc_data) in enumerate(results, 1):
            print(f"\n{i}. {filename} (similarity: {score:.4f})")
            
            if show_descriptions and desc_data:
                if "scene_summary" in desc_data:
                    print(f"   📝 {desc_data['scene_summary']}")
                
                if "objects" in desc_data and isinstance(desc_data["objects"], list):
                    print(f"   🏷️  Objects: {len(desc_data['objects'])} detected")
                    # Show first few objects
                    for obj in desc_data["objects"][:3]:
                        if isinstance(obj, dict) and "label" in obj:
                            print(f"      - {obj.get('label', 'unknown')}: {obj.get('color', '')}")

def main():
    """Example usage of the search engine."""
    # Initialize search engine
    engine = ImageSearchEngine(EMBEDDINGS_FILE, MAPPING_FILE, DESCRIPTIONS_FILE)
    
    # Example searches
    queries = [
        "red flowers in a vase",
        "person wearing glasses",
        "wooden furniture",
        "blue wall",
        "coffee mug on desk"
    ]
    
    print("=" * 80)
    print("SEMANTIC IMAGE SEARCH - Example Queries")
    print("=" * 80)
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 80)
        
        results = engine.search(query, top_k=3)
        engine.print_results(results, show_descriptions=True)
    
    # Example: Find similar images to a specific image
    print("\n\n" + "=" * 80)
    print("FIND SIMILAR IMAGES")
    print("=" * 80)
    
    if engine.filenames:
        example_filename = engine.filenames[0]
        print(f"\n🔍 Finding images similar to: {example_filename}")
        print("-" * 80)
        
        similar = engine.search_by_filename(example_filename, top_k=5)
        for i, (filename, score) in enumerate(similar, 1):
            print(f"{i}. {filename} (similarity: {score:.4f})")

def interactive_search():
    """Interactive search mode."""
    engine = ImageSearchEngine(EMBEDDINGS_FILE, MAPPING_FILE, DESCRIPTIONS_FILE)
    
    print("\n" + "=" * 80)
    print("INTERACTIVE IMAGE SEARCH")
    print("=" * 80)
    print("Enter your search queries (or 'quit' to exit)\n")
    
    while True:
        query = input("🔍 Search: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print()
        results = engine.search(query, top_k=5)
        engine.print_results(results, show_descriptions=True)
        print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    import sys
    
    # Check if embeddings exist
    if not EMBEDDINGS_FILE.exists():
        print("❌ Embeddings file not found!")
        print(f"Please run the embedding generation script first.")
        print(f"Looking for: {EMBEDDINGS_FILE}")
        sys.exit(1)
    
    # Run interactive mode if --interactive flag is passed
    if len(sys.argv) > 1 and sys.argv[1] in ['-i', '--interactive']:
        interactive_search()
    else:
        main()