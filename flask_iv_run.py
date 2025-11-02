from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# CONFIG
EMBEDDINGS_FILE = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\embeddings\\image_embeddings.npz")
DESCRIPTIONS_FILE = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\image_discriiption\\image_descriptions.json")
IMAGES_FOLDER = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\uuid_images")

# Global variables for embeddings
embeddings_data = None
model = None
descriptions = {}

def load_embeddings():
    """Load embeddings and model on startup."""
    global embeddings_data, model, descriptions
    
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_FILE)
    embeddings_data = {
        'embeddings': data['embeddings'],
        'filenames': list(data['filenames'])
    }
    
    print("Loading descriptions...")
    with open(DESCRIPTIONS_FILE, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"✓ Loaded {len(embeddings_data['filenames'])} images")

@app.route('/api/search', methods=['POST'])
def search():
    """Search for images based on text query."""
    try:
        data = request.json
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Generate query embedding
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings_data['embeddings'], query_embedding)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            filename = embeddings_data['filenames'][idx]
            score = float(similarities[idx])
            desc_data = descriptions.get(filename, {})
            
            # Extract scene summary or create from raw
            description = desc_data.get('scene_summary', '')
            if not description and 'raw' in desc_data:
                description = desc_data['raw'][:200] + '...'
            
            results.append({
                'filename': filename,
                'score': score,
                'description': description,
                'full_data': desc_data
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar/<filename>', methods=['GET'])
def find_similar(filename):
    """Find images similar to a specific image."""
    try:
        if filename not in embeddings_data['filenames']:
            return jsonify({'error': 'Image not found'}), 404
        
        top_k = request.args.get('top_k', 10, type=int)
        
        # Get image index
        idx = embeddings_data['filenames'].index(filename)
        query_embedding = embeddings_data['embeddings'][idx]
        
        # Calculate similarities
        similarities = np.dot(embeddings_data['embeddings'], query_embedding)
        
        # Get top-k (excluding query image)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for idx in top_indices:
            fn = embeddings_data['filenames'][idx]
            score = float(similarities[idx])
            desc_data = descriptions.get(fn, {})
            
            results.append({
                'filename': fn,
                'score': score,
                'description': desc_data.get('scene_summary', '')
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'query_image': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<filename>', methods=['GET'])
def get_image(filename):
    """Serve image file."""
    try:
        return send_from_directory(IMAGES_FOLDER, filename)
    except Exception as e:
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/image/<filename>/thumbnail', methods=['GET'])
def get_thumbnail(filename):
    """Get base64 encoded thumbnail."""
    try:
        image_path = IMAGES_FOLDER / filename
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        b64 = base64.b64encode(image_data).decode('utf-8')
        ext = filename.split('.')[-1].lower()
        mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
        
        return jsonify({
            'success': True,
            'data': f"data:{mime_type};base64,{b64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    return jsonify({
        'success': True,
        'total_images': len(embeddings_data['filenames']),
        'embedding_dim': embeddings_data['embeddings'].shape[1],
        'model': 'all-MiniLM-L6-v2'
    })

@app.route('/api/description/<filename>', methods=['GET'])
def get_description(filename):
    """Get full description for an image."""
    try:
        if filename not in descriptions:
            return jsonify({'error': 'Description not found'}), 404
        
        return jsonify({
            'success': True,
            'filename': filename,
            'data': descriptions[filename]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'embeddings_loaded': embeddings_data is not None,
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load embeddings on startup
    load_embeddings()
    
    # Run server
    print("\n🚀 Starting Image Vision API Server...")
    print("=" * 50)
    print(f"API running at: http://localhost:5000")
    print(f"Search endpoint: POST http://localhost:5000/api/search")
    print(f"Image endpoint: GET http://localhost:5000/api/image/<filename>")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)