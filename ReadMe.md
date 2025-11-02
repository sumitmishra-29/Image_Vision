# Image Vision 🔍

A semantic image search system that allows you to find images using natural language descriptions. Instead of searching by filename or tags, simply describe what you're looking for (e.g., "red flowers on a wooden table") and the system will find matching images.

## Overview

Image Vision uses AI-powered image analysis and vector embeddings to enable intelligent semantic search across your image collection. The system analyzes each image, generates detailed descriptions, converts them into mathematical vectors, and allows you to search using natural language queries.

## How It Works

### 1. Image Preprocessing
Each image in your collection is assigned a unique UUID identifier and saved in a dedicated folder. This ensures consistent identification and prevents filename conflicts.

**Input:** Original images  
**Output:** UUID-named images in `./uuid_images/`

Example:
```
vacation_photo.jpg  →  022bb205-cad4-4cf3-ac39-55e1f89b4310.png
```

### 2. AI-Powered Image Description
Using **Llama 4 Scout Vision Model** via Groq API, the system analyzes each image and generates comprehensive descriptions including:
- Objects present in the image
- Colors and textures
- Spatial relationships
- Scene context
- Visual details

The descriptions are saved in JSON format with the image UUID as the key.

**Input:** UUID images  
**Output:** `image_discriiption/image_descriptions.json`

Example JSON structure:
```json
{
  "022bb205-cad4-4cf3-ac39-55e1f89b4310.png": {
    "scene_summary": "Red roses in a ceramic vase on a wooden table",
    "objects": [
      {
        "label": "roses",
        "color": "red (high confidence)",
        "position": "center, foreground"
      },
      {
        "label": "vase",
        "color": "white ceramic",
        "position": "center"
      }
    ]
  }
}
```

### 3. Semantic Embedding Generation
The text descriptions are converted into high-dimensional vector embeddings using **Sentence-Transformers** (all-MiniLM-L6-v2 model). These 384-dimensional vectors capture the semantic meaning of the descriptions in a mathematical form that enables similarity comparison.

**Input:** Description JSON  
**Output:** 
- `embeddings/image_embeddings.npz` (vector data)
- `embeddings/filename_mapping.json` (metadata)

### 4. Semantic Search
When you search with a text query:
1. Your query is converted into the same 384-dimensional vector space
2. Cosine similarity is calculated between your query vector and all image description vectors
3. Results are ranked by similarity score (0-100%)
4. Top matching images are returned

**Example Search Flow:**
```
Query: "red flowers on table"
  ↓
Query Embedding: [0.23, -0.45, 0.67, ...]
  ↓
Similarity Calculation with all images
  ↓
Top Results:
  1. roses_on_table.png (95.3% match)
  2. tulips_vase.png (87.2% match)
  3. garden_flowers.png (76.8% match)
```

## Project Structure

```
image_vision/
├── scripts/
│   ├── 1_rename_images.py          # Assigns UUIDs to images
│   ├── 2_describe_images.py        # Generates AI descriptions
│   ├── 3_generate_embeddings.py    # Creates vector embeddings
│   └── 4_search_images.py          # CLI search utility
│
├── uuid_images/                # UUID-renamed images
├── descriptions/               # JSON descriptions
│   └── image_descriptions.json
└── embeddings/                 # Vector embeddings
    ├── image_embeddings.npz
    └── filename_mapping.json
|── flask_iv_run.py                      # Flask API server
```

## Installation

### Prerequisites
- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com))

### Setup

1. **Clone or download the project**
```bash
cd image_vision
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
```
groq
aiofiles
sentence-transformers
numpy
flask
flask-cors
```

3. **Set up API key**
```bash
# Windows
set GROQ_API_KEY=your_api_key_here

# Linux/Mac
export GROQ_API_KEY=your_api_key_here
```

## Usage

### Step 1: Prepare Your Images

Place your original images in a source folder, then run:

```bash
python scripts/1_rename_images.py
```

**Configuration:**
```python
INPUT_FOLDER = "path/to/your/images"
OUTPUT_FOLDER = "uuid_images"
```

### Step 2: Generate Descriptions

Analyze images using Llama 4 Scout vision model:

```bash
python scripts/2_describe_images.py
```

**Configuration:**
```python
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
INPUT_FOLDER = "data/uuid_images"
OUTPUT_FILE = "data/descriptions/image_descriptions.json"
CONCURRENT = 2  # Parallel processing limit
```

**Note:** This step requires a Groq API key and internet connection. Processing time depends on the number of images.

### Step 3: Create Embeddings

Convert descriptions into searchable vectors:

```bash
python scripts/3_generate_embeddings.py
```

**Configuration:**
```python
INPUT_JSON = "data/descriptions/image_descriptions.json"
OUTPUT_EMBEDDINGS = "data/embeddings/image_embeddings.npz"
USE_LOCAL_MODEL = True  # Uses local sentence-transformers
```

**Model Used:** `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Fast inference
- Good quality for semantic search

### Step 4: Search Images

#### Option A: Command Line Search
```bash
python scripts/4_search_images.py --interactive
```

Example session:
```
🔍 Search: red flowers on wooden table
1. 022bb205-cad4-4cf3-ac39-55e1f89b4310.png (0.9532)
2. 1ce294e9-03d3-4a18-88b2-14a5ae8e3f99.png (0.8721)
3. 20d545ad-789c-4f09-a0db-09519453bf01.png (0.7688)
```

#### Option B: Flask API Server
```bash
python backend/app.py
```

**API Endpoints:**

**Search Images**
```bash
POST http://localhost:5000/api/search
Content-Type: application/json

{
  "query": "person wearing glasses",
  "top_k": 10
}
```

**Get Image**
```bash
GET http://localhost:5000/api/image/{filename}
```

**Find Similar Images**
```bash
GET http://localhost:5000/api/similar/{filename}?top_k=5
```

**System Stats**
```bash
GET http://localhost:5000/api/stats
```

## Technical Details

### Image Description Model
- **Model:** Llama 4 Scout (17B parameters)
- **Provider:** Groq API
- **Speed:** ~1-2 seconds per image
- **Context:** Supports vision + text multimodal input

### Embedding Model
- **Model:** all-MiniLM-L6-v2
- **Provider:** Sentence-Transformers (local)
- **Dimension:** 384
- **Normalization:** L2 normalized for cosine similarity

### Similarity Calculation
Uses **cosine similarity** between vectors:

```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- A = query embedding vector
- B = image description embedding vector
- Result ranges from 0 (no similarity) to 1 (identical)

### Performance
- **Embedding generation:** ~100 images/second
- **Search speed:** <50ms for 10,000 images
- **Storage:** ~1.5KB per image embedding

## Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key
```

### Path Configuration
Edit the following in each script:

**1_rename_images.py:**
```python
INPUT_FOLDER = Path("path/to/original/images")
OUTPUT_FOLDER = Path("/uuid_images")
```

**2_describe_images.py:**
```python
INPUT_FOLDER = Path("/uuid_images")
OUTPUT_FILE = Path("/descriptions/image_descriptions.json")
CONCURRENT = 2  # Adjust based on rate limits
```

**3_generate_embeddings.py:**
```python
INPUT_JSON = Path("/descriptions/image_descriptions.json")
OUTPUT_EMBEDDINGS = Path("/embeddings/image_embeddings.npz")
```

**backend/app.py:**
```python
EMBEDDINGS_FILE = Path("/embeddings/image_embeddings.npz")
DESCRIPTIONS_FILE = Path("/descriptions/image_descriptions.json")
IMAGES_FOLDER = Path("/uuid_images")
```

## Troubleshooting

### Rate Limit Errors (Groq API)
**Error:** `rate_limit_exceeded`

**Solution:** Reduce `CONCURRENT` in `2_describe_images.py`:
```python
CONCURRENT = 1  # Process one image at a time
```

### Out of Memory
**Error:** Memory error during embedding generation

**Solution:** Process in batches or use a smaller model:
```python
# In 3_generate_embeddings.py
batch_size = 16  # Reduce from 32
```

### Images Not Found
**Error:** `Image not found` when searching

**Solution:** Verify paths are correct and images exist:
```bash
dir data\uuid_images
```

### No Results Found
**Issue:** Search returns no results

**Solutions:**
1. Check if embeddings were generated correctly
2. Try broader search terms
3. Verify descriptions contain relevant content

## Performance Optimization

### For Large Collections (>10,000 images)

1. **Use batch processing:**
```python
# Process descriptions in batches
BATCH_SIZE = 100
```

2. **Enable GPU acceleration (if available):**
```python
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

3. **Use approximate nearest neighbor search:**
```bash
pip install faiss-cpu
```

## Example Searches

| Query | Best Results |
|-------|-------------|
| "red flowers in vase" | Floral arrangements, bouquets |
| "person with glasses working on laptop" | Office/study scenes |
| "wooden furniture" | Tables, chairs, cabinets |
| "blue wall" | Interior scenes with blue backgrounds |
| "sunset over ocean" | Beach/ocean sunset photos |
| "green plants in pots" | Indoor/outdoor plant photos |

## Limitations

- **Image quality:** Blurry or low-quality images may get poor descriptions
- **Abstract art:** Model optimized for realistic images
- **Text in images:** OCR not included, text content not searchable
- **Rate limits:** Groq free tier has request limits
- **Language:** Descriptions and search are English-only

## Future Improvements

- [ ] Multi-language support
- [ ] OCR integration for text in images
- [ ] Real-time indexing for new images
- [ ] Advanced filters (date, size, color)
- [ ] Duplicate image detection
- [ ] Batch upload via web interface
- [ ] Export search results

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions welcome! Areas for improvement:
- Better error handling
- Additional vision models
- Performance optimizations
- Documentation improvements

## Credits

- **Llama 4 Scout:** Meta AI
- **Groq:** LLM inference platform
- **Sentence-Transformers:** Hugging Face
- **Flask:** Pallets Projects

---

**Built with ❤️ for semantic image search**