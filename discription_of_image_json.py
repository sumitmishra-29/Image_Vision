import os
import asyncio
import base64
import json
from pathlib import Path
import aiofiles
from groq import AsyncGroq

# CONFIG
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Vision model that supports images
INPUT_FOLDER = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\uuid_images")
OUTPUT_FILE = Path("C:\\Users\\Sumit\\OneDrive\\Desktop\\image_vision\\image_discriiption\\image_descriptions.json")
CONCURRENT = 2  # Reduced for rate limiting
RETRY_DELAY = 60  # Seconds to wait on rate limit

def build_prompt() -> str:
    """Return detailed prompt for exhaustive object-level description."""
    return (
        "Produce a detailed, exhaustive, human-readable and structured description "
        "of the image. For every object in the frame (people, furniture, mugs, flowers, "
        "walls, fabrics, text, bags, electronics, etc.) list:\n"
        "  1) object label (single-word and short phrase)\n"
        "  2) color(s) with confidence (e.g., 'light blue (high confidence)')\n"
        "  3) texture/material (e.g., 'matte plaster wall', 'ceramic mug')\n"
        "  4) relative position (e.g., 'bottom-left, foreground, 30% from left')\n"
        "  5) approximate size relative to frame (e.g., 'occupies ~12% of image height')\n"
        "  6) counts if multiple (e.g., '3 red flowers in cluster')\n"
        "  7) small distinguishing details (patterns, damage, reflections, shadows)\n"
        "  8) relationships to other objects (e.g., 'mug on wooden table, shadow cast to the right')\n\n"
        "Also include an overall scene summary (1-2 sentences) and a short list of "
        "uncertainties or assumptions. Return JSON only with keys: image, objects (list), "
        "scene_summary, uncertainties.\n"
        "Be precise, avoid filler, and prefer conservative guesses for ambiguous attributes."
    )

async def read_image_b64(path: Path) -> str:
    """Asynchronously read image and return base64 string."""
    async with aiofiles.open(path, "rb") as f:
        data = await f.read()
    return base64.b64encode(data).decode("utf-8")

async def describe_image(client: AsyncGroq, path: Path, sem: asyncio.Semaphore, max_retries: int = 3) -> tuple[str, dict]:
    """Describe one image and return (filename, description_dict)."""
    async with sem:
        for attempt in range(max_retries):
            try:
                b64 = await read_image_b64(path)
                
                # Proper vision API format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": build_prompt()},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                }
                            }
                        ]
                    }
                ]
                
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1600
                )
                
                text = resp.choices[0].message.content
                try:
                    data = json.loads(text)
                    data["image"] = path.name
                except Exception:
                    data = {"image": path.name, "raw": text}
                
                print(f"✓ Processed: {path.name}")
                return path.name, data
                
            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() and attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"⚠ Rate limit hit for {path.name}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif attempt == max_retries - 1:
                    print(f"✗ Failed after {max_retries} attempts: {path.name} - {error_msg}")
                    return path.name, {"error": error_msg}
                else:
                    print(f"✗ Error processing {path.name}: {error_msg}")
                    return path.name, {"error": error_msg}

async def process_folder(input_folder: Path, output_file: Path, concurrent: int = 2):
    """Process all images with concurrency limit; produce dict {filename: description}."""
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    sem = asyncio.Semaphore(concurrent)

    images = sorted([
        p for p in input_folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    ])
    
    print(f"Found {len(images)} images to process")

    # Load existing results if file exists
    results: dict[str, dict] = {}
    if output_file.exists():
        async with aiofiles.open(output_file, "r", encoding="utf-8") as f:
            content = await f.read()
            try:
                results = json.loads(content)
                print(f"Loaded {len(results)} existing results")
            except:
                pass

    # Filter out already processed images
    images_to_process = [p for p in images if p.name not in results or "error" in results.get(p.name, {})]
    print(f"Processing {len(images_to_process)} new/failed images")

    tasks = [asyncio.create_task(describe_image(client, p, sem)) for p in images_to_process]

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        filename, desc = await coro
        results[filename] = desc

        # Incremental write to file (progress saved continuously)
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, ensure_ascii=False, indent=2))
        
        print(f"Progress: {i}/{len(tasks)} images saved")

    await client.close()
    print(f"\n✓ Complete! Results saved to {output_file}")
    return results

if __name__ == "__main__":
    asyncio.run(process_folder(INPUT_FOLDER, OUTPUT_FILE, CONCURRENT))