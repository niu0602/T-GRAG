import os
import sys
import logging
import argparse
from pathlib import Path
from openai import AsyncOpenAI
from time_graphrag import GraphRAG, QueryParam
from time_graphrag.base import BaseKVStorage
from time_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
import numpy as np

# Add parent directory to path for module imports
sys.path.append("..")

def parse_args():
    parser = argparse.ArgumentParser(description="Time-GRAPH batch graph building script")
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API Key"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_API_BASE", ""),
        help="OpenAI API Base URL"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "qwen"),
        help="Name of the chat model"
    )
    parser.add_argument(
        "--embed-dir",
        default=os.getenv("EMBED_MODEL_DIR", "./models/embed"),
        help="Local path to the SentenceTransformer model"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "WARNING"),
        help="Logging level: DEBUG/INFO/WARNING/ERROR"
    )
    return parser.parse_args()

args = parse_args()

# Configure logging
logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.WARNING)
)
logging.getLogger("time_graphrag").setLevel(logging.INFO)

API_KEY = args.api_key
BASE_URL = args.base_url
MODEL   = args.model

async def model_if_cache(
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs
) -> str:
    """
    Send a chat completion request to OpenAI, with optional caching.

    If a BaseKVStorage instance is provided under kwargs['hashing_kv'],
    responses will be cached and retrieved based on a hash of the model
    and messages.
    """
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Extract the cache storage if provided
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Check cache for existing response
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    # Make the API call
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **kwargs
    )

    # Store response in cache if applicable
    if hashing_kv is not None:
        await hashing_kv.upsert({
            args_hash: {
                "return": response.choices[0].message.content,
                "model": MODEL
            }
        })

    return response.choices[0].message.content

# Initialize the embedding model
EMBED_MODEL = SentenceTransformer(
    args.embed_dir,
    device="cuda:0"
)

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    """
    Compute normalized sentence embeddings using a local SentenceTransformer model.
    """
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

def remove_if_exist(filepath):
    """
    Delete the file if it already exists.
    """
    if os.path.exists(filepath):
        os.remove(filepath)

def insert(working_dir, filepath, timestamp):
    """
    Read the content of the given file and insert it into a GraphRAG index.

    :param working_dir: Directory where the index will be stored
    :param filepath: Path to the Markdown file containing the text
    :param timestamp: A string representing the time context (e.g., year)
    """
    from time import time

    # Read the text file (handling BOM if present)
    with open(filepath, encoding="utf-8-sig") as f:
        text = f.read()

    # Initialize GraphRAG with caching and embedding settings
    rag = GraphRAG(
        working_dir=working_dir,
        enable_llm_cache=True,
        best_model_func=model_if_cache,
        cheap_model_func=model_if_cache,
        embedding_func=local_embedding,
        time=timestamp
    )

    start = time()
    rag.insert(text)
    print("Indexing time:", time() - start)

def batch_insert(directory):
    """
    Process all Markdown files in the specified directory, building
    a separate index for each based on its filename (e.g., year.md).
    """
    # List all files in the directory
    files = os.listdir(directory)
    # Filter and sort by filename (without extension)
    names = sorted(f.split(".")[0] for f in files)
    
    for name in names:
        print(f"Starting index build for {name}")
        work_dir = f"./index/index_time/{name}"
        os.makedirs(work_dir, exist_ok=True)
        file_path = os.path.join(directory, f"{name}.md")
        insert(work_dir, file_path, str(name))
        print(f"Completed index for {name}")

if __name__ == "__main__":
    # Example usage: process all files in './Dataset/audi_md'
    dataset_dir = './Dataset/audi_md'
    batch_insert(dataset_dir)
