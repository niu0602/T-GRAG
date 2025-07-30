import os
import sys
import logging
from openai import AsyncOpenAI
from T_GRAG import GraphRAG, QueryParam
from T_GRAG.base import BaseKVStorage
from T_GRAG._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
import numpy as np
from T_GRAG.prompt import PROMPTS
import asyncio
import re, json
import argparse
sys.path.append("..")

logging.basicConfig(level=logging.WARNING)  # Set minimum logging level to WARNING
logging.getLogger("T_GRAG").setLevel(logging.INFO)  # Set this logger to INFO level and above

WORKING_DIR = "./index/merge"

def parse_args():
    parser = argparse.ArgumentParser(description="Time-GRAPH batch graph construction script")
    parser.add_argument("--api-key",    default=os.getenv("OPENAI_API_KEY", ""),    help="OpenAI API key")
    parser.add_argument("--base-url",   default=os.getenv("OPENAI_API_BASE", ""),   help="OpenAI API base URL")
    parser.add_argument("--model",      default=os.getenv("LLM_MODEL", "qwen"),    help="Chat model name")
    parser.add_argument("--embed-dir",  default=os.getenv("EMBED_MODEL_DIR", "./models/embed"), help="Local SentenceTransformer model directory")
    parser.add_argument("--log-level",  default=os.getenv("LOG_LEVEL", "WARNING"), help="Logging level: DEBUG/INFO/WARNING/ERROR")
    return parser.parse_args()

args     = parse_args()
API_KEY  = args.api_key
BASE_URL = args.base_url
MODEL    = args.model

async def model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    openai_async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Attempt to retrieve cached response
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if available
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )

    return response.choices[0].message.content

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

EMBED_MODEL = SentenceTransformer(args.embed_dir, trust_remote_code=True, device="cuda:0")

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str], query=None) -> np.ndarray:
    if query is None:
        return EMBED_MODEL.encode(texts, normalize_embeddings=True)
    elif query is True:
        return EMBED_MODEL.encode(texts, prompt_name="s2p_query", normalize_embeddings=True)

# LLM determines the relevant time period
def llm_query_time(question):
    prompt = PROMPTS["time"].format(question=question)
    return asyncio.run(model_if_cache(prompt))

# Parse the time response from the LLM
def choose_time(LLM_time):
    match = re.search(r'time=([^\s,]+),\s*type=(\d+)', LLM_time)
    if match:
        time_value = match.group(1)
        type_value = int(match.group(2))
        return {'time': time_value, 'type': type_value}
    else:
        print("LLM response does not contain the correct format")
        return {'time': 0, 'type': 0}

def query(question, query_time, type):
    # Clean up previous run artifacts
    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    remove_if_exist(f"{WORKING_DIR}/kv_store_llm_response_cache.json")
    
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=model_if_cache,
        cheap_model_func=model_if_cache,
        embedding_func=local_embedding
    )
    print("Starting retrieval of time-specific KG")
    rag.search_graph(param=QueryParam(time=query_time, mode=int(type)))
    
    print("Starting query")
    answer = rag.query(
        question, 
        param=QueryParam(time=query_time, mode=int(type), local_max_token_for_text_unit=5200, top_k=30)
    )
    
    return answer

def process_queries(input_json, output_filename, name):
    output = {}
    batch_counter = 0
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            output = json.load(f)  # Load already processed entries

    for key, value in input_json.items():
        if key in output:
            continue

        print_output_file = f'./Print/rag_1time/{name}/{key}.txt'
        # Ensure the output file exists, creating or clearing it
        if not os.path.exists(print_output_file):
            os.makedirs(os.path.dirname(print_output_file), exist_ok=True)
            with open(print_output_file, "w"):
                pass
        else:
            with open(print_output_file, "w"):
                pass

        # Redirect stdout to the print file for logging
        with open(print_output_file, "a") as f:
            sys.stdout = f
            batch_counter += 1
            output[key] = value
            question = value["Question"]

            print(f"Starting answer for {key}")
            # Ask the LLM for the required time period
            LLM_answer = llm_query_time(question)
            print(f"LLM determined time needed: {LLM_answer}")

            # Parse the LLM's response
            time = choose_time(LLM_answer)['time']
            type = choose_time(LLM_answer)['type']

            if type == 1:
                query_time = str(time)
                print(f"Starting single timestamp retrieval: {time}")
            elif type == 2:
                query_time = str(time).split("<SEP>")
                if len(query_time) > 1:
                    result = []
                    for item in query_time:
                        try:
                            if '-' in item:
                                start, end = item.split('-')
                                for year in range(int(start), int(end) + 1):
                                    result.append(str(year))
                            else:
                                result.append(item)
                        except ValueError:
                            print(f"Error: {time} could not be processed correctly, skipping")
                            continue
                    query_time = list(dict.fromkeys(result))
                    print(f"Starting multiple timestamp retrieval: {query_time}")
                else:
                    print(f"Incorrect format for time: {time}")
            elif type == 3:
                if "<SEP>" in time:
                    query_time = str(time).split("<SEP>")
                else:
                    try:
                        start_year, end_year = map(int, str(time).split('-'))
                        query_time = [str(year) for year in range(start_year, end_year + 1)]
                        print(f"Starting time interval retrieval: {query_time}")
                    except ValueError:
                        print(f"Error: {time} could not be processed correctly, skipping")
                        continue
            elif type == 4:
                try:
                    start_year, end_year = map(int, str(time).split('-'))
                    query_time = [str(year) for year in range(start_year, end_year + 1)]
                    print(f"Starting fuzzy time retrieval: {query_time}")
                except ValueError:
                    print(f"Error: {time} could not be processed correctly, skipping")
                    continue
            else:
                print(f"Time parsing error for {key}: {time}, {type}")
                continue

            answer = query(question, query_time, type)
            output[key]['llm_answer'] = answer

            print(f"Completed answer for {key}")
            print(f"Large model answer: {answer}")

            # Save progress every 10 queries
            if batch_counter >= 10:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)
                print(f"Processed {batch_counter} questions, results saved to {output_filename}")
                batch_counter = 0

            sys.stdout = sys.__stdout__

    if batch_counter > 0:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"Remaining {batch_counter} questions saved to {output_filename}")

    print(f"Processing complete, results saved to {output_filename}")

if __name__ == "__main__":
    input_folder = "./Dataset/QA/audi_one_time_QA"
    output_folder = './Answer/1time_answer'
    for filename in os.listdir(input_folder):
        print(f"Starting processing of {filename}")
        input_file = os.path.join(input_folder, filename)
        name = os.path.splitext(filename)[0]
        output_filename = os.path.join(output_folder, f"{name}_answer.json")
        with open(input_file, 'r', encoding='utf-8') as f:
            input_json = json.load(f)
        process_queries(input_json, output_filename, name)
