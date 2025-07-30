import os
import sys
import logging
import argparse
from openai import AsyncOpenAI
from T_GRAG import GraphRAG, QueryParam
from T_GRAG.base import BaseKVStorage
from T_GRAG._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer
import numpy as np
from T_GRAG.prompt import PROMPTS
import asyncio
import re
import json
sys.path.append("..")

logging.basicConfig(level=logging.WARNING)  # Set minimum log level to WARNING
logging.getLogger("time_graphrag").setLevel(logging.INFO)  # Only INFO and above will be logged by this logger

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

async def model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Try to retrieve cached response
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

    # Cache the response
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

# Ask LLM to determine the relevant time period
def llm_query_time(question):
    prompt = PROMPTS["time"].format(question=question)
    return asyncio.run(model_if_cache(prompt))

# Extract year-question mapping from LLM output
def deal_llm_output(llm_answer):
    pattern = r'\[([^\[\]]+<SEP>[^\[\]]+)\]'
    matches = re.findall(pattern, llm_answer)
    question_dict = {}
    if matches:
        for item in matches:
            item_pattern = r'(\d{4}-\d{4}|\d{4})<SEP>(.*)'
            match = re.search(item_pattern, item)
            if match:
                year = match.group(1)
                question = match.group(2)
                if year in question_dict:
                    question_dict[year] += question
                else:
                    question_dict[year] = question
        return question_dict
    else:
        return "Error: No matching strings found."

def query(question, query_time, type):
    # Remove previous run files
    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    remove_if_exist(f"{WORKING_DIR}/kv_store_llm_response_cache.json")
    
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=model_if_cache,
        cheap_model_func=model_if_cache,
        embedding_func=local_embedding
    )
    print("Starting time-specific KG retrieval")
    rag.search_graph(param=QueryParam(time=query_time, mode=int(type)))
    
    print("Executing query")
    answer = rag.query(
        question,
        param=QueryParam(time=query_time, mode=int(type))
    )
    
    return answer

def process_queries(input_json, output_filename):
    output = {}
    batch_counter = 0  # Tracks number of queries processed
    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            output = json.load(f)
    for key, value in input_json.items():
        if key in output:
            continue

        print_output_file = f'./Print/rag_2time/{key}.txt'
        # Ensure the print file exists or clear its contents
        if not os.path.exists(print_output_file):
            with open(print_output_file, "w"):
                pass
        else:
            with open(print_output_file, "w"):
                pass

        # Redirect stdout to the print file
        with open(print_output_file, "a") as f:
            sys.stdout = f
            batch_counter += 1
            output[key] = value
            question = value["question"]

            print(f"Begin processing {key}")
            llm_answer = llm_query_time(question)
            print(f"LLM determined time input: {llm_answer}")

            question_dict = deal_llm_output(llm_answer)
            print(f"Parsed sub-questions: {question_dict}")

            if not isinstance(question_dict, dict):
                output[key]['llm_answer'] = "Time parsing error"
                continue

            qa_dict = {}
            i = 1
            for time, single_question in question_dict.items():
                query_time = []
                if '-' in time:
                    start, end = time.split('-')
                    for year in range(int(start), int(end) + 1):
                        query_time.append(str(year))
                else:
                    query_time.append(str(time))

                print(f"Time point: {query_time}")
                print(f"Question: {single_question}")
                single_answer = query(single_question, query_time, 2)
                print(f"Answer: {single_answer}")

                qa_dict[f"sub question-{i}"] = {
                    'sub question': single_question,
                    'answer': single_answer
                }
                i += 1

            print("Compiling final answer")
            print(f"Sub-question QA dict: {qa_dict}")
            prompt = PROMPTS['final answer']
            final_prompt = prompt.format(question=question, qa_dict=qa_dict)
            final_answer = asyncio.run(model_if_cache(final_prompt))
            output[key]['llm_answer'] = final_answer

            print(f"Completed {key}")
            print(f"Final model answer: {final_answer}")

            # Save progress every 10 queries
            if batch_counter >= 10:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)
                print(f"Saved {batch_counter} processed questions to {output_filename}")
                batch_counter = 0

            sys.stdout = sys.__stdout__

    if batch_counter > 0:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"Saved remaining {batch_counter} questions to {output_filename}")

    print(f"All processing complete, results saved to {output_filename}")

if __name__ == "__main__":
    input_file = './Dataset/QA/QA_2time.json'  # Path to input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        input_json = json.load(f)

    output_filename = './Answer/2time_answer.json'
    process_queries(input_json, output_filename)
