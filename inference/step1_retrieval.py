import argparse
import copy
import hashlib
import hmac
import json
import re
import time
from typing import List, Dict

import requests
from tqdm import tqdm

from utils import load_jsonl, save_jsonl


def get_md5(my_string):
    hash_object = hashlib.md5()
    hash_object.update(my_string.encode())
    md5_hash = hash_object.hexdigest()
    return md5_hash


def get_hmac_sha256_search_plan(body, url, key='799de706dba4bac8047e2c25e741ed1c5c46e3d966a426c5b0afa271b43fc490'):
    timestamp = int(time.time() * 1000)
    timestamp = str(timestamp)
    body = body.replace("\n", "")
    key = key.encode('utf-8')
    value = "timestamp=" + timestamp + "&url=" + url + "&body=" + body
    HAMCObj = hmac.new(key, value.encode('utf-8'), hashlib.sha256)
    auth = HAMCObj.hexdigest()
    return timestamp, auth


def get_search_result(query, sn_prefix, env="zc", debug=False, url=""):
    service_url = ""
    if env == "zc":
        service_url = "http://10.97.130.5:3399/search-gpt"
    elif env == "effect":
        service_url = "http://10.97.130.186:3399/search-gpt"

    if url:
        service_url = url

    sn = sn_prefix + get_md5(query)
    body = {
        "query": query,
        "sn": sn,
        "from": "CC002600",
        "sregion": "cn",
        "need_box_ids": [
            "108", "1801", "1311", "109", "200", "9003", "9007", "2601", "2602",
            "11004", "1200", "1221", "1260", "1430", "1002", "1000", "1001", "1112",
            "1113", "1114", "105", "1301", "1448", "1449", "1012", "1014", "1013",
            "1011", "2100", "2102", "2103", "1211", "1230", "1310", "1250", "8620",
            "8621", "8622", "8628", "1051", "4451", "1053", "8400", "8401", "8402",
            "8410", "8411", "1900044", "1900045", "1900046", "1900015", "1900011",
            "1900071", "1900006", "1900070", "1900013", "1900040", "1900023", "9995",
            "7720", "7721", "7722", "7723", "7724", "1900043", "2602", "9010"
        ],
        "device": {
            "vendor": "HUAWEI"
        },
        "extra_info": {}
    }

    if debug:
        body["extra_info"]["debug_search"] = True

    payload = json.dumps(body)
    sing1, sing2 = get_hmac_sha256_search_plan(payload, "/search-gpt")
    headers_ = {
        'Authorization': sing2,
        'timestamp': sing1,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", service_url, headers=headers_, data=payload)
    result = response.content.decode('utf-8')
    answer = json.loads(result)
    return answer


def extract_passages(search_result: Dict, top_k: int = 3) -> List[str]:
    passages = []
    try:
        if "results" in search_result:
            for item in search_result["results"][:top_k]:
                if "content" in item:
                    passages.append(item["content"])
                elif "text" in item:
                    passages.append(item["text"])
                elif "snippet" in item:
                    passages.append(item["snippet"])

        elif "items" in search_result:
            for item in search_result["items"][:top_k]:
                if "content" in item:
                    passages.append(item["content"])
                elif "text" in item:
                    passages.append(item["text"])

        if not passages:
            passages.append(json.dumps(search_result, ensure_ascii=False))

    except Exception as e:
        print(f"Error extracting passages: {e}")
        passages = []

    return passages[:top_k]


def retrieve_context(query: str, sn_prefix: str, top_k: int = 3, env: str = "zc", debug: bool = False, url: str = "") -> List[str]:
    search_result = get_search_result(query=query, sn_prefix=sn_prefix, env=env, debug=debug, url=url)
    passages = extract_passages(search_result, top_k=top_k)
    return passages


def retrieve_for_sub_questions(question: str, sub_questions: List[Dict], sn_prefix: str, top_k: int = 10, env: str = "zc", debug: bool = False) -> Dict:
    retrieval_results = {
        "question": question,
        "sub_questions": sub_questions,
        "retrieved_passages": {}
    }

    for subq_dict in sub_questions:
        q_label = subq_dict["label"]
        q_text = subq_dict["text"]

        try:
            passages = retrieve_context(query=q_text, sn_prefix=sn_prefix, top_k=top_k, env=env, debug=debug)
            retrieval_results["retrieved_passages"][q_label] = {
                "query": q_text,
                "passages": passages
            }
            print(f"Retrieved {len(passages)} passages for {q_label}: {q_text[:50]}...")

        except Exception as e:
            print(f"Error retrieving for {q_label}: {e}")
            retrieval_results["retrieved_passages"][q_label] = {
                "query": q_text,
                "passages": [],
                "error": str(e)
            }

    return retrieval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--sn_prefix", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--search_env", type=str, default="zc", choices=["zc", "effect"])
    parser.add_argument("--search_debug", action="store_true")
    args = parser.parse_args()

    if args.input_file:
        input_path = args.input_file
    else:
        dataset = args.dataset.split("-")[0] if "-" in args.dataset else args.dataset
        input_path = f"{args.save_dir}/{dataset}/prompts_decompose_test_t0.0_{args.expname}/generate.jsonl"

    if args.output_file:
        output_path = args.output_file
    else:
        dataset = args.dataset.split("-")[0] if "-" in args.dataset else args.dataset
        output_path = f"{args.save_dir}/{dataset}/prompts_decompose_test_{args.expname}/retrieved_k{args.k}.jsonl"

    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Search env: {args.search_env}, top-{args.k}")
    print("=" * 80)

    questions = load_jsonl(input_path)
    print(f"Loaded {len(questions)} questions")

    results = []
    for index, item in enumerate(tqdm(questions, desc="Retrieval")):
        try:
            retrieval_result = retrieve_for_sub_questions(
                question=item["question"],
                sub_questions=item["decomposed"],
                sn_prefix=args.sn_prefix,
                top_k=args.k,
                env=args.search_env,
                debug=args.search_debug
            )

            result_item = copy.deepcopy(item)
            result_item["index"] = index
            result_item["retrieved_passages"] = retrieval_result["retrieved_passages"]
            results.append(result_item)

        except Exception as e:
            print(f"Error at index {index}: {e}")
            result_item = copy.deepcopy(item)
            result_item["index"] = index
            result_item["retrieved_passages"] = {}
            result_item["error"] = str(e)
            results.append(result_item)
            continue

    save_jsonl(results, output_path)
    print("=" * 80)
    print(f"Completed: {len(results)} questions")
    print(f"Saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
