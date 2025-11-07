import argparse
import copy
import csv
import json
import numpy as np
from typing import List, Dict
from tqdm import tqdm, trange
from utils import load_tokenizer, init_llm, make_sampling_params, load_jsonl, save_jsonl, call_llm
import re

# 导入新的搜索服务模块
from search_service import retrieve_context_from_api


# Import other retrieval/embedding code as needed
def replace_placeholders(question_text: str, answers_so_far: Dict[str, str]) -> str:
    """
    Replaces placeholders like "#1", "#2", etc. in the question text with answers from previous sub-questions.
    """
    matches = re.findall(r"#(\d+)", question_text)
    for m in matches:
        placeholder = f"#{m}"
        q_key = f"Q{m}"
        if q_key in answers_so_far:
            question_text = question_text.replace(placeholder, answers_so_far[q_key])
    return question_text


def retrieve_context(query: str, sn_prefix: str, top_k: int = 3, env: str = "zc", debug: bool = False) -> List[str]:
    """
    使用新的搜索API检索上下文段落

    Args:
        query: 搜索查询
        sn_prefix: 工号前缀
        top_k: 返回的文档数量（默认3）
        env: 环境选择（"zc" 或 "effect"）
        debug: 是否开启debug模式

    Returns:
        检索到的文本段落列表
    """
    passages = retrieve_context_from_api(
        query=query,
        sn_prefix=sn_prefix,
        top_k=top_k,
        env=env,
        debug=debug
    )
    return passages


def load_data(dataset: str, expname: str, save_dir: str) -> List[Dict]:
    """
    Loads questions from a JSONL file for the given dataset.
    """
    questions = []
    if "-" in dataset:
        dataset = dataset.split("-")[0]
    with open(f"{save_dir}/{dataset}/prompts_decompose_test_t0.0_{expname}/generate.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data)
    print("========")
    print(f"Loaded {len(questions)} examples from {dataset}!")
    print("========")
    return questions


###################################
# Answering Functions
###################################
def zigzag_visit(lst: List) -> List:
    """
    Reorders the input list in a zigzag fashion.
    Example:
        Input: [1, 2, 3, 4, 5, 6, 7]
        Output: [1, 3, 5, 7, 6, 4, 2]
    """
    n = len(lst)
    result = [None] * n

    # Fill first half (odd indices)
    i, j = 0, 0
    while j < (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j += 1

    # Fill second half (even indices)
    i = 1
    j = n - 1
    while j >= (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j -= 1

    return result

def answer_sub_claim(sub_q: str, context_passages: List[str], model, tokenizer, sampling_params) -> str:
    """
    Uses the LLM to answer a sub-question given the retrieved context.
    The context passages are reordered in a zigzag manner before being concatenated.
    """
    reordered_passages = zigzag_visit(context_passages)
    context_text = "\n\n".join(reordered_passages)
    prompt = f"""You have the following context passages:
{context_text}

Please verify whether the claim '{sub_q}' is correct using the context as reference.
If no answer is found in the context, use your own knowledge.
Please only output Yes or No and do not give any explanation."""

    response = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params)
    return response.strip()

def answer_sub_question(sub_q: str, context_passages: List[str], model, tokenizer, sampling_params) -> str:
    """
    Uses the LLM to answer a sub-question given the retrieved context.
    The context passages are reordered in a zigzag manner before being concatenated.
    """
    reordered_passages = zigzag_visit(context_passages)
    context_text = "\n\n".join(reordered_passages)
    prompt = f"""You have the following context passages:
{context_text}

Please answer the question '{sub_q}' with a short span using the context as reference.
If no answer is found in the context, use your own knowledge. Your answer needs to be as short as possible."""
    response = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params)
    return response.strip()


def multi_turn_qa(question: str, sub_questions: List[Dict], sn_prefix: str,
                  llm_model, llm_tokenizer, sampling_params, dataset: str,
                  add_passage: int, topk: int, env: str = "zc", debug: bool = False):
    """
    Orchestrates the multi-turn QA process using the new search API:
    1. Resolve any placeholder references in sub-questions.
    2. Retrieve context using the new search API.
    3. Answer each sub-question.
    4. Combine sub-answers into a final answer.

    Args:
        question: 原始问题
        sub_questions: 分解后的子问题列表
        sn_prefix: 工号前缀（用于搜索API）
        llm_model: 语言模型
        llm_tokenizer: tokenizer
        sampling_params: 采样参数
        dataset: 数据集名称
        add_passage: 是否添加段落到最终答案
        topk: 检索的文档数量
        env: 搜索API环境（"zc" 或 "effect"）
        debug: 是否开启搜索API的debug模式

    Returns:
        final_answer: 最终答案
        answer_dict: 子问题答案字典
        passage_dict: 检索到的段落字典
    """
    # Create dictionaries to hold resolved sub-questions and answers
    subquestions_dict = {subq_dict["label"]: subq_dict["text"] for subq_dict in sub_questions}
    answer_dict = {}
    passage_dict = {}
    all_passages = []

    # Process each sub-question
    for subq_dict in sub_questions:
        q_label = subq_dict["label"]
        q_text = subq_dict["text"]

        # Replace placeholders (e.g., #1, #2) with previous answers
        q_text_resolved = replace_placeholders(q_text, answer_dict)

        # Retrieve context using the new search API
        passages = retrieve_context(
            query=q_text_resolved,
            sn_prefix=sn_prefix,
            top_k=topk,
            env=env,
            debug=debug
        )

        all_passages += passages[:5] if len(sub_questions) <= 3 else passages[:3]
        all_passages = list(set(all_passages))

        # Answer the sub-question
        sub_answer = answer_sub_question(q_text_resolved, passages, llm_model, llm_tokenizer, sampling_params)
        answer_dict[q_label] = sub_answer
        passage_dict[q_label] = passages
        subquestions_dict[q_label] = q_text_resolved

    # Generate final answer based on sub-answers
    final_answer = generate_final_answer(question, subquestions_dict, answer_dict,
                                         llm_model, llm_tokenizer, sampling_params, dataset, all_passages, add_passage)
    print("-------\nquestion:", question,
          "\nsub-questions:", sub_questions,
          "\nanswers:", answer_dict,
          "\nFinal Answer:", final_answer)
    return final_answer, answer_dict, passage_dict


def generate_final_answer(original_question: str, sub_questions: Dict[str, str], sub_answers: Dict[str, str], model, tokenizer, sampling_params, dataset: str, passages: List[str] = None, add_passage: int = 1) -> str:
    """
    Generates a final answer for the original question by summarizing sub-question answers.
    """
    sub_answer_text = "\n".join([f"### {k}: {sub_questions[k]}, Answer for {k}: {v}" for k, v in sub_answers.items()])
    final_prompt = "a short span"

    if dataset in ["hover", "exfever"]:
        prompt = f"""You are given some subquestions and their answers:
{sub_answer_text}

Please verify the correctness of the claim: '{original_question}' using the subquestions as reference. Please provide a concise and clear reasoning followed by a concise conclusion. Your answer should be Yes or No only.
Wrap your answer with <answer> and </answer> tags."""

    else:
        if add_passage:
            passages = "\n\n".join(list(set(passages)))
            prompt = f"""You have the following passages:
{passages}

You are also given some subquestions and their answers:
{sub_answer_text}

Please answer the question '{original_question}' with {final_prompt} using the documents and subquestions as reference.
Make sure your response is grounded in documents and provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge.
Wrap your answer with <answer> and </answer> tags."""
        else:
            prompt = f"""You are given some subquestions and their answers:
{sub_answer_text}

Please answer the question '{original_question}' with {final_prompt} using the subquestions as reference. Provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge.
Wrap your answer with <answer> and </answer> tags."""

    final = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params)
    return final.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_path", type=str, required=True, help="LLM模型路径")
    parser.add_argument("--llm_tokenizer", type=str, required=True, help="LLM tokenizer路径")
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="数据集名称")
    parser.add_argument("--expname", type=str, default="", help="实验名称")
    parser.add_argument("--save_dir", type=str, default="", help="保存目录")
    parser.add_argument("--top_p", type=float, default=0.99, help="采样top_p参数")
    parser.add_argument("--k", type=int, default=10, help="检索文档数量")
    parser.add_argument("--add_passage", type=int, default=1, help="是否添加段落到最终答案")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小")

    # 新增搜索API相关参数
    parser.add_argument("--sn_prefix", type=str, required=True, help="工号前缀（用于搜索API）")
    parser.add_argument("--search_env", type=str, default="zc", choices=["zc", "effect"],
                        help="搜索环境：zc=众测环境，effect=效果环境")
    parser.add_argument("--search_debug", action="store_true", help="开启搜索API的debug模式")

    args = parser.parse_args()

    # 加载问题数据
    questions = load_jsonl(f"{args.save_dir}/{args.dataset}/prompts_decompose_test_t0.0_{args.expname}/generate.jsonl")

    # 初始化LLM
    llm_tokenizer = load_tokenizer(args.llm_tokenizer)
    sampling_params = make_sampling_params(args.temperature, args.top_p, max_tokens=512)
    llm = init_llm(args.llm_model_path, args.tensor_parallel_size)

    saved_examples = []
    for index, item in enumerate(tqdm(questions)):
        try:
            final_answer, intermediate_answers, intermediate_passages = multi_turn_qa(
                question=item["question"],
                sub_questions=item["decomposed"],
                sn_prefix=args.sn_prefix,
                llm_model=llm,
                llm_tokenizer=llm_tokenizer,
                sampling_params=sampling_params,
                dataset=args.dataset,
                add_passage=args.add_passage,
                topk=args.k,
                env=args.search_env,
                debug=args.search_debug
            )
            new_item = copy.deepcopy(item)
            new_item.update({
                "index": index,
                "final_answer": final_answer,
                "intermediate_answers": intermediate_answers,
                "intermediate_passages": intermediate_passages
            })
            saved_examples.append(new_item)
        except Exception as e:
            print(f"Error at {index}: {e}")
            continue

    # 保存结果
    output_path = f"{args.save_dir}/{args.dataset}/prompts_decompose_test_{args.expname}/test_searchapi_k{args.k}_passage{args.add_passage}.jsonl"
    save_jsonl(saved_examples, output_path)
    print(f"Saved {len(saved_examples)} results to {output_path}")

if __name__ == "__main__":
    main()
