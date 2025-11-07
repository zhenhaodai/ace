# encoding=utf-8
"""
步骤2: LLM生成答案脚本
读取步骤1的检索结果，使用LLM生成答案
"""
import argparse
import copy
import json
import re
from typing import List, Dict
from tqdm import tqdm

from utils import load_tokenizer, init_llm, make_sampling_params, load_jsonl, save_jsonl, call_llm


def replace_placeholders(question_text: str, answers_so_far: Dict[str, str]) -> str:
    """
    替换问题中的占位符，如 #1, #2 等
    """
    matches = re.findall(r"#(\d+)", question_text)
    for m in matches:
        placeholder = f"#{m}"
        q_key = f"Q{m}"
        if q_key in answers_so_far:
            question_text = question_text.replace(placeholder, answers_so_far[q_key])
    return question_text


def zigzag_visit(lst: List) -> List:
    """
    以之字形方式重排列表
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


def answer_sub_question(sub_q: str, context_passages: List[str], model, tokenizer, sampling_params) -> str:
    """
    使用LLM回答子问题
    """
    if not context_passages:
        # 如果没有检索到文档，直接使用LLM的知识
        prompt = f"""Please answer the question '{sub_q}' with a short span.
Your answer needs to be as short as possible."""
    else:
        reordered_passages = zigzag_visit(context_passages)
        context_text = "\n\n".join(reordered_passages)
        prompt = f"""You have the following context passages:
{context_text}

Please answer the question '{sub_q}' with a short span using the context as reference.
If no answer is found in the context, use your own knowledge. Your answer needs to be as short as possible."""

    response = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params)
    return response.strip()


def generate_final_answer(
    original_question: str,
    sub_questions: Dict[str, str],
    sub_answers: Dict[str, str],
    model,
    tokenizer,
    sampling_params,
    dataset: str,
    passages: List[str] = None,
    add_passage: int = 1
) -> str:
    """
    生成最终答案
    """
    sub_answer_text = "\n".join([f"### {k}: {sub_questions[k]}, Answer for {k}: {v}" for k, v in sub_answers.items()])
    final_prompt = "a short span"

    if dataset in ["hover", "exfever"]:
        prompt = f"""You are given some subquestions and their answers:
{sub_answer_text}

Please verify the correctness of the claim: '{original_question}' using the subquestions as reference. Please provide a concise and clear reasoning followed by a concise conclusion. Your answer should be Yes or No only.
Wrap your answer with <answer> and </answer> tags."""

    else:
        if add_passage and passages:
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


def process_with_retrieved_passages(
    item: Dict,
    llm_model,
    llm_tokenizer,
    sampling_params,
    dataset: str,
    add_passage: int,
    topk: int
) -> Dict:
    """
    使用检索到的段落和LLM生成答案

    Args:
        item: 包含问题、子问题和检索结果的字典
        llm_model: 语言模型
        llm_tokenizer: tokenizer
        sampling_params: 采样参数
        dataset: 数据集名称
        add_passage: 是否在最终答案中添加段落
        topk: 使用的段落数量

    Returns:
        包含答案的字典
    """
    question = item["question"]
    sub_questions = item["decomposed"]
    retrieved_passages = item.get("retrieved_passages", {})

    # 创建字典保存子问题和答案
    subquestions_dict = {subq_dict["label"]: subq_dict["text"] for subq_dict in sub_questions}
    answer_dict = {}
    passage_dict = {}
    all_passages = []

    # 处理每个子问题
    for subq_dict in sub_questions:
        q_label = subq_dict["label"]
        q_text = subq_dict["text"]

        # 替换占位符（使用前面子问题的答案）
        q_text_resolved = replace_placeholders(q_text, answer_dict)

        # 获取该子问题的检索结果
        retrieved_info = retrieved_passages.get(q_label, {})
        passages = retrieved_info.get("passages", [])[:topk]

        # 保存段落用于最终答案
        if len(sub_questions) <= 3:
            all_passages += passages[:5]
        else:
            all_passages += passages[:3]
        all_passages = list(set(all_passages))

        # 使用LLM回答子问题
        sub_answer = answer_sub_question(
            q_text_resolved,
            passages,
            llm_model,
            llm_tokenizer,
            sampling_params
        )

        answer_dict[q_label] = sub_answer
        passage_dict[q_label] = passages
        subquestions_dict[q_label] = q_text_resolved

    # 生成最终答案
    final_answer = generate_final_answer(
        question,
        subquestions_dict,
        answer_dict,
        llm_model,
        llm_tokenizer,
        sampling_params,
        dataset,
        all_passages,
        add_passage
    )

    print("-------\nquestion:", question,
          "\nsub-questions:", sub_questions,
          "\nanswers:", answer_dict,
          "\nFinal Answer:", final_answer)

    return {
        "final_answer": final_answer,
        "intermediate_answers": answer_dict,
        "intermediate_passages": passage_dict
    }


def main():
    parser = argparse.ArgumentParser(description="步骤2: 使用检索结果生成答案")

    # 输入输出路径
    parser.add_argument("--input_file", type=str, required=True,
                        help="步骤1的输出文件（包含检索结果的jsonl文件）")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径（如果不指定，在输入文件名基础上生成）")

    # LLM参数
    parser.add_argument("--llm_model_path", type=str, required=True,
                        help="LLM模型路径")
    parser.add_argument("--llm_tokenizer", type=str, required=True,
                        help="LLM tokenizer路径")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.99,
                        help="采样top_p参数")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="张量并行大小")

    # 任务参数
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        help="数据集名称")
    parser.add_argument("--k", type=int, default=10,
                        help="使用的检索文档数量")
    parser.add_argument("--add_passage", type=int, default=1,
                        help="是否在最终答案中添加段落")

    args = parser.parse_args()

    # 确定输出文件路径
    if args.output_file:
        output_path = args.output_file
    else:
        # 在输入文件名基础上添加后缀
        input_base = args.input_file.replace(".jsonl", "")
        output_path = f"{input_base}_answers.jsonl"

    print("=" * 80)
    print(f"步骤2: LLM生成答案")
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {output_path}")
    print(f"LLM模型: {args.llm_model_path}")
    print(f"使用 top-{args.k} 检索文档")
    print("=" * 80)

    # 加载检索结果
    items = load_jsonl(args.input_file)
    print(f"加载了 {len(items)} 个问题及其检索结果")

    # 初始化LLM
    print("正在初始化LLM模型...")
    llm_tokenizer = load_tokenizer(args.llm_tokenizer)
    sampling_params = make_sampling_params(args.temperature, args.top_p, max_tokens=512)
    llm = init_llm(args.llm_model_path, args.tensor_parallel_size)
    print("LLM模型初始化完成")

    # 处理每个问题
    results = []
    for index, item in enumerate(tqdm(items, desc="生成答案")):
        try:
            # 生成答案
            answer_result = process_with_retrieved_passages(
                item=item,
                llm_model=llm,
                llm_tokenizer=llm_tokenizer,
                sampling_params=sampling_params,
                dataset=args.dataset,
                add_passage=args.add_passage,
                topk=args.k
            )

            # 合并结果
            result_item = copy.deepcopy(item)
            result_item.update({
                "final_answer": answer_result["final_answer"],
                "intermediate_answers": answer_result["intermediate_answers"],
                "intermediate_passages": answer_result["intermediate_passages"]
            })

            results.append(result_item)

        except Exception as e:
            print(f"Error at index {index}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存结果
    save_jsonl(results, output_path)
    print("=" * 80)
    print(f"✅ 答案生成完成！")
    print(f"成功处理 {len(results)} 个问题")
    print(f"结果已保存到: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
