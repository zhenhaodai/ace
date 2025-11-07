# encoding=utf-8
"""
步骤1: 搜索检索脚本
读取分解后的问题，对每个子问题调用搜索API，保存检索结果
"""
import argparse
import copy
import json
import re
from typing import List, Dict
from tqdm import tqdm

from search_service import retrieve_context_from_api
from utils import load_jsonl, save_jsonl


def replace_placeholders(question_text: str, answers_so_far: Dict[str, str]) -> str:
    """
    替换问题中的占位符，如 #1, #2 等
    注意：在搜索阶段，我们无法替换占位符（因为还没有答案）
    这个函数保留用于后续可能的扩展
    """
    matches = re.findall(r"#(\d+)", question_text)
    for m in matches:
        placeholder = f"#{m}"
        q_key = f"Q{m}"
        if q_key in answers_so_far:
            question_text = question_text.replace(placeholder, answers_so_far[q_key])
    return question_text


def retrieve_for_sub_questions(
    question: str,
    sub_questions: List[Dict],
    sn_prefix: str,
    top_k: int = 10,
    env: str = "zc",
    debug: bool = False
) -> Dict:
    """
    为所有子问题检索相关文档

    Args:
        question: 原始主问题
        sub_questions: 分解后的子问题列表
        sn_prefix: 工号前缀
        top_k: 每个子问题检索的文档数量
        env: 搜索环境
        debug: 是否开启debug模式

    Returns:
        包含所有检索结果的字典
    """
    retrieval_results = {
        "question": question,
        "sub_questions": sub_questions,
        "retrieved_passages": {}
    }

    # 对每个子问题进行检索
    for subq_dict in sub_questions:
        q_label = subq_dict["label"]
        q_text = subq_dict["text"]

        try:
            # 调用搜索API
            passages = retrieve_context_from_api(
                query=q_text,
                sn_prefix=sn_prefix,
                top_k=top_k,
                env=env,
                debug=debug
            )

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
    parser = argparse.ArgumentParser(description="步骤1: 检索子问题的相关文档")

    # 输入输出路径
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--expname", type=str, default="", help="实验名称")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    parser.add_argument("--input_file", type=str, default=None,
                        help="输入文件路径（如果不指定，使用默认路径）")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径（如果不指定，使用默认路径）")

    # 搜索API参数
    parser.add_argument("--sn_prefix", type=str, required=True,
                        help="工号前缀（用于搜索API）")
    parser.add_argument("--k", type=int, default=10,
                        help="每个子问题检索的文档数量")
    parser.add_argument("--search_env", type=str, default="zc",
                        choices=["zc", "effect"],
                        help="搜索环境：zc=众测环境，effect=效果环境")
    parser.add_argument("--search_debug", action="store_true",
                        help="开启搜索API的debug模式")

    args = parser.parse_args()

    # 确定输入文件路径
    if args.input_file:
        input_path = args.input_file
    else:
        dataset = args.dataset.split("-")[0] if "-" in args.dataset else args.dataset
        input_path = f"{args.save_dir}/{dataset}/prompts_decompose_test_t0.0_{args.expname}/generate.jsonl"

    # 确定输出文件路径
    if args.output_file:
        output_path = args.output_file
    else:
        dataset = args.dataset.split("-")[0] if "-" in args.dataset else args.dataset
        output_path = f"{args.save_dir}/{dataset}/prompts_decompose_test_{args.expname}/retrieved_k{args.k}.jsonl"

    print("=" * 80)
    print(f"步骤1: 搜索检索")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"搜索环境: {args.search_env}")
    print(f"每个子问题检索 top-{args.k} 文档")
    print("=" * 80)

    # 加载问题数据
    questions = load_jsonl(input_path)
    print(f"加载了 {len(questions)} 个问题")

    # 处理每个问题
    results = []
    for index, item in enumerate(tqdm(questions, desc="检索进度")):
        try:
            # 获取检索结果
            retrieval_result = retrieve_for_sub_questions(
                question=item["question"],
                sub_questions=item["decomposed"],
                sn_prefix=args.sn_prefix,
                top_k=args.k,
                env=args.search_env,
                debug=args.search_debug
            )

            # 合并原始数据和检索结果
            result_item = copy.deepcopy(item)
            result_item["index"] = index
            result_item["retrieved_passages"] = retrieval_result["retrieved_passages"]

            results.append(result_item)

        except Exception as e:
            print(f"Error at index {index}: {e}")
            # 即使出错也保存原始数据
            result_item = copy.deepcopy(item)
            result_item["index"] = index
            result_item["retrieved_passages"] = {}
            result_item["error"] = str(e)
            results.append(result_item)
            continue

    # 保存结果
    save_jsonl(results, output_path)
    print("=" * 80)
    print(f"✅ 检索完成！")
    print(f"成功处理 {len(results)} 个问题")
    print(f"结果已保存到: {output_path}")
    print("=" * 80)
    print(f"\n下一步运行:")
    print(f"python inference/step2_generate_answers.py \\")
    print(f"    --input_file {output_path} \\")
    print(f"    --llm_model_path <模型路径> \\")
    print(f"    --llm_tokenizer <tokenizer路径> \\")
    print(f"    ...")


if __name__ == "__main__":
    main()
