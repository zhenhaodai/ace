#!/usr/bin/env python3
"""
评测脚本：比较生成的答案与标准答案
支持多种数据集格式和评测指标
"""

import argparse
import json
import re
import string
from collections import Counter
from typing import List, Dict, Any, Tuple
from utils import load_jsonl


def normalize_answer(s: str) -> str:
    """标准化答案：小写化、去标点、去冠词、去多余空格"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_answer_from_tags(text: str) -> str:
    """从 <answer></answer> 标签中提取答案"""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # 如果没有标签，返回整个文本
    return text.strip()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配（Exact Match）"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算 F1 分数"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # 如果都为空，返回 1.0
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0

    # 如果其中一个为空，返回 0.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0

    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common_tokens.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_yes_no_accuracy(prediction: str, ground_truth: str) -> float:
    """计算 Yes/No 准确率（用于 hover, exfever 等数据集）"""
    pred_normalized = normalize_answer(prediction)
    truth_normalized = normalize_answer(ground_truth)

    # 提取 Yes/No
    pred_answer = None
    if 'yes' in pred_normalized:
        pred_answer = 'yes'
    elif 'no' in pred_normalized:
        pred_answer = 'no'

    truth_answer = None
    if 'yes' in truth_normalized:
        truth_answer = 'yes'
    elif 'no' in truth_normalized:
        truth_answer = 'no'

    if pred_answer is None or truth_answer is None:
        return 0.0

    return float(pred_answer == truth_answer)


def get_ground_truth_answer(item: Dict[str, Any], dataset: str) -> str:
    """根据数据集类型提取标准答案"""
    # 尝试多种可能的字段名
    if "answer" in item:
        return str(item["answer"])

    if "answers" in item:
        answers = item["answers"]
        if isinstance(answers, list) and len(answers) > 0:
            return str(answers[0])
        return str(answers)

    if "answers_objects" in item:
        # Bamboogle 等数据集格式
        answers_obj = item["answers_objects"]
        if isinstance(answers_obj, list) and len(answers_obj) > 0:
            spans = answers_obj[0].get("spans", [])
            if spans and len(spans) > 0:
                return str(spans[0])

    if "label" in item:
        # hover, exfever 等数据集
        label = item["label"]
        # 将数字标签转换为 Yes/No
        if isinstance(label, int):
            return "Yes" if label == 1 else "No"
        return str(label)

    # 兼容中文数据
    if "claim" in item:
        # 如果有 verdict 字段
        if "verdict" in item:
            return str(item["verdict"])

    return ""


def evaluate_predictions(
    results: List[Dict[str, Any]],
    dataset: str,
    verbose: bool = False
) -> Dict[str, float]:
    """评测预测结果"""

    total = len(results)
    em_scores = []
    f1_scores = []
    yes_no_correct = 0
    has_answer_count = 0

    for idx, item in enumerate(results):
        # 获取生成的答案
        predicted_answer = item.get("final_answer", "")
        predicted_answer = extract_answer_from_tags(predicted_answer)

        # 获取标准答案
        ground_truth = get_ground_truth_answer(item, dataset)

        if not ground_truth:
            if verbose:
                print(f"[Warning] 样本 {idx} 缺少标准答案")
            continue

        has_answer_count += 1

        # 计算指标
        em = compute_exact_match(predicted_answer, ground_truth)
        f1 = compute_f1(predicted_answer, ground_truth)
        em_scores.append(em)
        f1_scores.append(f1)

        # 对于分类任务，额外计算准确率
        if dataset in ["hover", "exfever"]:
            acc = compute_yes_no_accuracy(predicted_answer, ground_truth)
            if acc == 1.0:
                yes_no_correct += 1

        if verbose and idx < 10:  # 打印前10个样本的对比
            print(f"\n样本 {idx}:")
            print(f"  问题: {item.get('question') or item.get('claim', '')[:100]}")
            print(f"  预测: {predicted_answer[:100]}")
            print(f"  标准: {ground_truth[:100]}")
            print(f"  EM: {em:.2f}, F1: {f1:.2f}")

    # 计算平均指标
    metrics = {}
    if has_answer_count > 0:
        metrics["exact_match"] = sum(em_scores) / len(em_scores) * 100
        metrics["f1"] = sum(f1_scores) / len(f1_scores) * 100
        metrics["evaluated_samples"] = has_answer_count

        if dataset in ["hover", "exfever"]:
            metrics["yes_no_accuracy"] = yes_no_correct / has_answer_count * 100

    metrics["total_samples"] = total

    return metrics


def main():
    parser = argparse.ArgumentParser(description="评测生成答案与标准答案的匹配度")
    parser.add_argument("--input_file", type=str, required=True,
                        help="生成结果文件路径（包含 final_answer 字段的 JSONL）")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                        choices=["hotpotqa", "hover", "exfever", "bamboogle", "musique", "2wikimqa"],
                        help="数据集类型")
    parser.add_argument("--verbose", action="store_true",
                        help="是否打印详细对比信息")
    parser.add_argument("--output_file", type=str, default=None,
                        help="保存评测结果的文件路径（可选）")

    args = parser.parse_args()

    print("=" * 80)
    print(f"评测配置:")
    print(f"  输入文件: {args.input_file}")
    print(f"  数据集: {args.dataset}")
    print("=" * 80)

    # 加载结果
    print("\n正在加载生成结果...")
    results = load_jsonl(args.input_file)
    print(f"加载了 {len(results)} 条结果\n")

    # 评测
    print("开始评测...")
    metrics = evaluate_predictions(results, args.dataset, args.verbose)

    # 打印结果
    print("\n" + "=" * 80)
    print("评测结果:")
    print("=" * 80)
    print(f"总样本数: {metrics['total_samples']}")
    print(f"有效评测样本数: {metrics.get('evaluated_samples', 0)}")

    if "exact_match" in metrics:
        print(f"\nExact Match (EM): {metrics['exact_match']:.2f}%")
        print(f"F1 Score: {metrics['f1']:.2f}%")

    if "yes_no_accuracy" in metrics:
        print(f"Yes/No Accuracy: {metrics['yes_no_accuracy']:.2f}%")

    print("=" * 80)

    # 保存结果到文件（可选）
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n评测结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
