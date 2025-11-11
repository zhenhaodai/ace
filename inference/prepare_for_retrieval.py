"""
转换中文分解结果为 step1_retrieval.py 需要的格式
从 decomposed 的 text 字段中提取 label（如 #1, #2 等）
"""
import argparse
import json
import re
from typing import Dict, List
from utils import load_jsonl, save_jsonl


def extract_label_from_text(text: str) -> str:
    """从文本中提取标签，例如从 '#1 新冠病毒...' 提取 '#1'"""
    # 匹配开头的 #数字 格式
    match = re.match(r'^(#\d+)', text)
    if match:
        return match.group(1)
    # 如果没有匹配到，返回空字符串
    return ""


def add_labels_to_decomposed(item: Dict) -> Dict:
    """为 decomposed 字段中的每个子问题添加 label"""
    if "decomposed" not in item:
        return item

    decomposed_with_labels = []
    for idx, subq in enumerate(item["decomposed"]):
        # 复制原有字段
        new_subq = {"text": subq["text"]}

        # 尝试从 text 中提取 label
        label = extract_label_from_text(subq["text"])

        # 如果没有提取到 label，使用索引生成
        if not label:
            label = f"#{idx + 1}"

        new_subq["label"] = label

        # 保留其他字段（如 needs_context）
        for key in subq:
            if key not in ["text", "label"]:
                new_subq[key] = subq[key]

        decomposed_with_labels.append(new_subq)

    # 更新 item
    result = item.copy()
    result["decomposed"] = decomposed_with_labels
    return result


def main():
    parser = argparse.ArgumentParser(description="为中文分解结果添加 label 字段以适配 step1_retrieval.py")
    parser.add_argument("--input", type=str, required=True, help="输入 jsonl 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 jsonl 文件路径")
    args = parser.parse_args()

    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print("=" * 80)

    # 读取数据
    data = load_jsonl(args.input)
    print(f"已加载 {len(data)} 条数据")

    # 处理数据
    processed_data = []
    for item in data:
        processed_item = add_labels_to_decomposed(item)
        processed_data.append(processed_item)

    # 保存结果
    save_jsonl(processed_data, args.output)

    print("=" * 80)
    print(f"处理完成，共 {len(processed_data)} 条数据")
    print(f"已保存到: {args.output}")

    # 显示示例
    if processed_data:
        print("\n示例（第一条数据的前3个子问题）:")
        print("-" * 80)
        first_item = processed_data[0]
        print(f"问题: {first_item['question'][:50]}...")
        if "decomposed" in first_item and first_item["decomposed"]:
            for subq in first_item["decomposed"][:3]:
                print(f"  {subq['label']}: {subq['text'][:60]}...")
        print("-" * 80)

    print("=" * 80)


if __name__ == "__main__":
    main()
