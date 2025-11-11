#!/usr/bin/env python3
"""测试标准答案提取函数（独立版本）"""
import json
from typing import Dict, Any

def get_ground_truth_answer(item: Dict[str, Any], dataset: str) -> str:
    """根据数据集类型提取标准答案（复制的函数用于测试）"""
    # 中文数据集字段（优先检查）
    # 检查 original_row 中的中文字段
    if "original_row" in item:
        original = item["original_row"]
        # 尝试各种可能的中文字段名
        for field in ["人工评测结果", "标准答案", "答案", "label"]:
            if field in original and original[field]:
                answer = original[field]
                # 跳过 NaN 值
                if answer != answer:  # NaN check
                    continue
                return str(answer)

    # 直接检查根节点的中文字段
    for field in ["人工评测结果", "标准答案", "答案"]:
        if field in item and item[field]:
            answer = item[field]
            if answer != answer:  # NaN check
                continue
            return str(answer)

    # 尝试多种可能的英文字段名
    if "answer" in item:
        return str(item["answer"])

    if "answers" in item:
        answers = item["answers"]
        if isinstance(answers, list) and len(answers) > 0:
            return str(answers[0])
        return str(answers)

    if "label" in item:
        label = item["label"]
        if isinstance(label, int):
            return "Yes" if label == 1 else "No"
        return str(label)

    return ""

# 测试用例
test_cases = [
    {
        "name": "original_row中的人工评测结果",
        "data": {
            "row_id": 0,
            "claim": "测试内容",
            "original_row": {
                "query": 1.0,
                "人工评测结果": "F",
                "10-02版本结果": "F"
            },
            "final_answer": "这是生成的答案"
        },
        "expected": "F"
    },
    {
        "name": "根节点的人工评测结果",
        "data": {
            "row_id": 1,
            "人工评测结果": "T",
            "final_answer": "另一个答案"
        },
        "expected": "T"
    },
    {
        "name": "标准英文字段",
        "data": {
            "row_id": 2,
            "answer": "Standard English Answer",
            "final_answer": "Generated answer"
        },
        "expected": "Standard English Answer"
    },
    {
        "name": "没有标准答案",
        "data": {
            "row_id": 3,
            "final_answer": "Some answer"
        },
        "expected": ""
    }
]

print("=" * 80)
print("测试标准答案提取函数")
print("=" * 80)

passed = 0
failed = 0

for test in test_cases:
    print(f"\n测试: {test['name']}")
    result = get_ground_truth_answer(test['data'], "hotpotqa")
    expected = test['expected']

    if result == expected:
        print(f"✓ 通过 - 提取到: '{result}'")
        passed += 1
    else:
        print(f"✗ 失败 - 期望: '{expected}', 实际: '{result}'")
        failed += 1

print("\n" + "=" * 80)
print(f"测试结果: {passed} 通过, {failed} 失败")
print("=" * 80)
