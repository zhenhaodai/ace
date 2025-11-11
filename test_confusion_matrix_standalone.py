#!/usr/bin/env python3
"""测试混淆矩阵功能（独立版本）"""
import json
from typing import List, Dict, Any

# 复制必要的函数
def normalize_verdict(s: str) -> str:
    """标准化事实核查的判定结果"""
    s = s.strip().lower()

    if s in ['t', 'true', '成立', '正确', '支持', 'yes']:
        return 'T'
    if s in ['f', 'false', '不成立', '错误', '不支持', 'no']:
        return 'F'
    if s in ['uncertain', 'u', '不确定', '证据不足', '无法判断', '未知']:
        return 'uncertain'

    s_clean = s.replace(' ', '').replace('\n', '').replace('\t', '')

    if any(word in s_clean for word in ['不成立', '不支持', '不正确', '错误', 'false', 'notrue']):
        return 'F'
    if any(word in s_clean for word in ['不确定', '证据不足', '无法判断', 'uncertain']):
        return 'uncertain'
    if any(word in s_clean for word in ['成立', '正确', '支持', 'true']):
        return 'T'

    return s


def get_ground_truth_answer(item: Dict[str, Any]) -> str:
    """提取标准答案"""
    if "original_row" in item:
        original = item["original_row"]
        for field in ["人工评测结果", "标准答案", "答案"]:
            if field in original and original[field]:
                return str(original[field])
    return ""


def compute_confusion_matrix(results: List[Dict[str, Any]]) -> Dict:
    """计算混淆矩阵"""
    confusion_matrix = {
        'T': {'T': 0, 'F': 0, 'uncertain': 0},
        'F': {'T': 0, 'F': 0, 'uncertain': 0},
        'uncertain': {'T': 0, 'F': 0, 'uncertain': 0}
    }

    for item in results:
        predicted_answer = item.get("final_answer", "")
        ground_truth = get_ground_truth_answer(item)

        if not ground_truth:
            continue

        pred_normalized = normalize_verdict(predicted_answer)
        truth_normalized = normalize_verdict(ground_truth)

        if pred_normalized in confusion_matrix and truth_normalized in ['T', 'F', 'uncertain']:
            confusion_matrix[pred_normalized][truth_normalized] += 1

    return confusion_matrix


def print_confusion_matrix(cm: Dict):
    """打印混淆矩阵"""
    print("\n" + "=" * 80)
    print("混淆矩阵 (Confusion Matrix):")
    print("=" * 80)

    # 表头
    header = "预测 \\ 真实"
    print(f"{header:<15} {'T':>10} {'F':>10} {'uncertain':>10} {'总计':>10}")
    print("-" * 57)

    # 每一行
    for pred_label in ['T', 'F', 'uncertain']:
        row_sum = sum(cm[pred_label].values())
        print(f"{pred_label:<15} {cm[pred_label]['T']:>10} {cm[pred_label]['F']:>10} {cm[pred_label]['uncertain']:>10} {row_sum:>10}")

    # 列总计
    col_t = sum(cm[pred]['T'] for pred in ['T', 'F', 'uncertain'])
    col_f = sum(cm[pred]['F'] for pred in ['T', 'F', 'uncertain'])
    col_u = sum(cm[pred]['uncertain'] for pred in ['T', 'F', 'uncertain'])
    total_count = col_t + col_f + col_u

    print("-" * 57)
    print(f"{'总计':<15} {col_t:>10} {col_f:>10} {col_u:>10} {total_count:>10}")

    # 各类别指标
    print("\n" + "=" * 80)
    print("各类别指标:")
    print("=" * 80)
    print(f"{'类别':<12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 60)

    for label in ['T', 'F', 'uncertain']:
        tp = cm[label][label]
        fp = sum(cm[label][other] for other in ['T', 'F', 'uncertain'] if other != label)
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0

        fn = sum(cm[other][label] for other in ['T', 'F', 'uncertain'] if other != label)
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        support = sum(cm[pred][label] for pred in ['T', 'F', 'uncertain'])

        print(f"{label:<12} {precision:>11.2f}% {recall:>11.2f}% {f1:>11.2f}% {support:>12}")

    print("=" * 80)

    # 计算总体准确率
    correct = sum(cm[label][label] for label in ['T', 'F', 'uncertain'])
    accuracy = correct / total_count * 100 if total_count > 0 else 0.0
    print(f"\n总体准确率 (Accuracy): {accuracy:.2f}%")


# 创建测试数据
test_results = [
    # T 类别：3个样本（2对1错）
    {"claim": "测试1", "final_answer": "成立", "original_row": {"人工评测结果": "T"}},
    {"claim": "测试2", "final_answer": "该主张成立", "original_row": {"人工评测结果": "T"}},
    {"claim": "测试3", "final_answer": "不成立", "original_row": {"人工评测结果": "T"}},

    # F 类别：4个样本（2对2错）
    {"claim": "测试4", "final_answer": "不成立", "original_row": {"人工评测结果": "F"}},
    {"claim": "测试5", "final_answer": "该主张不成立", "original_row": {"人工评测结果": "F"}},
    {"claim": "测试6", "final_answer": "成立", "original_row": {"人工评测结果": "F"}},
    {"claim": "测试7", "final_answer": "不确定", "original_row": {"人工评测结果": "F"}},

    # uncertain 类别：3个样本（2对1错）
    {"claim": "测试8", "final_answer": "证据不足", "original_row": {"人工评测结果": "uncertain"}},
    {"claim": "测试9", "final_answer": "不确定", "original_row": {"人工评测结果": "uncertain"}},
    {"claim": "测试10", "final_answer": "成立", "original_row": {"人工评测结果": "uncertain"}},
]

print("=" * 80)
print("测试混淆矩阵功能")
print("=" * 80)
print(f"\n创建了 {len(test_results)} 个测试样本")

# 显示每个样本的详情
print("\n样本详情:")
for i, item in enumerate(test_results):
    pred = normalize_verdict(item["final_answer"])
    truth = normalize_verdict(get_ground_truth_answer(item))
    match = "✓" if pred == truth else "✗"
    print(f"  {i+1}. {match} 预测={pred}, 真实={truth} - {item['final_answer'][:20]}")

# 计算并显示混淆矩阵
cm = compute_confusion_matrix(test_results)
print_confusion_matrix(cm)
