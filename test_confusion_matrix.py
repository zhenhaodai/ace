#!/usr/bin/env python3
"""测试混淆矩阵功能"""
import sys
import json
sys.path.insert(0, '/home/user/ace/inference')

# 创建测试数据
test_results = [
    # T 类别测试
    {"claim": "测试1", "final_answer": "成立", "original_row": {"人工评测结果": "T"}},
    {"claim": "测试2", "final_answer": "该主张成立", "original_row": {"人工评测结果": "T"}},
    {"claim": "测试3", "final_answer": "不成立", "original_row": {"人工评测结果": "T"}},  # 错误预测

    # F 类别测试
    {"claim": "测试4", "final_answer": "不成立", "original_row": {"人工评测结果": "F"}},
    {"claim": "测试5", "final_answer": "该主张不成立", "original_row": {"人工评测结果": "F"}},
    {"claim": "测试6", "final_answer": "成立", "original_row": {"人工评测结果": "F"}},  # 错误预测
    {"claim": "测试7", "final_answer": "不确定", "original_row": {"人工评测结果": "F"}},  # 错误预测

    # uncertain 类别测试
    {"claim": "测试8", "final_answer": "证据不足", "original_row": {"人工评测结果": "uncertain"}},
    {"claim": "测试9", "final_answer": "不确定", "original_row": {"人工评测结果": "uncertain"}},
    {"claim": "测试10", "final_answer": "成立", "original_row": {"人工评测结果": "uncertain"}},  # 错误预测
]

# 保存为临时文件
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
    for item in test_results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
    temp_file = f.name

print("=" * 80)
print("测试混淆矩阵功能")
print("=" * 80)
print(f"\n创建了 {len(test_results)} 个测试样本")
print("\n预期混淆矩阵:")
print("预测 \\ 真实        T         F  uncertain      总计")
print("-" * 57)
print("T                  2         1         1         4")
print("F                  1         2         0         3")
print("uncertain          0         1         2         3")
print("-" * 57)
print("总计               3         4         3        10")

print("\n" + "=" * 80)
print("运行评测...")
print("=" * 80)

# 运行评测
import subprocess
result = subprocess.run(
    ['python', 'inference/evaluate.py',
     '--input_file', temp_file,
     '--dataset', 'factcheck'],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("错误信息:", result.stderr)

# 清理临时文件
import os
os.unlink(temp_file)
