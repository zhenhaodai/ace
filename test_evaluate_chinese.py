#!/usr/bin/env python3
"""测试评测脚本对中文字段的支持"""
import json
import sys
sys.path.insert(0, '/home/user/ace/inference')
from evaluate import get_ground_truth_answer

# 测试用例1：original_row 中包含中文字段
test_case_1 = {
    "row_id": 0,
    "claim": "测试内容",
    "original_row": {
        "query": 1.0,
        "人工评测结果": "F",
        "10-02版本结果": "F"
    },
    "final_answer": "这是生成的答案"
}

# 测试用例2：根节点包含中文字段
test_case_2 = {
    "row_id": 1,
    "人工评测结果": "T",
    "final_answer": "另一个答案"
}

# 测试用例3：标准英文字段
test_case_3 = {
    "row_id": 2,
    "answer": "Standard English Answer",
    "final_answer": "Generated answer"
}

# 测试用例4：没有标准答案
test_case_4 = {
    "row_id": 3,
    "final_answer": "Some answer"
}

print("=" * 80)
print("测试评测脚本对中文字段的支持")
print("=" * 80)

for i, test_case in enumerate([test_case_1, test_case_2, test_case_3, test_case_4], 1):
    print(f"\n测试用例 {i}:")
    print(f"输入: {json.dumps(test_case, ensure_ascii=False, indent=2)}")

    ground_truth = get_ground_truth_answer(test_case, "hotpotqa")

    if ground_truth:
        print(f"✓ 成功提取标准答案: '{ground_truth}'")
    else:
        print(f"✗ 未找到标准答案")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
