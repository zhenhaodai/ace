#!/usr/bin/env python3
"""测试判定结果标准化功能"""
import sys
sys.path.insert(0, '/home/user/ace/inference')

# 复制函数以避免导入依赖问题
def normalize_verdict(s: str) -> str:
    """
    标准化事实核查的判定结果（支持中英文）

    映射规则：
    - T/True/成立/正确/支持 → T
    - F/False/不成立/错误/不支持 → F
    - uncertain/不确定/证据不足/无法判断 → uncertain
    """
    s = s.strip().lower()

    # True/成立 相关
    if s in ['t', 'true', '成立', '正确', '支持', 'yes']:
        return 'T'

    # False/不成立 相关
    if s in ['f', 'false', '不成立', '错误', '不支持', 'no']:
        return 'F'

    # Uncertain/不确定 相关
    if s in ['uncertain', 'u', '不确定', '证据不足', '无法判断', '未知']:
        return 'uncertain'

    # 如果包含关键词（注意：要先检查否定形式，避免误匹配）
    s_clean = s.replace(' ', '').replace('\n', '').replace('\t', '')

    # 先检查否定形式和不确定形式
    if any(word in s_clean for word in ['不成立', '不支持', '不正确', '错误', 'false', 'notrue']):
        return 'F'
    if any(word in s_clean for word in ['不确定', '证据不足', '无法判断', 'uncertain']):
        return 'uncertain'
    # 再检查肯定形式
    if any(word in s_clean for word in ['成立', '正确', '支持', 'true']):
        return 'T'

    # 如果无法识别，返回原始小写文本
    return s


# 测试用例
test_cases = [
    # True 类
    ("T", "T"),
    ("t", "T"),
    ("True", "T"),
    ("true", "T"),
    ("成立", "T"),
    ("正确", "T"),
    ("支持", "T"),
    ("Yes", "T"),
    ("该主张成立", "T"),

    # False 类
    ("F", "F"),
    ("f", "F"),
    ("False", "F"),
    ("false", "F"),
    ("不成立", "F"),
    ("错误", "F"),
    ("不支持", "F"),
    ("No", "F"),
    ("该主张不成立", "F"),

    # Uncertain 类
    ("uncertain", "uncertain"),
    ("Uncertain", "uncertain"),
    ("U", "uncertain"),
    ("不确定", "uncertain"),
    ("证据不足", "uncertain"),
    ("无法判断", "uncertain"),
    ("未知", "uncertain"),
]

print("=" * 80)
print("测试判定结果标准化功能")
print("=" * 80)

passed = 0
failed = 0

for input_text, expected in test_cases:
    result = normalize_verdict(input_text)

    status = "✓" if result == expected else "✗"
    if result == expected:
        passed += 1
    else:
        failed += 1

    print(f"{status} '{input_text}' → '{result}' (期望: '{expected}')")

print("\n" + "=" * 80)
print(f"测试结果: {passed} 通过, {failed} 失败")
print("=" * 80)

# 测试实际评测场景
print("\n" + "=" * 80)
print("实际场景测试")
print("=" * 80)

scenarios = [
    {"pred": "成立", "truth": "T", "should_match": True},
    {"pred": "不成立", "truth": "F", "should_match": True},
    {"pred": "证据不足", "truth": "uncertain", "should_match": True},
    {"pred": "成立", "truth": "F", "should_match": False},
    {"pred": "不成立", "truth": "T", "should_match": False},
]

for scenario in scenarios:
    pred = scenario["pred"]
    truth = scenario["truth"]
    should_match = scenario["should_match"]

    pred_norm = normalize_verdict(pred)
    truth_norm = normalize_verdict(truth)
    matches = (pred_norm == truth_norm)

    status = "✓" if matches == should_match else "✗"
    print(f"{status} 预测='{pred}' ({pred_norm}) vs 标准='{truth}' ({truth_norm}) → 匹配={matches} (期望={should_match})")

print("=" * 80)
