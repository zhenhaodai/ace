# 评测脚本中文字段支持说明

## 问题描述

原评测脚本 `inference/evaluate.py` 只支持英文字段名（如 `answer`、`answers`、`label` 等），无法识别中文数据集中的标准答案字段，导致提示"缺少标准答案"。

## 解决方案

修改了 `get_ground_truth_answer` 函数，添加对中文字段的支持。

### 支持的中文字段名（按优先级）

1. **`original_row` 中的字段**:
   - `人工评测结果`
   - `标准答案`
   - `答案`
   - `label`

2. **根节点的字段**:
   - `人工评测结果`
   - `标准答案`
   - `答案`

3. **英文字段**（保持向后兼容）:
   - `answer`
   - `answers`
   - `answers_objects`
   - `label`

## 数据格式示例

### 中文数据集格式

```json
{
  "row_id": 0,
  "claim": "网页标题：pura80和mate70谁更优...",
  "original_row": {
    "query": 1.0,
    "claim": "网页内容...",
    "人工评测结果": "F",
    "10-02版本结果": "F"
  },
  "decomposed": [...],
  "retrieved_passages": {...},
  "final_answer": "生成的答案内容"
}
```

### 英文数据集格式（兼容）

```json
{
  "question": "What is the capital of France?",
  "answer": "Paris",
  "final_answer": "The capital of France is Paris."
}
```

## 使用方法

### 1. 运行评测

```bash
python inference/evaluate.py \
  --input_file your_results.jsonl \
  --dataset hotpotqa \
  --verbose
```

### 2. 参数说明

- `--input_file`: 包含生成结果和标准答案的 JSONL 文件
- `--dataset`: 数据集类型（hotpotqa、hover、exfever 等）
- `--verbose`: 显示详细对比信息
- `--output_file`: （可选）保存评测结果到文件

### 3. 输出示例

```
================================================================================
评测配置:
  输入文件: results.jsonl
  数据集: hotpotqa
================================================================================

正在加载生成结果...
加载了 5 条结果

开始评测...

================================================================================
评测结果:
================================================================================
总样本数: 5
有效评测样本数: 5

Exact Match (EM): 60.00%
F1 Score: 75.50%
================================================================================
```

## 测试验证

运行测试脚本验证功能：

```bash
python test_ground_truth.py
```

预期输出：
```
测试: original_row中的人工评测结果
✓ 通过 - 提取到: 'F'

测试: 根节点的人工评测结果
✓ 通过 - 提取到: 'T'

测试结果: 4 通过, 0 失败
```

## 注意事项

1. **标准答案值**：
   - 对于事实核查任务，标准答案可能是 "T"（True）、"F"（False）或 "U"（Uncertain）
   - 评测时会将生成的答案与标准答案进行归一化比较

2. **NaN 值处理**：
   - 如果字段值为 NaN（常见于 pandas DataFrame 转换的数据），会自动跳过

3. **优先级**：
   - 优先从 `original_row` 中提取，确保使用原始数据集的标准答案
   - 如果 `original_row` 中没有，再查找根节点字段

## 相关文件

- `inference/evaluate.py` - 评测主脚本（已修改）
- `test_ground_truth.py` - 测试脚本
- `inference/step2_generation.py` - 生成答案脚本（会保留 `original_row`）

## 修改日期

2025-11-11
