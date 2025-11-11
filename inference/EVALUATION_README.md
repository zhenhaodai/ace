# 评测脚本使用说明

本说明介绍如何使用评测脚本对生成的答案进行评测。

## 快速开始

### 方法 1: 使用便捷脚本

```bash
cd /home/user/ace
./inference/run_evaluation.sh <生成结果文件> <数据集名称>
```

**示例：**
```bash
# 评测 HotpotQA 数据集
./inference/run_evaluation.sh data/output_answers.jsonl hotpotqa

# 评测 HOVER 数据集
./inference/run_evaluation.sh data/hover_answers.jsonl hover
```

### 方法 2: 直接调用 Python 脚本

```bash
python3 inference/evaluate.py \
    --input_file <生成结果文件路径> \
    --dataset <数据集名称> \
    --output_file <输出文件路径(可选)> \
    --verbose  # 可选，打印详细对比信息
```

**示例：**
```bash
python3 inference/evaluate.py \
    --input_file data/retrieved_answers.jsonl \
    --dataset hotpotqa \
    --output_file results/metrics.json \
    --verbose
```

## 支持的数据集

- `hotpotqa` - HotpotQA 多跳问答数据集
- `hover` - HOVER 事实核查数据集（Yes/No）
- `exfever` - ExFEVER 扩展事实核查数据集（Yes/No）
- `bamboogle` - Bamboogle 问答数据集
- `musique` - MuSiQue 多跳推理数据集
- `2wikimqa` - 2WikiMQA 多跳问答数据集

## 输入文件格式要求

生成结果文件必须是 JSONL 格式，每行包含一个 JSON 对象，必需字段：

```json
{
  "question": "问题文本",
  "final_answer": "生成的答案（可以包含 <answer>标签</answer>）",
  "answer": "标准答案（或 answers, label 等字段）"
}
```

**示例：**
```json
{"question": "谁是美国总统？", "final_answer": "<answer>拜登</answer>", "answer": "拜登"}
```

## 评测指标

### 1. Exact Match (EM)
- 精确匹配率，预测答案与标准答案完全一致的比例
- 在比较前会进行标准化（小写化、去标点、去冠词等）

### 2. F1 Score
- Token 级别的 F1 分数
- 衡量预测答案和标准答案之间的词汇重叠度

### 3. Yes/No Accuracy（仅适用于 hover, exfever）
- 分类准确率，用于 Yes/No 类型的事实核查任务

## 输出示例

```
================================================================================
评测结果:
================================================================================
总样本数: 1000
有效评测样本数: 998

Exact Match (EM): 65.23%
F1 Score: 78.45%
Yes/No Accuracy: 82.16%
================================================================================
```

## 常见问题

### Q1: 如何处理中文数据？
评测脚本支持中文，会自动处理 `claim` 字段（兼容中文事实核查数据集）。

### Q2: 我的答案没有 `<answer>` 标签怎么办？
脚本会自动处理，如果没有标签，会使用整个 `final_answer` 字段的内容。

### Q3: 标准答案字段名不一致怎么办？
脚本会自动尝试多种常见字段名：`answer`, `answers`, `label`, `answers_objects` 等。

### Q4: 如何查看详细的对比信息？
使用 `--verbose` 参数，会打印前 10 个样本的详细对比信息。

## 自定义评测

如果需要自定义评测逻辑，可以修改 `inference/evaluate.py` 中的函数：

- `get_ground_truth_answer()` - 提取标准答案的逻辑
- `compute_*()` - 各种指标的计算方法
- `normalize_answer()` - 答案标准化方法

## 批量评测示例

```bash
# 评测多个文件
for file in data/*_answers.jsonl; do
    echo "评测 $file"
    python3 inference/evaluate.py \
        --input_file "$file" \
        --dataset hotpotqa \
        --output_file "${file%.jsonl}_metrics.json"
done
```
