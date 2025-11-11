# Excel 结果处理工具使用说明

本目录包含两个用于处理模型预测结果的Python脚本：

## 1. write_predictions_to_excel.py

### 功能
将模型预测结果写入到现有Excel文件的指定列（默认R列）

### 依赖安装
```bash
pip install openpyxl
```

### 使用方法

#### 基础用法（写入R列）
```bash
python write_predictions_to_excel.py \
    --excel /path/to/0918_data.xlsx \
    --predictions /path/to/results.jsonl
```

#### 指定列（例如S列）
```bash
python write_predictions_to_excel.py \
    --excel /path/to/0918_data.xlsx \
    --predictions /path/to/results.jsonl \
    --column S
```

#### 指定工作表
```bash
python write_predictions_to_excel.py \
    --excel /path/to/0918_data.xlsx \
    --predictions /path/to/results.jsonl \
    --sheet "Sheet1"
```

#### 自定义起始行
```bash
python write_predictions_to_excel.py \
    --excel /path/to/0918_data.xlsx \
    --predictions /path/to/results.jsonl \
    --header-row 1 \
    --start-row 2
```

### 参数说明
- `--excel`: Excel文件路径（必需）
- `--predictions`: 预测结果JSONL文件路径（必需）
- `--column`: 目标列，默认为R（可选）
- `--sheet`: 工作表名称，默认使用活动工作表（可选）
- `--header-row`: 表头所在行号，默认为1（可选）
- `--start-row`: 数据开始行号，默认为2（可选）

### 输出
- 生成带有后缀 `_with_predictions.xlsx` 的新文件
- 原文件保持不变

---

## 2. generate_comparison_table.py

### 功能
生成包含原始claim、人工评测结果和模型预测结果的对比表格，并自动标注一致性

### 依赖安装
```bash
pip install openpyxl pandas
```

### 使用方法

#### 方式1：仅从JSONL生成（推荐）
如果JSONL文件中已经包含了claim和人工评测结果：

```bash
python generate_comparison_table.py \
    --predictions /path/to/results.jsonl \
    --output comparison.xlsx
```

#### 方式2：从Excel和JSONL生成
如果人工评测结果在Excel中：

```bash
python generate_comparison_table.py \
    --excel /path/to/0918_data.xlsx \
    --predictions /path/to/results.jsonl \
    --output comparison.xlsx
```

#### 指定列名（从Excel读取时）
```bash
python generate_comparison_table.py \
    --excel /path/to/0918_data.xlsx \
    --predictions /path/to/results.jsonl \
    --human-column "人工评测结果" \
    --claim-column "claim" \
    --output comparison.xlsx
```

#### 生成CSV格式
```bash
python generate_comparison_table.py \
    --predictions /path/to/results.jsonl \
    --output comparison.csv \
    --format csv
```

### 参数说明
- `--predictions`: 预测结果JSONL文件路径（必需）
- `--output`: 输出文件路径（必需）
- `--excel`: 原始Excel文件路径（可选）
- `--format`: 输出格式，xlsx或csv，默认xlsx（可选）
- `--sheet`: Excel工作表名称（可选）
- `--human-column`: 人工评测结果所在列名（可选，自动检测）
- `--claim-column`: Claim所在列名（可选，自动检测）

### 输出特性
- **颜色标记**：
  - 绿色：模型预测与人工评测一致 ✓
  - 红色：模型预测与人工评测不一致 ✗
  - 白色：无法比较（缺少数据）

- **包含列**：
  1. 序号
  2. Claim/问题
  3. 人工评测结果（原始）
  4. 模型预测结果（原始）
  5. 标准化-人工（T/F/uncertain）
  6. 标准化-模型（T/F/uncertain）
  7. 是否一致（✓/✗）

- **统计信息**：
  - 总样本数
  - 一致数量及百分比
  - 不一致数量及百分比

---

## JSONL 文件格式要求

脚本期望JSONL文件包含以下字段：

```json
{
  "row_id": 0,
  "claim": "某个需要验证的声明",
  "final_answer": "模型的预测结果",
  "original_row": {
    "人工评测结果": "T",
    "claim": "某个需要验证的声明"
  }
}
```

或者：

```json
{
  "row_id": 0,
  "claim": "某个需要验证的声明",
  "final_answer": "<answer>F</answer>",
  "人工评测结果": "F"
}
```

### 支持的字段名（自动识别）：
- **人工评测结果**: `人工评测结果`, `标准答案`, `答案`, `answer`, `label`
- **问题/声明**: `claim`, `question`
- **模型预测**: `final_answer`

---

## 判定结果标准化

脚本会自动将中英文判定结果标准化为统一格式：

| 原始值 | 标准化结果 |
|--------|-----------|
| T, True, 成立, 正确, 支持, yes | T |
| F, False, 不成立, 错误, 不支持, no | F |
| uncertain, U, 不确定, 证据不足, 无法判断 | uncertain |

---

## 完整示例

### 示例1：写入R列
```bash
python write_predictions_to_excel.py \
    --excel /mnt/disk0/s84396777/AceSearcher/data/0918_data.xlsx \
    --predictions ./output/predictions_results.jsonl
```

输出: `0918_data_with_predictions.xlsx`

### 示例2：生成对比表格（从JSONL）
```bash
python generate_comparison_table.py \
    --predictions ./output/predictions_results.jsonl \
    --output ./output/comparison_table.xlsx
```

输出示例：
```
================================================================================
统计信息:
================================================================================
总样本数: 100
一致数量: 85 (85.00%)
不一致数量: 15 (15.00%)
================================================================================
```

### 示例3：生成对比表格（从Excel + JSONL）
```bash
python generate_comparison_table.py \
    --excel /mnt/disk0/s84396777/AceSearcher/data/0918_data.xlsx \
    --predictions ./output/predictions_results.jsonl \
    --output ./output/comparison_table.xlsx
```

---

## 常见问题

### Q1: 如何找到正确的Excel路径？
```bash
find /mnt -name "0918_data.xlsx" 2>/dev/null
```

### Q2: 如何查看Excel中的列名？
```bash
python -c "import pandas as pd; df = pd.read_excel('file.xlsx'); print(df.columns.tolist())"
```

### Q3: JSONL文件格式不对怎么办？
确保每行都是有效的JSON，可以使用以下命令检查：
```bash
head -5 results.jsonl | python -m json.tool
```

### Q4: 生成的Excel文件很大怎么办？
使用CSV格式：
```bash
python generate_comparison_table.py \
    --predictions results.jsonl \
    --output comparison.csv \
    --format csv
```

---

## 技术支持

如有问题，请检查：
1. Python版本 >= 3.7
2. 依赖包已正确安装：`pip install openpyxl pandas`
3. 文件路径正确且有访问权限
4. JSONL格式正确
