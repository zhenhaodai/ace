# 中文 Claim 分解脚本使用说明

## 功能说明

`decompose_vllm_chinese.py` 是一个专门用于分解中文声明（claim）的脚本，支持：
- 读取 xlsx 格式的输入文件
- 自动处理第二列作为 claim 列
- 使用 vLLM 加速推理
- 输出 JSONL 和 XLSX 两种格式

## 安装依赖

```bash
pip install pandas openpyxl transformers vllm
```

## 输入文件格式

输入的 xlsx 文件应该包含一个名为 `claim` 的列（或者使用第二列作为 claim）。

示例：

| id | claim | label |
|----|-------|-------|
| 1  | 中国是世界上人口最多的国家，并且拥有悠久的历史文化 | SUPPORT |
| 2  | 人工智能技术在医疗领域有广泛应用 | SUPPORT |

## 使用方法

### 基本用法

```bash
python inference/decompose_vllm_chinese.py \
  --model_path /path/to/your/model \
  --input_file data/claims.xlsx \
  --output_dir output/decomposed_claims
```

### 完整参数示例

```bash
python inference/decompose_vllm_chinese.py \
  --tokenizer meta-llama/Llama-3.2-3B-Instruct \
  --model_path /path/to/your/chinese/llm \
  --expname chinese_claim_test \
  --temperature 0.0 \
  --top_p 0.99 \
  --tensor_parallel_size 1 \
  --input_file data/chinese_claims.xlsx \
  --output_dir output/decomposed_claims \
  --max_tokens 2048 \
  --batch_size 32
```

## 参数说明

- `--model_path`: vLLM 模型路径（必需）
- `--input_file`: 输入的 xlsx 文件路径（必需）
- `--tokenizer`: tokenizer 名称或路径（默认：meta-llama/Llama-3.2-3B-Instruct）
- `--expname`: 实验名称（默认：chinese_claim_decompose）
- `--temperature`: 采样温度，0.0 表示贪心解码（默认：0.0）
- `--top_p`: Top-p 采样参数（默认：0.99）
- `--tensor_parallel_size`: GPU 数量（默认：1）
- `--output_dir`: 输出目录（默认：output/decomposed_claims）
- `--max_tokens`: 最大生成 token 数（默认：2048）
- `--batch_size`: 批处理大小（默认：32）

## 输出格式

脚本会生成两个文件：

### 1. JSONL 格式 (`decomposed_claims_t{temperature}_{expname}.jsonl`)

每行包含：
```json
{
  "row_id": 0,
  "claim": "原始声明文本",
  "original_row": {...},
  "claim_id": 0,
  "decompose_id": 0,
  "decomposed": [
    {
      "label": "C1",
      "text": "子声明1"
    },
    {
      "label": "C2",
      "text": "子声明2"
    }
  ],
  "raw_generation": "模型原始输出"
}
```

### 2. XLSX 格式 (`decomposed_claims_t{temperature}_{expname}.xlsx`)

包含以下列：
- `claim_id`: 声明 ID
- `original_claim`: 原始声明
- `num_subclaims`: 子声明数量
- `decomposed_text`: 分解后的子声明（用 | 分隔）
- `raw_generation`: 模型原始输出

## 示例

### 输入
```
claim: "中国是世界上人口最多的国家，并且拥有悠久的历史文化"
```

### 输出
```
### 1: 中国是世界上人口最多的国家
### 2: 中国拥有悠久的历史文化
```

## 提示词模板

脚本使用以下中文提示词模板：

```
请将以下声明（claim）分解为多个更小的子声明，每个子声明关注原始声明的一个具体组成部分，以便模型更容易验证。
每个子声明以 ### 开头。如果需要，可以使用 #1、#2 等引用之前子声明的答案。

原始声明："{claim}"

分解后的子声明：
```

## 注意事项

1. 确保输入 xlsx 文件的第二列是 `claim` 列，或者列名为 `claim`
2. 空的 claim 行会被自动过滤
3. 使用支持中文的 LLM 模型以获得最佳效果
4. 根据 GPU 显存调整 `tensor_parallel_size` 参数
5. 输出文件使用 UTF-8 编码，确保中文正确显示

## 故障排除

### 问题：找不到 'claim' 列
解决方案：确保 xlsx 文件的第二列是 claim 数据，或者将列名改为 'claim'

### 问题：GPU 内存不足
解决方案：
- 减小 `batch_size`
- 增加 `tensor_parallel_size` 使用多个 GPU
- 调整 vLLM 的 `gpu_memory_utilization` 参数

### 问题：生成结果为空
解决方案：
- 检查模型是否支持中文
- 调整 `temperature` 和 `top_p` 参数
- 查看控制台输出的示例结果
