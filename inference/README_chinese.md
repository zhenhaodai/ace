# 中文 Claim 分解脚本使用说明

## 功能介绍

`decompose_vllm_chinese.py` 是一个用于分解中文 claim 陈述的脚本。它可以：

- 读取 xlsx 格式的文件
- 从第二列（列名为 "claim"）提取中文 claim 数据
- 使用 vLLM 模型将复杂的 claim 分解为多个简单的子陈述
- 输出多种格式的结果（JSONL、JSON、Excel）

## 依赖安装

```bash
pip install -r requirements.txt
```

## 输入文件格式

Excel 文件（.xlsx）应包含至少两列：

| 列1 (任意) | claim (第二列) | 列3 (可选) | ... |
|-----------|---------------|-----------|-----|
| 1 | 这是第一个需要分解的中文陈述 | 其他信息 | ... |
| 2 | 这是第二个需要分解的中文陈述 | 其他信息 | ... |

**注意：**
- 第二列的列名应该是 "claim"（如果不是，脚本会自动使用第二列）
- 其他列的信息会被保留在元数据中

## 使用方法

### 基本用法

```bash
python inference/decompose_vllm_chinese.py \
    --model_path /path/to/your/model \
    --input_file /path/to/your/claims.xlsx \
    --output_dir output \
    --expname my_experiment
```

### 完整参数说明

```bash
python inference/decompose_vllm_chinese.py \
    --tokenizer meta-llama/Llama-3.2-3B-Instruct \  # 分词器路径
    --model_path /path/to/model \                    # 模型路径（必需）
    --input_file claims.xlsx \                       # 输入 xlsx 文件（必需）
    --output_dir output \                            # 输出目录
    --expname chinese_claim_decompose \              # 实验名称
    --temperature 0.0 \                              # 生成温度（0.0 表示确定性生成）
    --top_p 0.99 \                                   # top_p 采样参数
    --tensor_parallel_size 1 \                       # 使用的 GPU 数量
    --batch_size 1                                   # 批处理大小
```

### 参数详解

- `--model_path`: （必需）vLLM 模型的路径
- `--input_file`: （必需）包含 claim 数据的 xlsx 文件路径
- `--tokenizer`: 分词器路径，默认使用 Llama-3.2-3B-Instruct
- `--output_dir`: 输出目录，默认为 "output"
- `--expname`: 实验名称，用于区分不同的实验
- `--temperature`: 生成温度
  - `0.0`: 确定性生成（推荐用于一致性结果）
  - `> 0.0`: 随机采样（会生成 3 个不同的分解版本）
- `--top_p`: nucleus 采样参数
- `--tensor_parallel_size`: 并行使用的 GPU 数量

## 输出格式

脚本会在输出目录中生成三种格式的文件：

### 1. JSONL 格式 (`decomposed_claims.jsonl`)

每行一个 JSON 对象：

```json
{"claim_id": 0, "claim": "原始陈述", "metadata": {...}, "decompose_id": 0, "decomposed": [...], "raw_output": "..."}
```

### 2. JSON 格式 (`decomposed_claims.json`)

格式化的 JSON 数组，便于阅读：

```json
[
  {
    "claim_id": 0,
    "claim": "原始陈述",
    "metadata": {
      "index": 0,
      "claim": "原始陈述",
      "其他列": "其他数据"
    },
    "decompose_id": 0,
    "decomposed": [
      {
        "label": "C1",
        "text": "子陈述1",
        "needs_context": true
      },
      {
        "label": "C2",
        "text": "子陈述2",
        "needs_context": true
      }
    ],
    "raw_output": "模型的原始输出"
  }
]
```

### 3. Excel 格式 (`decomposed_claims.xlsx`)

表格格式，便于查看和分析：

| 原始Claim | Claim ID | 分解ID | 子陈述序号 | 子陈述标签 | 子陈述内容 |
|----------|----------|--------|-----------|-----------|-----------|
| 原始陈述1 | 0 | 0 | 1 | C1 | 子陈述1 |
| 原始陈述1 | 0 | 0 | 2 | C2 | 子陈述2 |

## 示例

### 输入示例（claims.xlsx）

| ID | claim | source |
|----|-------|--------|
| 1 | 张三是一位著名的科学家，他在2020年获得了诺贝尔奖，并在北京大学任教 | Wikipedia |
| 2 | 这部电影由李四导演，于2021年上映，票房超过10亿元 | IMDB |

### 运行命令

```bash
python inference/decompose_vllm_chinese.py \
    --model_path /models/llama-3.2-3b-instruct \
    --input_file claims.xlsx \
    --output_dir results \
    --expname test_run \
    --temperature 0.0
```

### 输出示例

对于第一个 claim，可能的分解结果：

```
### C1: 张三是一位科学家
### C2: 张三是一位著名的科学家
### C3: 张三在2020年获得了诺贝尔奖
### C4: 张三在北京大学任教
```

## 注意事项

1. **模型选择**：建议使用支持中文的模型，如 Qwen、ChatGLM 或中文优化的 Llama 模型
2. **GPU 内存**：脚本默认使用 90% 的 GPU 内存，确保有足够的显存
3. **批处理**：目前批处理大小默认为 1，可根据 GPU 内存调整
4. **编码**：所有输出文件使用 UTF-8 编码，确保中文正确显示

## 故障排除

### 问题1：找不到 "claim" 列

**解决方法**：确保 xlsx 文件的第二列列名为 "claim"，或者脚本会自动使用第二列

### 问题2：GPU 内存不足

**解决方法**：修改脚本中的 `gpu_memory_utilization` 参数，或使用更小的模型

### 问题3：中文显示乱码

**解决方法**：确保使用 UTF-8 编码打开输出文件

## 后续处理

生成的分解结果可以用于：

- 事实验证（Fact Verification）
- 问答系统（QA）
- 信息检索（Information Retrieval）
- 知识图谱构建（Knowledge Graph Construction）

## 联系方式

如有问题，请提交 Issue 或联系开发团队。
