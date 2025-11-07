# 搜索与问答流程说明

本目录包含两步式的问答流程，将搜索和LLM生成完全分离：

## 流程概览

```
原始问题（分解后）
    ↓
【步骤1】调用搜索API检索文档
    ↓
检索结果文件（.jsonl）
    ↓
【步骤2】使用LLM生成答案
    ↓
最终答案文件（.jsonl）
```

## 文件说明

- `search_service.py` - 搜索API服务模块（底层封装）
- `step1_search_retrieval.py` - **步骤1**: 搜索检索脚本
- `step2_generate_answers.py` - **步骤2**: LLM答案生成脚本
- `main_qa.py` - 原始的单步流程（使用FAISS检索）

## 使用方法

### 步骤1: 检索文档

```bash
python inference/step1_search_retrieval.py \
    --dataset hotpotqa \
    --expname your_experiment_name \
    --save_dir ./results \
    --sn_prefix y84387018 \
    --k 10 \
    --search_env zc
```

**参数说明**:
- `--dataset`: 数据集名称
- `--expname`: 实验名称
- `--save_dir`: 保存目录
- `--sn_prefix`: **必需**，你的工号前缀（用于搜索API）
- `--k`: 每个子问题检索的文档数量（默认10）
- `--search_env`: 搜索环境，可选 `zc`（众测）或 `effect`（效果环境）
- `--search_debug`: 可选，开启debug模式

**输入文件**（自动生成路径）:
```
{save_dir}/{dataset}/prompts_decompose_test_t0.0_{expname}/generate.jsonl
```

**输出文件**（自动生成路径）:
```
{save_dir}/{dataset}/prompts_decompose_test_{expname}/retrieved_k{k}.jsonl
```

### 步骤2: 生成答案

```bash
python inference/step2_generate_answers.py \
    --input_file ./results/hotpotqa/prompts_decompose_test_exp1/retrieved_k10.jsonl \
    --llm_model_path /path/to/your/llm/model \
    --llm_tokenizer /path/to/your/tokenizer \
    --dataset hotpotqa \
    --k 10 \
    --add_passage 1 \
    --temperature 0.0 \
    --top_p 0.99 \
    --tensor_parallel_size 1
```

**参数说明**:
- `--input_file`: **必需**，步骤1的输出文件
- `--llm_model_path`: **必需**，LLM模型路径
- `--llm_tokenizer`: **必需**，tokenizer路径
- `--dataset`: 数据集名称
- `--k`: 使用的检索文档数量
- `--add_passage`: 是否在最终答案中添加段落（0或1）
- `--temperature`: 采样温度
- `--top_p`: 采样top_p参数
- `--tensor_parallel_size`: 张量并行大小

**输出文件**（自动生成路径）:
```
{input_file_base}_answers.jsonl
```

例如: `retrieved_k10.jsonl` → `retrieved_k10_answers.jsonl`

## 完整示例

```bash
# 步骤1: 检索文档
python inference/step1_search_retrieval.py \
    --dataset hotpotqa \
    --expname exp1 \
    --save_dir ./results \
    --sn_prefix y84387018 \
    --k 10 \
    --search_env zc

# 步骤2: 生成答案
python inference/step2_generate_answers.py \
    --input_file ./results/hotpotqa/prompts_decompose_test_exp1/retrieved_k10.jsonl \
    --llm_model_path /path/to/llama \
    --llm_tokenizer /path/to/llama \
    --dataset hotpotqa \
    --k 10 \
    --add_passage 1
```

## 优势

1. **解耦设计**: 搜索和生成完全独立，便于调试和优化
2. **可重复使用**: 检索结果可以被不同的生成配置重复使用
3. **节省成本**: 不需要重复调用搜索API
4. **灵活性**: 可以单独替换搜索或生成模块
5. **可扩展**: 易于添加新的检索源或生成策略

## 注意事项

⚠️ **重要提示**:
1. `--sn_prefix` 必须使用你自己的工号
2. `--search_env=zc` 只能用于测试，效果环境请谨慎使用
3. 不能多并发运行搜索API
4. 检索结果文件较大，注意磁盘空间

## 输出格式

**步骤1输出** (`retrieved_k10.jsonl`):
```json
{
  "question": "原始问题",
  "decomposed": [...子问题列表...],
  "retrieved_passages": {
    "Q1": {
      "query": "子问题1文本",
      "passages": ["段落1", "段落2", ...]
    },
    "Q2": {...}
  }
}
```

**步骤2输出** (`retrieved_k10_answers.jsonl`):
```json
{
  "question": "原始问题",
  "decomposed": [...子问题列表...],
  "retrieved_passages": {...检索结果...},
  "final_answer": "最终答案",
  "intermediate_answers": {
    "Q1": "答案1",
    "Q2": "答案2"
  },
  "intermediate_passages": {...}
}
```

## 故障排除

### 搜索API调用失败
- 检查 `--sn_prefix` 是否正确
- 确认网络连接和环境配置
- 检查搜索服务是否可用

### 段落提取为空
- 检查 `search_service.py` 中的 `extract_passages_from_search_result()` 函数
- 根据实际API返回格式调整字段提取逻辑
- 可以先运行测试代码查看API返回格式：
  ```bash
  python inference/search_service.py
  ```

### LLM生成错误
- 检查模型路径是否正确
- 确认输入文件包含 `retrieved_passages` 字段
- 调整 `temperature` 和 `top_p` 参数
