from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os
import pandas as pd
from tqdm import tqdm
import copy
import json


parser = argparse.ArgumentParser("中文 Claim 分解脚本")
parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="分词器路径")
parser.add_argument("--model_path", type=str, required=True, help="模型路径")
parser.add_argument("--input_file", type=str, required=True, help="输入的 xlsx 文件路径")
parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
parser.add_argument("--expname", type=str, default="chinese_claim_decompose", help="实验名称")
parser.add_argument("--temperature", type=float, default=0.0, help="生成温度")
parser.add_argument("--top_p", type=float, default=0.99, help="top_p 采样参数")
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPU 并行数量")
parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")

args = parser.parse_args()

# 中文 Claim 分解提示词
prompt_plan_claim_chinese = """请将以下陈述 "{claim}" 分解为多个更小的子陈述，每个子陈述专注于原始陈述的一个具体组成部分，这样更便于模型进行验证。
每个子陈述以 ### 开头。如果需要，可以使用 #1、#2 等来引用前面子陈述的答案。
分解后的陈述："""

# 初始化分词器
print(f"加载分词器: {args.tokenizer}")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

# 采样参数
sampling_params = SamplingParams(
    temperature=args.temperature,
    top_p=args.top_p,
    repetition_penalty=1.05,
    max_tokens=2048
)

# 加载模型
print(f"加载模型: {args.model_path}")
llm = LLM(
    model=args.model_path,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)

# 读取 xlsx 文件
print(f"读取 xlsx 文件: {args.input_file}")
try:
    df = pd.read_excel(args.input_file)
    print(f"成功读取文件，共 {len(df)} 行数据")
    print(f"列名: {df.columns.tolist()}")
except Exception as e:
    print(f"读取文件失败: {e}")
    exit(1)

# 确保至少有3列：第一列(ID)、第二列(claim)、第三列(label)
if len(df.columns) < 3:
    print(f"错误：文件至少需要3列，当前只有 {len(df.columns)} 列")
    exit(1)

# 使用列的位置索引，而不是列名
id_column = df.columns[0]      # 第一列：编号
claim_column = df.columns[1]    # 第二列：claim
label_column = df.columns[2]    # 第三列：人工测评结果（T/F/uncertain）

print(f"第一列（编号）: {id_column}")
print(f"第二列（Claim）: {claim_column}")
print(f"第三列（标签）: {label_column}")

# 准备提示词
prompts = []
contexts = []

print("准备提示词...")
for idx, row in df.iterrows():
    # 跳过 claim 为空的行
    if pd.isna(row[claim_column]):
        continue

    claim = str(row[claim_column]).strip()
    claim_id = row[id_column] if pd.notna(row[id_column]) else idx

    # 处理 label：T/F/uncertain -> 标准格式
    label_value = row[label_column] if pd.notna(row[label_column]) else "uncertain"
    label_value = str(label_value).strip().upper()

    # 转换为标准格式，符合原脚本的要求
    if label_value in ["T", "TRUE", "SUPPORT", "SUPPORTED", "YES"]:
        answer = ["Yes"]
    elif label_value in ["F", "FALSE", "REFUTE", "REFUTED", "NO"]:
        answer = ["No"]
    else:
        answer = ["Uncertain"]

    # 构建提示词
    prompt_tmp = prompt_plan_claim_chinese.replace("{claim}", claim)

    prompts.append([
        {"role": "user", "content": prompt_tmp.strip()}
    ])

    # 保存上下文信息，完全符合原脚本格式
    ctx = {
        "question": claim,      # 必需：原脚本使用 "question" 字段
        "answer": answer,        # 必需：原脚本使用 "answer" 字段（列表格式）
    }
    contexts.append(ctx)

print(f"共准备了 {len(prompts)} 个提示词")

# 创建输出目录
output_dir = os.path.join(args.output_dir, f"decompose_chinese_t{args.temperature}_{args.expname}")
os.makedirs(output_dir, exist_ok=True)
print(f"输出目录: {output_dir}")

# 批量处理
examples = []
print("开始分解 claims...")

for i in tqdm(range(len(prompts)), desc="处理进度"):
    # 应用对话模板
    if "qwen" in args.expname.lower() or "qwen" in args.model_path.lower():
        text = tokenizer.apply_chat_template(
            prompts[i],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    else:
        text = tokenizer.apply_chat_template(
            prompts[i],
            tokenize=False,
            add_generation_prompt=True
        )

    # 根据温度决定采样次数
    N_samples = 1 if args.temperature == 0 else 3

    for j in range(N_samples):
        ctx = copy.deepcopy(contexts[i])

        # 生成分解结果
        outputs = llm.generate([text], sampling_params)
        generated_text = outputs[0].outputs[0].text

        if j == 0 and i % 10 == 0:  # 每 10 个样本打印一次
            print(f"\n样本 {i} 生成结果:")
            print('======')
            print(generated_text)
            print('======\n')

        # 解析分解的子陈述
        decomposed_claims = []
        for line in generated_text.strip().split("\n"):
            line = line.strip()
            # 查找以 ### 开头的行
            if line.startswith("###"):
                try:
                    # 移除 ### 前缀
                    claim_text = line[3:].strip()

                    # 如果有编号格式（如 "### 1. xxx" 或 "### Q1: xxx"）
                    if ":" in claim_text:
                        parts = claim_text.split(":", 1)
                        label = parts[0].strip()
                        text_content = parts[1].strip()
                    else:
                        # 简单的序号
                        label = f"C{len(decomposed_claims) + 1}"
                        text_content = claim_text

                    decomposed_claims.append({
                        "label": label,
                        "text": text_content,
                        "needs_context": True
                    })
                except Exception as e:
                    print(f"解析错误 (样本 {i}): {e}")
                    print(f"问题行: {line}")
                    continue

        # 如果没有解析到子陈述，保存原始文本
        if not decomposed_claims:
            decomposed_claims = [{
                "label": "raw",
                "text": generated_text.strip(),
                "needs_context": True
            }]

        ctx["question_id"] = i
        ctx["decompose_id"] = j
        ctx["decomposed"] = decomposed_claims
        examples.append(ctx)

# 保存结果
print(f"\n保存结果到: {output_dir}")

# 保存为 JSONL 格式
output_file = os.path.join(output_dir, "decomposed_claims.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for example in examples:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"JSONL 格式已保存: {output_file}")

# 同时保存为更易读的 JSON 格式
output_file_json = os.path.join(output_dir, "decomposed_claims.json")
with open(output_file_json, "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print(f"JSON 格式已保存: {output_file_json}")

# 保存为 xlsx 格式，方便查看
print("生成 Excel 输出...")
output_rows = []
for example in examples:
    base_info = {
        "Question ID": example.get("question_id", ""),
        "原始Claim": example["question"],
        "答案": ", ".join(example.get("answer", [])),
        "分解ID": example["decompose_id"]
    }

    # 如果有分解的子陈述
    if example.get("decomposed"):
        for idx, sub_claim in enumerate(example["decomposed"]):
            row = base_info.copy()
            row["子陈述序号"] = idx + 1
            row["子陈述标签"] = sub_claim.get("label", "")
            row["子陈述内容"] = sub_claim.get("text", "")
            output_rows.append(row)
    else:
        output_rows.append(base_info)

output_df = pd.DataFrame(output_rows)
output_excel = os.path.join(output_dir, "decomposed_claims.xlsx")
output_df.to_excel(output_excel, index=False, engine='openpyxl')
print(f"Excel 格式已保存: {output_excel}")

print(f"\n处理完成！")
print(f"总共处理了 {len(contexts)} 条 claim")
print(f"生成了 {len(examples)} 个分解结果")
print(f"所有结果已保存到: {output_dir}")
