from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import os
from tqdm import tqdm
import copy
import json
import pandas as pd


parser = argparse.ArgumentParser("Chinese Claim Decomposition with vLLM")
parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Tokenizer name or path")
parser.add_argument("--model_path", type=str, required=True, help="Model path for vLLM")
parser.add_argument("--expname", type=str, default="chinese_claim_decompose", help="Experiment name")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=0.99, help="Top-p sampling parameter")
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs")
parser.add_argument("--input_file", type=str, required=True, help="Path to input xlsx file")
parser.add_argument("--output_dir", type=str, default="output/decomposed_claims", help="Output directory")
parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum tokens to generate")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")

args = parser.parse_args()

# Chinese prompt for claim decomposition
prompt_plan_claim_chinese = """请将以下声明（claim）分解为多个更小的子声明，每个子声明关注原始声明的一个具体组成部分，以便模型更容易验证。
每个子声明以 ### 开头。如果需要，可以使用 #1、#2 等引用之前子声明的答案。

原始声明："{claim}"

分解后的子声明："""


def load_xlsx_data(file_path):
    """
    Load and process xlsx file
    Expected format: second column should be named 'claim'
    """
    print(f"Loading data from {file_path}...")

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading xlsx file: {e}")
        print("Trying alternative engines...")
        try:
            df = pd.read_excel(file_path)
        except Exception as e2:
            raise ValueError(f"Failed to read xlsx file: {e2}")

    print(f"Loaded {len(df)} rows from xlsx file")
    print(f"Columns: {df.columns.tolist()}")

    # Check if 'claim' column exists
    if 'claim' not in df.columns:
        # Try to use the second column as 'claim'
        if len(df.columns) >= 2:
            second_col = df.columns[1]
            print(f"Warning: 'claim' column not found. Using second column '{second_col}' as claim column.")
            df = df.rename(columns={second_col: 'claim'})
        else:
            raise ValueError(f"'claim' column not found and cannot use second column. Available columns: {df.columns.tolist()}")

    # Remove rows with empty claims
    df = df[df['claim'].notna()]
    df = df[df['claim'].astype(str).str.strip() != '']

    print(f"After filtering empty claims: {len(df)} rows")

    return df


def prepare_prompts(df):
    """
    Prepare prompts for vLLM inference
    """
    prompts = []
    contexts = []

    for idx, row in df.iterrows():
        claim_text = str(row['claim']).strip()

        # Create context for tracking
        item = {
            "row_id": idx,
            "claim": claim_text,
            "original_row": row.to_dict()
        }

        # Prepare prompt
        prompt_tmp = prompt_plan_claim_chinese.replace("{claim}", claim_text)

        prompts.append([
            {"role": "user", "content": prompt_tmp.strip()}
        ])
        contexts.append(item)

    return prompts, contexts


def parse_decomposed_claims(generated_text):
    """
    Parse the generated text to extract decomposed sub-claims
    """
    decomposed_claims = []

    for line in generated_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("###"):
            try:
                # Remove the ### prefix
                claim_text = line.replace("###", "").strip()

                # Extract label if exists (e.g., "1:", "2:", etc.)
                if ":" in claim_text and claim_text.split(":")[0].strip().isdigit():
                    parts = claim_text.split(":", 1)
                    label = f"C{parts[0].strip()}"
                    text = parts[1].strip()
                else:
                    label = f"C{len(decomposed_claims) + 1}"
                    text = claim_text

                decomposed_claims.append({
                    "label": label,
                    "text": text
                })
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {e}")
                continue

    return decomposed_claims


def main():
    # Load data
    df = load_xlsx_data(args.input_file)

    # Prepare prompts
    prompts, contexts = prepare_prompts(df)
    print(f"Prepared {len(prompts)} prompts for inference")

    # Initialize tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Initialize vLLM model
    print(f"Loading vLLM model from {args.model_path}...")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.05,
        max_tokens=args.max_tokens
    )

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process prompts in batches
    print("Starting inference...")
    examples = []
    N_samples = 1 if args.temperature == 0 else 3

    for i in tqdm(range(len(prompts)), desc="Processing claims"):
        # Apply chat template
        if "qwen" in args.model_path.lower() or "qwen" in args.expname.lower():
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

        # Generate decompositions
        for j in range(N_samples):
            ctx = copy.deepcopy(contexts[i])
            outputs = llm.generate([text], sampling_params)
            generated_text = outputs[0].outputs[0].text

            # Print first example for debugging
            if i == 0 and j == 0:
                print("\n" + "="*50)
                print("Example Input Claim:")
                print(ctx['claim'])
                print("\nGenerated Decomposition:")
                print(generated_text)
                print("="*50 + "\n")

            # Parse decomposed claims
            decomposed_claims = parse_decomposed_claims(generated_text)

            # Add to context
            ctx["claim_id"] = i
            ctx["decompose_id"] = j
            ctx["decomposed"] = decomposed_claims
            ctx["raw_generation"] = generated_text

            examples.append(ctx)

    # Save results
    output_file = os.path.join(
        args.output_dir,
        f"decomposed_claims_t{args.temperature}_{args.expname}.jsonl"
    )

    print(f"Saving {len(examples)} decomposed claims to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Also save as xlsx for easy viewing
    output_xlsx = os.path.join(
        args.output_dir,
        f"decomposed_claims_t{args.temperature}_{args.expname}.xlsx"
    )

    print(f"Saving results to xlsx: {output_xlsx}...")
    results_for_excel = []
    for example in examples:
        result_row = {
            "claim_id": example["claim_id"],
            "original_claim": example["claim"],
            "num_subclaims": len(example["decomposed"]),
            "decomposed_text": " | ".join([f"{d['label']}: {d['text']}" for d in example["decomposed"]]),
            "raw_generation": example["raw_generation"]
        }
        results_for_excel.append(result_row)

    results_df = pd.DataFrame(results_for_excel)
    results_df.to_excel(output_xlsx, index=False, engine='openpyxl')

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total claims processed: {len(df)}")
    print(f"Total decompositions generated: {len(examples)}")
    print(f"Average sub-claims per claim: {sum(len(ex['decomposed']) for ex in examples) / len(examples):.2f}")
    print(f"\nResults saved to:")
    print(f"  - JSONL: {output_file}")
    print(f"  - XLSX: {output_xlsx}")
    print("="*50)


if __name__ == "__main__":
    main()
