import argparse
import copy
import json
import re
from typing import List, Dict
from tqdm import tqdm
from vllm.lora.request import LoRARequest

from utils import load_tokenizer, init_llm, make_sampling_params, load_jsonl, save_jsonl, call_llm


def replace_placeholders(question_text: str, answers_so_far: Dict[str, str]) -> str:
    matches = re.findall(r"#(\d+)", question_text)
    for m in matches:
        placeholder = f"#{m}"
        q_key = f"Q{m}"
        if q_key in answers_so_far:
            question_text = question_text.replace(placeholder, answers_so_far[q_key])
    return question_text


def zigzag_visit(lst: List) -> List:
    n = len(lst)
    result = [None] * n

    i, j = 0, 0
    while j < (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j += 1

    i = 1
    j = n - 1
    while j >= (n + 1) // 2:
        result[j] = lst[i]
        i += 2
        j -= 1

    return result


def answer_sub_question(sub_q: str, context_passages: List[str], model, tokenizer, sampling_params, lora_request=None) -> str:
    if not context_passages:
        prompt = f"""Please answer the question '{sub_q}' with a short span.
Your answer needs to be as short as possible."""
    else:
        reordered_passages = zigzag_visit(context_passages)
        context_text = "\n\n".join(reordered_passages)
        prompt = f"""You have the following context passages:
{context_text}

Please answer the question '{sub_q}' with a short span using the context as reference.
If no answer is found in the context, use your own knowledge. Your answer needs to be as short as possible."""

    response = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params, lora_request=lora_request)
    return response.strip()


def generate_final_answer(original_question: str, sub_questions: Dict[str, str], sub_answers: Dict[str, str],
                          model, tokenizer, sampling_params, dataset: str, passages: List[str] = None, add_passage: int = 1, lora_request=None) -> str:
    sub_answer_text = "\n".join([f"### {k}: {sub_questions[k]}, Answer for {k}: {v}" for k, v in sub_answers.items()])
    final_prompt = "a short span"

    if dataset in ["hover", "exfever"]:
        prompt = f"""You are given some subquestions and their answers:
{sub_answer_text}

Please verify the correctness of the claim: '{original_question}' using the subquestions as reference. Please provide a concise and clear reasoning followed by a concise conclusion. Your answer should be Yes or No only.
Wrap your answer with <answer> and </answer> tags."""

    else:
        if add_passage and passages:
            passages = "\n\n".join(list(set(passages)))
            prompt = f"""You have the following passages:
{passages}

You are also given some subquestions and their answers:
{sub_answer_text}

Please answer the question '{original_question}' with {final_prompt} using the documents and subquestions as reference.
Make sure your response is grounded in documents and provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge.
Wrap your answer with <answer> and </answer> tags."""
        else:
            prompt = f"""You are given some subquestions and their answers:
{sub_answer_text}

Please answer the question '{original_question}' with {final_prompt} using the subquestions as reference. Provides clear reasoning followed by a concise conclusion. If no relevant information is found, use your own knowledge.
Wrap your answer with <answer> and </answer> tags."""

    final = call_llm(prompt, model=model, tokenizer=tokenizer, sampling_params=sampling_params, lora_request=lora_request)
    return final.strip()


def process_with_retrieved_passages(item: Dict, llm_model, llm_tokenizer, sampling_params,
                                    dataset: str, add_passage: int, topk: int, lora_request=None) -> Dict:
    question = item["question"]
    sub_questions = item["decomposed"]
    retrieved_passages = item.get("retrieved_passages", {})

    subquestions_dict = {subq_dict["label"]: subq_dict["text"] for subq_dict in sub_questions}
    answer_dict = {}
    passage_dict = {}
    all_passages = []

    for subq_dict in sub_questions:
        q_label = subq_dict["label"]
        q_text = subq_dict["text"]

        q_text_resolved = replace_placeholders(q_text, answer_dict)

        retrieved_info = retrieved_passages.get(q_label, {})
        passages = retrieved_info.get("passages", [])[:topk]

        if len(sub_questions) <= 3:
            all_passages += passages[:5]
        else:
            all_passages += passages[:3]
        all_passages = list(set(all_passages))

        sub_answer = answer_sub_question(q_text_resolved, passages, llm_model, llm_tokenizer, sampling_params, lora_request)

        answer_dict[q_label] = sub_answer
        passage_dict[q_label] = passages
        subquestions_dict[q_label] = q_text_resolved

    final_answer = generate_final_answer(question, subquestions_dict, answer_dict,
                                         llm_model, llm_tokenizer, sampling_params, dataset, all_passages, add_passage, lora_request)

    print("-------\nquestion:", question,
          "\nsub-questions:", sub_questions,
          "\nanswers:", answer_dict,
          "\nFinal Answer:", final_answer)

    return {
        "final_answer": final_answer,
        "intermediate_answers": answer_dict,
        "intermediate_passages": passage_dict
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--llm_model_path", type=str, required=True, help="基座模型路径（需要包含 config.json）")
    parser.add_argument("--llm_tokenizer", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 适配器路径（可选）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.99)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--add_passage", type=int, default=1)
    args = parser.parse_args()

    if args.output_file:
        output_path = args.output_file
    else:
        input_base = args.input_file.replace(".jsonl", "")
        output_path = f"{input_base}_answers.jsonl"

    print("=" * 80)
    print(f"Input: {args.input_file}")
    print(f"Output: {output_path}")
    print(f"LLM: {args.llm_model_path}")
    print(f"Using top-{args.k} documents")
    print("=" * 80)

    items = load_jsonl(args.input_file)
    print(f"Loaded {len(items)} questions")

    print("Initializing LLM...")
    llm_tokenizer = load_tokenizer(args.llm_tokenizer)
    sampling_params = make_sampling_params(args.temperature, args.top_p, max_tokens=512)
    llm = init_llm(args.llm_model_path, args.tensor_parallel_size, lora_path=args.lora_path)
    print("LLM initialized")

    # 如果有 LoRA 路径，创建 LoRARequest
    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest("lora_adapter", 1, args.lora_path)
        print(f"LoRA request created for: {args.lora_path}")

    results = []
    for index, item in enumerate(tqdm(items, desc="Generation")):
        try:
            answer_result = process_with_retrieved_passages(
                item=item,
                llm_model=llm,
                llm_tokenizer=llm_tokenizer,
                sampling_params=sampling_params,
                dataset=args.dataset,
                add_passage=args.add_passage,
                topk=args.k,
                lora_request=lora_request
            )

            result_item = copy.deepcopy(item)
            result_item.update({
                "final_answer": answer_result["final_answer"],
                "intermediate_answers": answer_result["intermediate_answers"],
                "intermediate_passages": answer_result["intermediate_passages"]
            })

            results.append(result_item)

        except Exception as e:
            print(f"Error at index {index}: {e}")
            import traceback
            traceback.print_exc()
            continue

    save_jsonl(results, output_path)
    print("=" * 80)
    print(f"Completed: {len(results)} questions")
    print(f"Saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
