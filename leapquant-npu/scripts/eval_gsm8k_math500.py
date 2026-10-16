#!/usr/bin/env python3
"""GSM8K + MATH-500 evaluation for vLLM 0.30 with reference answers."""

import argparse
import glob
import json
import math
import os
import re
import time

from vllm import LLM, SamplingParams

GSM8K_PATH = "/data1/f00538480/DeepSpec/eval_datasets/gsm8k.jsonl"
MATH500_PATH = "/data1/f00538480/DeepSpec/eval_datasets/math500.jsonl"
GSM_REF_PATH = "/data1/w00842045/logs/evalscope_gsm8k_results/20260908_165231/predictions/qwen_3.5_9B_MLA_SFT_no_think_ckpt776/gsm8k_main.jsonl"
MATH_REF_GLOB = "/data1/w00842045/logs/evalscope_math_500_results/20260908_165231/reviews/qwen_3.5_9B_MLA_SFT_no_think_ckpt776/*.jsonl"

FEWSHOT = [
    {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
    {"role": "assistant", "content": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"},
    {"role": "user", "content": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"},
    {"role": "assistant", "content": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 * 50 = $10.\n#### 10"},
]


def load_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def boxed(text):
    index = text.rfind("\\boxed{")
    if index < 0:
        return None
    depth, start = 1, index + 7
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def extract_answer(text, dataset):
    if dataset == "gsm8k":
        matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return matches[-1] if matches else None
    return boxed(text)


def normalize(value, dataset):
    if value is None:
        return None
    value = str(value).strip().replace(",", "").replace("$", "")
    if dataset == "gsm8k":
        try:
            number = float(value)
            return str(int(number)) if number == int(number) else str(number)
        except ValueError:
            return value
    value = value.replace(" ", "").replace("\\!", "").replace("\\,", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return value or None


def grade(pred, ref, dataset):
    pred, ref = normalize(pred, dataset), normalize(ref, dataset)
    if pred is None or ref is None:
        return False
    if pred == ref:
        return True
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except ValueError:
        return False


def references():
    gsm = {}
    for row in load_jsonl(GSM_REF_PATH):
        question = row["raw_input"]["question"]
        answer = row["raw_input"]["answer"]
        gsm[question] = answer.rsplit("####", 1)[-1].strip()
    math = {}
    for path in glob.glob(MATH_REF_GLOB):
        for row in load_jsonl(path):
            item = row.get("raw_input", {})
            if item.get("problem") and item.get("answer"):
                math[item["problem"]] = item["answer"]
    return gsm, math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/models/Qwen3.5-9B")
    parser.add_argument("--gsm8k-limit", type=int, default=200)
    parser.add_argument("--math500-limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    gsm_refs, math_refs = references()
    gsm_items = load_jsonl(GSM8K_PATH)
    if args.gsm8k_limit:
        gsm_items = gsm_items[: args.gsm8k_limit]
    math_items = load_jsonl(MATH500_PATH)
    if args.math500_limit:
        math_items = math_items[: args.math500_limit]

    messages, refs, datasets = [], [], []
    for item in gsm_items:
        question = item["turns"][0].rsplit("\nPlease reason step by step", 1)[0]
        messages.append(FEWSHOT + [{"role": "user", "content": question}])
        refs.append(gsm_refs.get(question))
        datasets.append("gsm8k")
    for item in math_items:
        problem = item["turns"][0].rsplit("\nPlease reason step by step", 1)[0]
        messages.append([{"role": "user", "content": item["turns"][0]}])
        refs.append(math_refs.get(problem))
        datasets.append("math500")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0, "video": 0},
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    llm.chat([{"role": "user", "content": "Hi"}], sampling, use_tqdm=False,
             chat_template_kwargs={"enable_thinking": False})
    started = time.perf_counter()
    outputs = llm.chat(messages, sampling, use_tqdm=True,
                       chat_template_kwargs={"enable_thinking": False})
    wall = time.perf_counter() - started

    results = {"gsm8k": {"correct": 0, "total": 0, "missing_ref": 0},
               "math500": {"correct": 0, "total": 0, "missing_ref": 0}}
    records = []
    for index, output in enumerate(outputs):
        dataset = datasets[index]
        text = output.outputs[0].text
        pred = extract_answer(text, dataset)
        correct = grade(pred, refs[index], dataset) if refs[index] is not None else False
        results[dataset]["total"] += 1
        results[dataset]["correct"] += correct
        if refs[index] is None:
            results[dataset]["missing_ref"] += 1
        records.append({"index": index, "dataset": dataset, "pred": pred,
                        "ref": refs[index], "correct": correct,
                        "output": text,
                        "prompt_token_ids": list(output.prompt_token_ids),
                        "token_ids": list(output.outputs[0].token_ids)})
    summary = {"tag": args.tag,
               "mode": int(os.getenv("VLLM_ASCEND_GDN_LEAPQUANT", "0")),
               "step_v2": int(os.getenv("VLLM_ASCEND_GDN_LQ_STEP_V2", "0")),
               "wall_s": wall,
               "gsm8k": {"accuracy_pct": 100 * results["gsm8k"]["correct"] / results["gsm8k"]["total"], **results["gsm8k"]},
               "math500": {"accuracy_pct": 100 * results["math500"]["correct"] / results["math500"]["total"], **results["math500"]}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    with open(args.dump, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
