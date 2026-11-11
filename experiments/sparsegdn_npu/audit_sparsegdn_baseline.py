from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import statistics
import threading
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer


def request_json(port: int, method: str, path: str, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def wait_ready(port: int, timeout: int) -> dict:
    start = time.perf_counter()
    last_error = None
    while time.perf_counter() - start < timeout:
        try:
            health = request_json(port, "GET", "/health", timeout=5)
            models = request_json(port, "GET", "/v1/models", timeout=5)
            return {
                "ready": True,
                "wait_seconds": time.perf_counter() - start,
                "health": health,
                "models": models,
            }
        except Exception as error:
            last_error = repr(error)
            time.sleep(2)
    return {"ready": False, "wait_seconds": time.perf_counter() - start, "last_error": last_error}


def make_prompts(model: Path, lengths: list[int]) -> dict[int, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    corpus = (model / "README.md").read_text()
    token_ids = tokenizer(corpus, add_special_tokens=False).input_ids
    prompts = {}
    for length in lengths:
        repeats = math.ceil(length / len(token_ids))
        ids = (token_ids * repeats)[:length]
        prompts[length] = ids
    return prompts


def stream_completion(port: int, prompt_ids: list[int], output_tokens: int) -> dict:
    payload = {
        "model": "Qwen3.5-4B",
        "prompt": prompt_ids,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1800)
    start = time.perf_counter()
    first_event = None
    first_nonempty = None
    usage = None
    delta_events = 0
    nonempty_events = 0
    finish_reason = None
    text_parts = []
    error = None
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                if usage is not None:
                    break
                continue
            choice = choices[0]
            if "text" not in choice:
                continue
            if first_event is None:
                first_event = time.perf_counter()
            delta_events += 1
            text = choice.get("text") or ""
            if text:
                if first_nonempty is None:
                    first_nonempty = time.perf_counter()
                nonempty_events += 1
            text_parts.append(text)
            finish_reason = choice.get("finish_reason") or finish_reason
    except Exception as caught:
        error = repr(caught)
    finally:
        connection.close()
    end = time.perf_counter()
    completion_tokens = usage.get("completion_tokens") if usage else None
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    ttft = None if first_event is None else first_event - start
    decode_time = None if first_event is None else end - first_event
    return {
        "success": error is None and usage is not None,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens") if usage else None,
        "expected_prompt_tokens": len(prompt_ids),
        "expected_completion_tokens": output_tokens,
        "ttft_seconds": ttft,
        "first_nonempty_seconds": None if first_nonempty is None else first_nonempty - start,
        "wall_seconds": end - start,
        "decode_after_first_event_seconds": decode_time,
        "output_tokens_per_second_wall": (completion_tokens / (end - start) if completion_tokens else None),
        "decode_output_tokens_per_second": (
            (completion_tokens - 1) / decode_time
            if completion_tokens and completion_tokens > 1 and decode_time
            else None
        ),
        "tpot_after_first_event_seconds": (
            decode_time / (completion_tokens - 1)
            if completion_tokens and completion_tokens > 1 and decode_time
            else None
        ),
        "sse_delta_content_events": delta_events,
        "sse_nonempty_content_events": nonempty_events,
        "usage_accounting_closed": completion_tokens is not None and prompt_tokens == len(prompt_ids),
        "fixed_output_accounting_closed": completion_tokens == output_tokens,
        "finish_reason": finish_reason,
        "generated_text_sha256": hashlib.sha256("".join(text_parts).encode()).hexdigest(),
        "generated_text_characters": sum(map(len, text_parts)),
    }


def summarize_metric(values: list[float]) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {"samples": [], "geometric_mean": None, "sample_count": 0}
    logs = [math.log(value) for value in values]
    mean = statistics.mean(logs)
    result = {
        "samples": values,
        "geometric_mean": math.exp(mean),
        "sample_count": len(values),
    }
    if len(values) > 1:
        interval = 2.776 * statistics.stdev(logs) / math.sqrt(len(values))
        result["log_normal_95_interval"] = [math.exp(mean - interval), math.exp(mean + interval)]
    return result


def run_single(port: int, model: Path, lengths: list[int], repeats: int, output_tokens: int) -> dict:
    prompts = make_prompts(model, lengths)
    warmups = []
    for length in lengths:
        result = stream_completion(port, prompts[length], output_tokens)
        result.update({"length": length, "warmup": True})
        warmups.append(result)
        if not result["success"] or not result["fixed_output_accounting_closed"]:
            return {"ready": True, "warmups": warmups, "passed": False, "blocks": []}
    blocks = []
    for block in range(repeats):
        order = lengths if block % 2 == 0 else list(reversed(lengths))
        records = []
        for length in order:
            result = stream_completion(port, prompts[length], output_tokens)
            result.update(
                {
                    "length": length,
                    "warmup": False,
                    "block": block,
                    "prompt_sha256": hashlib.sha256(
                        torch.tensor(prompts[length], dtype=torch.int32).numpy().tobytes()
                    ).hexdigest(),
                }
            )
            records.append(result)
        blocks.append({"block": block, "records": records})
    per_length = {}
    for length in lengths:
        records = [record for block in blocks for record in block["records"] if record["length"] == length]
        valid = [record for record in records if record["success"]]
        per_length[str(length)] = {
            "records": records,
            "success_count": len(valid),
            "ttft": summarize_metric([record["ttft_seconds"] for record in valid]),
            "wall": summarize_metric([record["wall_seconds"] for record in valid]),
            "output_tokens_per_second_wall": summarize_metric(
                [record["output_tokens_per_second_wall"] for record in valid]
            ),
            "decode_output_tokens_per_second": summarize_metric(
                [record["decode_output_tokens_per_second"] for record in valid]
            ),
        }
    passed = all(
        record["success"] and record["usage_accounting_closed"] and record["fixed_output_accounting_closed"]
        for block in blocks
        for record in block["records"]
    )
    return {
        "schema_version": 1,
        "scope": "910B3 development/cross-validation only; not 910C acceptance",
        "model": str(model),
        "tp": 1,
        "dense": True,
        "profiler": False,
        "graph_mode_requested": "FULL_DECODE_ONLY",
        "prefix_caching": False,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 64,
        "protocol": {
            "warmup_per_length": 1,
            "paired_block_count": repeats,
            "length_order": "alternating forward/reverse",
            "output_tokens": output_tokens,
            "timing": "OpenAI streaming; no profiler",
            "token_accounting": "final usage, not SSE event count",
        },
        "warmups": warmups,
        "blocks": blocks,
        "per_length": per_length,
        "passed": passed,
    }


def metric_values(text: str) -> dict[str, float]:
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            values[name] = float(parts[1])
        except ValueError:
            continue
    return values


def read_metrics(port: int) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            return metric_values(response.read().decode())
    except Exception:
        return {}


def run_mixed(
    port: int,
    model: Path,
    concurrency: int,
    prompt_tokens: int | None,
    output_tokens: int,
) -> dict:
    lengths = (
        [prompt_tokens] * concurrency
        if prompt_tokens
        else [(512, 1024, 2048, 4096)[index % 4] for index in range(concurrency)]
    )
    prompts = make_prompts(model, sorted(set(lengths)))
    stop = threading.Event()
    samples = []

    def poll():
        while not stop.is_set():
            values = read_metrics(port)
            samples.append({"time": time.perf_counter(), "metrics": values})
            stop.wait(0.25)

    polling = threading.Thread(target=poll, daemon=True)
    start = time.perf_counter()
    polling.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(stream_completion, port, prompts[length], output_tokens) for length in lengths]
        records = [future.result() for future in futures]
    wall = time.perf_counter() - start
    stop.set()
    polling.join(timeout=5)
    for index, record in enumerate(records):
        record["request_index"] = index
        record["length"] = lengths[index]
    success = [record for record in records if record["success"]]
    total_output = sum(record["completion_tokens"] or 0 for record in records)
    interesting = {}
    for sample in samples:
        for name, value in sample["metrics"].items():
            if any(key in name for key in ("running", "waiting", "preempt", "prefill", "decode", "success")):
                interesting.setdefault(name, []).append(value)
    return {
        "scope": "scheduler wiring diagnostic, not an SLO/goodput result",
        "requested_concurrency": concurrency,
        "prompt_tokens_override": prompt_tokens,
        "lengths": lengths,
        "wall_seconds": wall,
        "success_count": len(success),
        "failure_count": concurrency - len(success),
        "records": records,
        "actual_usage": {
            "prompt_tokens": sum(record["prompt_tokens"] or 0 for record in records),
            "completion_tokens": total_output,
            "expected_completion_tokens": concurrency * output_tokens,
        },
        "completed_output_tokens_per_second": total_output / wall,
        "metric_series": interesting,
        "raw_metric_sample_count": len(samples),
        "passed": len(success) == concurrency and total_output == concurrency * output_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("wait", "single", "mixed", "profile"))
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--model", type=Path, default=Path("/data/models/Qwen3.5-4B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--profile-length", type=int, default=4096)
    args = parser.parse_args()
    ready = wait_ready(args.port, args.timeout)
    if not ready["ready"] or args.command == "wait":
        result = ready
    elif args.command == "single":
        result = run_single(args.port, args.model, args.lengths, args.repeats, args.output_tokens)
    elif args.command == "mixed":
        result = run_mixed(
            args.port,
            args.model,
            args.concurrency,
            args.prompt_tokens,
            args.output_tokens,
        )
    else:
        prompts = make_prompts(args.model, [args.profile_length])
        request_json(args.port, "POST", "/start_profile", {})
        record = stream_completion(args.port, prompts[args.profile_length], 1)
        stopped = request_json(args.port, "POST", "/stop_profile", {})
        result = {
            "scope": "short profiler window; not formal TTFT/wall",
            "length": args.profile_length,
            "record": record,
            "stop_response": stopped,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"output": str(args.output), "passed": result.get("passed"), "ready": result.get("ready")}, indent=2)
    )
    if result.get("passed") is False or result.get("ready") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
