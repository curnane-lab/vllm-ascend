import argparse
import concurrent.futures
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import regex as re

DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 48, 64)


def tokens(seed, length):
    return [1000 + ((seed + index * 17) % 90000) for index in range(length)]


def summarize(samples):
    ordered = sorted(samples)
    return {
        "rounds": len(samples),
        "median": statistics.median(samples),
        "p95": ordered[math.ceil(0.95 * len(samples)) - 1],
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def invoke(url, model, rid, input_ids, output_len):
    payload = json.dumps(
        {
            "model": model,
            "prompt": input_ids,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    ).encode()
    request = urllib.request.Request(
        url + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json", "X-Request-Id": rid},
    )
    started = time.perf_counter_ns()
    first_token = None
    output_ids = []
    finish_reason = None
    usage = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data:"):
                continue
            body = raw_line[5:].strip()
            if body == b"[DONE]":
                break
            item = json.loads(body)
            if "error" in item:
                raise RuntimeError(item["error"])
            choices = item.get("choices") or []
            if choices:
                choice = choices[0]
                delta_ids = choice.get("token_ids") or []
                if delta_ids and first_token is None:
                    first_token = time.perf_counter_ns()
                output_ids.extend(delta_ids)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
            if item.get("usage") is not None:
                usage = item["usage"]
    ended = time.perf_counter_ns()
    if finish_reason != "length" or len(output_ids) != output_len:
        raise RuntimeError(f"{rid}: finish_reason={finish_reason}, output_tokens={len(output_ids)}, usage={usage}")
    cached_tokens = None
    if usage is not None:
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "rid": rid,
        "input_tokens": len(input_ids),
        "cached_tokens": cached_tokens,
        "decode_tokens": len(output_ids),
        "ttft_ms": None if first_token is None else (first_token - started) / 1e6,
        "e2e_ms": (ended - started) / 1e6,
        "output_ids": output_ids,
    }


def run_batch(url, model, name, requests, output_len):
    started = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [
            executor.submit(invoke, url, model, f"{name}-{index}", ids, output_len)
            for index, ids in enumerate(requests)
        ]
        records = [future.result() for future in futures]
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    output_tokens = sum(record["decode_tokens"] for record in records)
    return {
        "wall_ms": wall_ms,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens * 1000 / wall_ms,
        "requests": records,
    }


def ensure_port_available(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
    except ConnectionRefusedError:
        return
    except OSError as error:
        raise RuntimeError(f"cannot verify whether port {port} is available: {error}") from error
    raise RuntimeError(f"port {port} is already in use; choose another port")


def wait_ready(proc, url, log_path, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s; see {log_path}")


def stop_server(proc):
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def call_counts(path):
    counts = {
        "pool_prefill": 0,
        "cow_skip": 0,
        "supports_rejected": 0,
        "prepare_calls": 0,
        "prepare_cow_copies": 0,
    }
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "pool_prefill":
            counts["pool_prefill"] += 1
        elif line.startswith("cow_skip:"):
            counts["cow_skip"] += 1
        elif line == "supports:0;op:1":
            counts["supports_rejected"] += 1
        elif line.startswith("prepare:copies="):
            counts["prepare_calls"] += 1
            counts["prepare_cow_copies"] += int(line.split(";", 1)[0].split("=", 1)[1])
    return counts


def graph_capture_metadata(server_log, graph_mode, capture_sizes):
    result = {
        "requested_cudagraph_mode": graph_mode,
        "cudagraph_capture_sizes": capture_sizes,
        "effective_cudagraph_modes_in_log": [],
        "piecewise_capture_observed": False,
        "full_capture_observed": False,
        "graph_capture_finished": False,
        "acl_graph_replay_observed": False,
    }
    if graph_mode == "NONE":
        return result
    log = server_log.read_text(encoding="utf-8")
    replay_ready = list(server_log.parent.glob(f"{server_log.name.removesuffix('.server.log')}.replay.log.*.ready"))
    replay_summaries = list(server_log.parent.glob(f"{server_log.name.removesuffix('.server.log')}.replay.log.*.json"))
    replay_counts = [
        json.loads(path.read_text(encoding="utf-8"))["full_graph_replay_calls"] for path in replay_summaries
    ]
    result.update(
        {
            "effective_cudagraph_modes_in_log": sorted(
                set(re.findall(r"cudagraph_mode['\"]?\s*[:=]\s*<CUDAGraphMode\.([A-Z_]+):", log))
            ),
            "piecewise_capture_observed": "Capturing CUDA graphs (PIECEWISE):" in log,
            "full_capture_observed": "Capturing CUDA graphs (FULL):" in log,
            "graph_capture_finished": "Graph capturing finished" in log,
            "acl_graph_replay_observed": (
                "ACL graph replay is active for the V2 target model" in log
                or bool(replay_ready)
                or any(count > 0 for count in replay_counts)
            ),
            "full_graph_replay_calls": sum(replay_counts) if replay_counts else None,
            "replay_marker_processes": len(replay_ready) + sum(count > 0 for count in replay_counts),
        }
    )
    valid = graph_mode in result["effective_cudagraph_modes_in_log"] and result["graph_capture_finished"]
    if graph_mode == "FULL_AND_PIECEWISE":
        valid = valid and result["piecewise_capture_observed"] and result["full_capture_observed"]
    if not valid:
        raise RuntimeError(f"requested graph capture was not confirmed in {server_log}")
    return result


def main():
    """Launch one server arm, warm it, and measure paired client rounds."""
    parser = argparse.ArgumentParser(
        description="Run a Qwen3.5-4B OpenAI-server A/B for indexed GDN prefill state I/O."
    )
    parser.add_argument("--mode", choices=["legacy", "pool"], required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="/data/f00538480/Qwen3.5-4B")
    parser.add_argument("--served-model-name", default="gdn-pool-test")
    parser.add_argument("--clients", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prefix-tokens", type=int, default=384)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--graph-mode", default="NONE", help="vLLM cudagraph_mode; NONE enforces eager execution")
    parser.add_argument(
        "--capture-sizes", type=int, nargs="+", help="cudagraph_capture_sizes; defaults through client count"
    )
    parser.add_argument("--prefix-match-unit", type=int)
    parser.add_argument("--instrument", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.chunk_size < args.suffix_tokens or args.chunk_size % args.suffix_tokens:
        raise ValueError("chunk size must be a positive multiple of the suffix length")
    if min(args.clients, args.rounds, args.output_tokens) < 1 or args.prefix_tokens < 1:
        raise ValueError("clients, rounds, output tokens and prefix tokens must be positive")
    if args.prefix_match_unit is not None and args.prefix_match_unit < 1:
        raise ValueError("prefix match unit must be positive")
    capture_sizes = args.capture_sizes
    if args.graph_mode == "NONE":
        if capture_sizes is not None:
            raise ValueError("capture sizes require a graph mode")
        capture_sizes = []
    elif capture_sizes is None:
        capture_sizes = [size for size in DEFAULT_CAPTURE_SIZES if size <= args.clients]
        if args.clients not in capture_sizes:
            capture_sizes.append(args.clients)
    if capture_sizes != sorted(set(capture_sizes)) or (capture_sizes and min(capture_sizes) < 1):
        raise ValueError("capture sizes must be unique, positive, and sorted")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    server_log = output.with_suffix(".server.log")
    calls_log = output.with_suffix(".calls.log")
    for path in (server_log, calls_log):
        if path.exists():
            path.unlink()
    for path in server_log.parent.glob(f"{server_log.name.removesuffix('.server.log')}.replay.log.*"):
        path.unlink()
    url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": args.device,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ASCEND_GDN_POOL_IO": "1" if args.mode == "pool" else "0",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3000",
        }
    )
    if args.instrument:
        env["GDN_INSTRUMENT_FILE"] = str(calls_log)
    else:
        env.pop("GDN_INSTRUMENT_FILE", None)
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.35",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--mamba-block-size",
        "128",
        "--mamba-ssm-cache-dtype",
        "bfloat16",
        "--mamba-backend",
        "TRITON",
        "--max-num-batched-tokens",
        str(args.chunk_size),
        "--max-num-seqs",
        str(args.clients),
        "--disable-log-stats",
    ]
    if args.graph_mode == "NONE":
        command.append("--enforce-eager")
    else:
        command.extend(
            [
                "--compilation-config",
                json.dumps({"cudagraph_mode": args.graph_mode, "cudagraph_capture_sizes": capture_sizes}),
            ]
        )
    if args.prefix_match_unit is not None:
        command.extend(["--prefix-match-unit", str(args.prefix_match_unit)])
    ensure_port_available(args.port)
    proc = None
    try:
        with server_log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_ready(proc, url, server_log, args.startup_timeout)

            prefix = tokens(20000, args.prefix_tokens)
            run_batch(
                url,
                args.served_model_name,
                "warm-cold",
                [prefix + tokens(120000, args.suffix_tokens)],
                args.output_tokens,
            )
            run_batch(
                url,
                args.served_model_name,
                "warm-hot",
                [prefix + tokens(130000 + index * 1000, args.suffix_tokens) for index in range(2)],
                args.output_tokens,
            )
            run_batch(
                url,
                args.served_model_name,
                "warm-shape",
                [prefix + tokens(140000 + index * 37, args.suffix_tokens) for index in range(args.clients)],
                args.output_tokens,
            )

            rounds = []
            for round_index in range(args.rounds):
                requests = [
                    prefix
                    + tokens(
                        300000 + round_index * 10000 + index * 37,
                        args.suffix_tokens,
                    )
                    for index in range(args.clients)
                ]
                rounds.append(
                    run_batch(url, args.served_model_name, f"timed-{round_index}", requests, args.output_tokens)
                )
                print(
                    f"{args.mode} chunk={args.chunk_size} trial={args.trial} "
                    f"round={round_index + 1}/{args.rounds} wall={rounds[-1]['wall_ms']:.3f} ms",
                    flush=True,
                )
    finally:
        if proc is not None:
            stop_server(proc)

    result = {
        "mode": args.mode,
        "trial": args.trial,
        "device": args.device,
        "model": args.model,
        "server": {
            "chunked_prefill_size": args.chunk_size,
            "max_num_seqs": args.clients,
            "max_model_len": 4096,
            "mamba_block_size": 128,
            "mamba_ssm_cache_dtype": "bfloat16",
            "mamba_backend": "TRITON",
            "prefix_match_unit": args.prefix_match_unit,
            "enforce_eager": args.graph_mode == "NONE",
            "instrumentation_enabled": args.instrument,
            **graph_capture_metadata(server_log, args.graph_mode, capture_sizes),
        },
        "workload": {
            "clients": args.clients,
            "rounds": args.rounds,
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens_per_request": args.suffix_tokens,
            "output_tokens_per_request": args.output_tokens,
            "warmup": "one cold, two hot, and one full-shape batch",
        },
        "wall_ms": summarize([record["wall_ms"] for record in rounds]),
        "throughput_tokens_per_second": summarize([record["output_tokens_per_second"] for record in rounds]),
        "ttft_ms": summarize(
            [
                request["ttft_ms"]
                for record in rounds
                for request in record["requests"]
                if request["ttft_ms"] is not None
            ]
        ),
        "cached_tokens": sorted(
            {
                request["cached_tokens"]
                for record in rounds
                for request in record["requests"]
                if request["cached_tokens"] is not None
            }
        ),
        "instrumentation": call_counts(calls_log),
        "rounds": rounds,
        "server_log": str(server_log),
        "calls_log": str(calls_log),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rounds"}, indent=2))


if __name__ == "__main__":
    main()
