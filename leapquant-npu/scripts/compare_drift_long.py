"""Compare two long-drift dumps: bucketed |dlogprob| curve over context length."""

import argparse
import json

import torch


def load(p):
    xs = json.load(open(p))
    return torch.tensor([x if x is not None else float("nan") for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--bucket", type=int, default=4096)
    args = ap.parse_args()

    a = load(args.reference)
    b = load(args.candidate)
    n = min(len(a), len(b))
    d = (a[:n] - b[:n]).abs()
    print(f"pairs={n}")
    for lo in range(0, n, args.bucket):
        seg = d[lo : lo + args.bucket]
        seg = seg[torch.isfinite(seg)]
        if len(seg) == 0:
            continue
        print(
            f"tokens[{lo:6d}:{lo + len(seg):6d}] mean|d|={seg.mean().item():.5f} "
            f"p99={seg.quantile(0.99).item():.4f} max={seg.max().item():.4f}"
        )
    fin = d[torch.isfinite(d)]
    print(f"overall mean|d|={fin.mean().item():.5f}")


if __name__ == "__main__":
    main()
