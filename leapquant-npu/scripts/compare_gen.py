"""Compare two generation dumps (jsonl with prompt/text/token_ids) token by token."""

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    args = ap.parse_args()

    base = [json.loads(l) for l in open(args.baseline)]
    cand = [json.loads(l) for l in open(args.candidate)]
    assert len(base) == len(cand)
    for i, (b, c) in enumerate(zip(base, cand)):
        bt, ct = b["token_ids"], c["token_ids"]
        n = min(len(bt), len(ct))
        div = next((k for k in range(n) if bt[k] != ct[k]), None)
        match_len = n if div is None else div
        print(
            f"prompt {i}: base={len(bt)} cand={len(ct)} tokens, "
            f"common-prefix={match_len}, "
            + ("IDENTICAL" if div is None and len(bt) == len(ct) else f"first-divergence@{div}")
        )
        if div is not None:
            print(f"  base [{div-2}:{div+3}] = {bt[max(0, div-2):div+3]}")
            print(f"  cand [{div-2}:{div+3}] = {ct[max(0, div-2):div+3]}")


if __name__ == "__main__":
    main()
