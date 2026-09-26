#!/usr/bin/env python3
"""Run the HTTP portion of the Item 2 acceptance matrix.

The server is started separately with the desired prefix-cache/prefetch
configuration. This client records repeat-request behavior, n=2 behavior,
and concurrency-8 latency/error results without assuming a text-only output
shape.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, payload: dict, timeout: float) -> tuple[int, bytes, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(), time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), time.perf_counter() - start
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, str(exc).encode(), time.perf_counter() - start


async def one(url: str, payload: dict, timeout: float) -> dict:
    status, body, elapsed = await asyncio.to_thread(request, url, payload, timeout)
    return {
        "status": status,
        "elapsed_s": round(elapsed, 4),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_bytes": len(body),
        "error": None if 200 <= status < 300 else body[:1000].decode(errors="replace"),
    }


async def run(args: argparse.Namespace) -> dict:
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    base = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }
    results: dict = {"base_url": args.base_url, "model": args.model, "prefetch_label": args.prefetch_label}

    results["repeat"] = [await one(url, base, args.timeout) for _ in range(2)]
    n2 = dict(base)
    n2["n"] = 2
    results["n2"] = [await one(url, n2, args.timeout)]

    # Keep the request body identical so the concurrency run exercises the
    # same-prefix path instead of creating a different cache key per request.
    payloads = [dict(base) for _ in range(args.concurrency)]
    started = time.perf_counter()
    results["concurrency"] = {
        "requested": args.concurrency,
        "wall_s": round(time.perf_counter() - started, 4),
        "responses": await asyncio.gather(*(one(url, item, args.timeout) for item in payloads)),
    }
    results["concurrency"]["wall_s"] = round(time.perf_counter() - started, 4)
    responses = results["repeat"] + results["n2"] + results["concurrency"]["responses"]
    results["ok"] = all(item["error"] is None for item in responses)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Reply with exactly: item-2-ok")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prefetch-label", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
