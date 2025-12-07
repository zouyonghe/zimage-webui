"""
Stress test batch generation via /generate (non-stream) to compare with streaming.
Run after starting the server on 127.0.0.1:9000.
"""
import asyncio
import json
import random
import time

import aiohttp

URL = "http://127.0.0.1:9000/generate"
HEADERS = {"Content-Type": "application/json"}

PAYLOAD = {
    "prompt": "a cat sitting on a chair, high quality, detailed",
    "negative_prompt": "low quality, blurry",
    "steps": 9,
    "guidance": 0.0,
    "height": 512,
    "width": 512,
}


async def run_one(session, seed=None, idx=0):
    payload = dict(PAYLOAD)
    payload["seed"] = seed if seed is not None else random.randint(0, 999999)
    t0 = time.perf_counter()
    try:
        async with session.post(URL, json=payload, timeout=60) as resp:
            txt = await resp.text()
            dt = time.perf_counter() - t0
            ok = resp.status == 200
            return ok, resp.status, dt, txt[:200]
    except Exception as exc:
        return False, None, time.perf_counter() - t0, str(exc)


async def main(total=10):
    results = []
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for i in range(total):
            ok, status, dt, snippet = await run_one(session, idx=i)
            print(f"[{i+1}/{total}] ok={ok} status={status} time={dt:.2f}s body={snippet}")
            results.append(ok)
    succ = sum(1 for r in results if r)
    fail = len(results) - succ
    print(f"Done: success={succ}, fail={fail}")


if __name__ == "__main__":
    asyncio.run(main(total=10))
