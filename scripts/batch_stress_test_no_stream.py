"""
Simple non-stream batch test for /generate.
Run with: conda run -n zimage python scripts/batch_stress_test_no_stream.py
"""
import asyncio
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


async def run_one(session):
    payload = dict(PAYLOAD)
    payload["seed"] = random.randint(0, 999999)
    t0 = time.perf_counter()
    try:
        async with session.post(URL, json=payload, timeout=60) as resp:
            txt = await resp.text()
            dt = time.perf_counter() - t0
            return resp.status == 200, resp.status, dt, txt[:120]
    except Exception as exc:
        return False, None, time.perf_counter() - t0, str(exc)


async def main(total=10):
    results = []
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for i in range(total):
            ok, status, dt, snippet = await run_one(session)
            print(f"[{i+1}/{total}] ok={ok} status={status} time={dt:.2f}s body={snippet}")
            results.append(ok)
    succ = sum(1 for r in results if r)
    fail = len(results) - succ
    print(f"Done: success={succ}, fail={fail}")


if __name__ == "__main__":
    asyncio.run(main(total=10))
