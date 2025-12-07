"""
Stress test batch generation via /generate_stream to reproduce dropped items.
Run after starting the server on 127.0.0.1:9000.
Uses asyncio + aiohttp; prints success/failure counts and per-request duration.
"""
import asyncio
import json
import random
import time

import aiohttp

URL = "http://127.0.0.1:9000/generate_stream"
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
        async with session.post(URL, json=payload, timeout=15) as resp:
            if resp.status != 200:
                return False, resp.status, time.perf_counter() - t0
            buffer = ""
            async for chunk in resp.content:
                buffer += chunk.decode("utf-8", errors="ignore")
            # Look for complete event
            if "\nevent: complete\n" in buffer:
                return True, resp.status, time.perf_counter() - t0
            return False, resp.status, time.perf_counter() - t0
    except Exception:
        return False, None, time.perf_counter() - t0


async def main(total=10):
    tasks = []
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        for i in range(total):
            # sequential to mimic UI batch loop; change to gather for parallel
            ok, status, dt = await run_one(session, idx=i)
            print(f"[{i+1}/{total}] ok={ok} status={status} time={dt:.2f}s")
            tasks.append(ok)
    succ = sum(tasks)
    fail = len(tasks) - succ
    print(f"Done: success={succ}, fail={fail}")


if __name__ == "__main__":
    asyncio.run(main(total=10))
