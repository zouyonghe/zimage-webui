"""
Batch test generate_stream with fallback logging.
- Uses Playwright in Node-like fashion to run in-browser fetch to match frontend behavior.
- Prints success/failure count and last error.
Run with: conda run -n zimage python scripts/batch_stress_test.py
Requires: Playwright Chromium installed in zimage env.
"""
import asyncio
import json
import random
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:9000"
TOTAL = 10

JS_TEST = f"""
(async () => {{
  const results = [];
  const doOne = async (i) => {{
    const seed = Math.floor(Math.random() * 1_000_000);
    const payload = {{
      prompt: 'a cat sitting on a chair, high quality, detailed',
      negative_prompt: 'low quality, blurry',
      steps: 9,
      guidance: 0.0,
      height: 512,
      width: 512,
      seed,
    }};
    const t0 = performance.now();
    try {{
      const res = await fetch('/generate_stream', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload),
      }});
      if (!res.ok || !res.body) {{
        return {{ ok: false, status: res.status, dt: performance.now() - t0, err: 'no body' }};
      }}
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let complete = false;
      while (true) {{
        const {{ value, done }} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {{ stream: true }});
        if (buffer.includes('\nevent: complete\n')) {{
          complete = true;
          break;
        }}
      }}
      return {{ ok: complete, status: res.status, dt: performance.now() - t0, err: complete ? '' : 'closed before complete' }};
    }} catch (e) {{
      return {{ ok: false, status: null, dt: performance.now() - t0, err: String(e) }};
    }}
  }};
  for (let i = 0; i < {TOTAL}; i += 1) {{
    results.push(await doOne(i));
  }}
  return results;
}})();
"""

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        await page.goto(URL, wait_until="domcontentloaded", timeout=15000)
        results = await page.evaluate(JS_TEST)
        await browser.close()

    succ = sum(1 for r in results if r.get("ok"))
    fail = len(results) - succ
    print("Batch results:")
    for i, r in enumerate(results, 1):
        print(f"[{i}/{len(results)}] ok={r['ok']} status={r['status']} time={r['dt']:.1f}ms err={r['err']}")
    print(f"Done: success={succ}, fail={fail}")

if __name__ == "__main__":
    asyncio.run(main())
