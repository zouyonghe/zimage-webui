"""
Batch test generate_stream with Playwright; mirrors frontend fetch.
Run with: conda run -n zimage python scripts/batch_stress_test.py
Requires: Playwright Chromium installed in zimage env.
"""
import asyncio
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:9000"
TOTAL = 10


async def main():
    print(f"Starting batch stream test to {URL}, total={TOTAL}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda msg: print(f"[browser] {msg.type} {msg.text}"))
        await page.goto(URL, wait_until="domcontentloaded", timeout=15000)
        results = await page.evaluate(
            """
            async (total) => {
              const results = [];
              for (let i = 0; i < total; i += 1) {
                console.log(`running ${i+1}/${total}`);
                const seed = Math.floor(Math.random() * 1000000);
                const payload = {
                  prompt: 'a cat sitting on a chair, high quality, detailed',
                  negative_prompt: 'low quality, blurry',
                  steps: 9,
                  guidance: 0.0,
                  height: 512,
                  width: 512,
                  seed,
                };
                const t0 = performance.now();
                try {
                  const res = await fetch('/generate_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                  });
                  if (!res.ok || !res.body) {
                    results.push({ ok: false, status: res.status, dt: performance.now() - t0, err: 'no body' });
                    continue;
                  }
                  const reader = res.body.getReader();
                  const decoder = new TextDecoder();
                  let buffer = '';
                  let complete = false;
                  while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    if (buffer.includes('\\nevent: complete\\n')) {
                      complete = true;
                      break;
                    }
                  }
                  results.push({ ok: complete, status: res.status, dt: performance.now() - t0, err: complete ? '' : 'closed before complete' });
                } catch (e) {
                  results.push({ ok: false, status: null, dt: performance.now() - t0, err: String(e) });
                }
              }
              return results;
            }
            """,
            TOTAL,
        )
        await browser.close()

    succ = sum(1 for r in results if r.get("ok"))
    fail = len(results) - succ
    print("Batch results:")
    for i, r in enumerate(results, 1):
        print(f"[{i}/{len(results)}] ok={r['ok']} status={r['status']} time={r['dt']:.1f}ms err={r['err']}")
    print(f"Done: success={succ}, fail={fail}")


if __name__ == "__main__":
    asyncio.run(main())
